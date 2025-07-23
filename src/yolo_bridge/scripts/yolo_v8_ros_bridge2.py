#!/home/xhy/xhy_env/bin/python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 会自动初始化 CUDA 上下文
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from torchvision.ops import nms
import torch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class TRT_YOLOv8:
    def __init__(self, engine_path, input_shape=(1,3,640,640)):
        # 1) 反序列化 engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 2) 分配缓冲区
        self.h_input = np.zeros(input_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        # 假设输出为 [1, N, 5+num_classes]，先估个最大 N（如6300）
        C = 5 + 3  # 若 80 类，可改
        max_det = 6300
        out_shape = (1, max_det, C)
        self.h_output = np.zeros(out_shape, dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # 3) 绑定输入输出
        self.bindings = [int(self.d_input), int(self.d_output)]

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]  # NHWC->NCHW
        return img

    def infer(self, img_bgr):
        # 4) 预处理
        self.h_input[:] = self.preprocess(img_bgr)

        # 5) DtoH
        cuda.memcpy_htod(self.d_input, self.h_input)

        # 6) 执行推理
        self.context.execute_v2(self.bindings)

        # 7) HtoD
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        # h_output 现在是 shape (1, max_det, C)
        return self.h_output[0]

    def postprocess(self, dets, conf_thres=0.25, iou_thres=0.45):
        # dets: np.array [N,5+num_classes]
        obj = dets[:,4:5]
        cls = dets[:,5:]
        scores = (obj * cls.max(1,keepdims=True)[0]).reshape(-1)
        cls_ids = cls.argmax(1)
        mask = scores > conf_thres
        if not mask.any(): return []
        dets, scores, cls_ids = dets[mask], scores[mask], cls_ids[mask]
        # xywh -> x1y1x2y2
        x,y,w,h = dets[:,:4].T
        boxes = torch.tensor(np.stack([x-w/2,y-h/2,x+w/2,y+h/2],1))
        keep = nms(boxes.cuda(), torch.tensor(scores).cuda(), iou_thres).cpu()
        results = []
        for i in keep:
            results.append((boxes[i].cpu().numpy().tolist(),
                            float(scores[i]), int(cls_ids[i])))
        return results

class YOLONode:
    def __init__(self):
        rospy.init_node("yolov8_tensorrt", anonymous=True)
        engine_path = rospy.get_param('~engine_path',
                                    "/home/xhy/catkin_ws/models/shapes_model0719_fp16.trt")
        self.detector = TRT_YOLOv8(engine_path)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/yolov8/target_center",
                                PointStamped, queue_size=1)
        rospy.Subscriber("/left/image_raw", Image, self.cb)
        rospy.loginfo("YOLOv8 TRT Node Ready")

    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        raw = self.detector.infer(img)
        dets = self.detector.postprocess(raw, conf_thres=0.2, iou_thres=0.45)
        for box,score,cls_id in dets:
            u = int((box[0]+box[2])/2)
            v = int((box[1]+box[3])/2)
            pt = PointStamped()
            pt.header = msg.header
            pt.header.frame_id = str(cls_id)
            pt.point.x = u; pt.point.y = v; pt.point.z = score
            self.pub.publish(pt)
        # 可视化同前…

if __name__=="__main__":
    YOLONode()
    rospy.spin()
