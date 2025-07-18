import cv2
import os

def split_stereo_video(video_path, output_dir='images'):

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        h, w, _ = frame.shape
        mid = w // 2

        # 分割左右图
        left_img = frame[:, :mid]
        right_img = frame[:, mid:]

        # 保存为jpg
        left_filename = os.path.join(output_dir, f"left_{frame_idx:06d}.jpg")
        right_filename = os.path.join(output_dir, f"right_{frame_idx:06d}.jpg")
        cv2.imwrite(left_filename, left_img)
        cv2.imwrite(right_filename, right_img)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"处理帧数: {frame_idx}")

    cap.release()
    print("处理完成。共处理帧数：", frame_idx)


if __name__ == '__main__':
    # 替换成你自己的mkv文件路径
    video_file = './video/my_video-3.mkv'
    output_dir='images/balls'
    
    os.makedirs(output_dir, exist_ok=True)
    split_stereo_video(video_file, output_dir)
