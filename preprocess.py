import os
import cv2
import argparse

def extract_frames(args):
    # 출력 디렉토리
    output_dir = os.path.join(args.output_dir, "input")
    os.makedirs(output_dir, exist_ok=True)

    # 비디오 캡처
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {args.video_path}")
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(tot_frames // args.num_frames, 1)
    
    cnt_cur, cnt_trg = 0, 0
    try:
        while cnt_cur < tot_frames:
            success = cap.grab()
            if not success: break

            if cnt_cur % interval == 0 and cnt_trg < args.num_frames:
                success, img = cap.retrieve()
                if success:
                    output_path = os.path.join(output_dir, f"rgb_{cnt_trg:03d}.png")
                    cv2.imwrite(output_path, img)
                    cnt_trg += 1
            
            cnt_cur += 1
    
    except Exception as e:
        print(f"\nError during frame extraction: {str(e)}")
        raise
    
    finally:
        cap.release()

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="data/custom")
parser.add_argument("--num_frames", type=int, default=80)


if __name__ == "__main__":
    args = parser.parse_args()
    extract_frames(args)
