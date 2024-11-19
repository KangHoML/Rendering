import os
import shutil
import argparse
import subprocess

def extract_frames(args):
    # 출력 디렉토리
    output_dir = os.path.join(args.output_dir, "input")
    os.makedirs(output_dir, exist_ok=True)

    # ffmpeg를 통해 비디오 변환
    cmd = f'ffmpeg -i {args.video_path} -vf "fps=10" -frame_pts 1 {output_dir}/rgb_%04d.jpg'
    subprocess.run(cmd, shell=True, check=True)

    # 프레임 개수
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith('rgb_')])
    interval = max(len(frames) // args.num_frames, 1)

    # 일정 간격으로 남기고 삭제
    for i, frame in enumerate(frames):
        if i % interval != 0 or i // interval >= args.num_frames:
            os.remove(os.path.join(output_dir, frame))

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="data/custom")
parser.add_argument("--num_frames", type=int, default=80)

if __name__ == "__main__":
    args = parser.parse_args()
    extract_frames(args)
