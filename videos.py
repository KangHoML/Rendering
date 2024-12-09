import os
import cv2
import argparse
import subprocess
from pathlib import Path

class VideoCreator:
    def __init__(self, root, iteration, model, fps=10):
        self.root = root
        self.iteration = iteration
        self.model = model
        self.fps = fps
        self.video_dir = os.path.join(root, f"videos_{iteration}")
        os.makedirs(self.video_dir, exist_ok=True)
    
    def create_video(self, src_dir, video_path, imgs=None):
        if imgs is None:
            imgs = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
        else:
            imgs = sorted(imgs)
        
        # ffmpeg even dimensions
        sample = cv2.imread(os.path.join(src_dir, imgs[0]))
        height, width = sample.shape[:2]
        height -= height % 2
        width -= width % 2

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-pattern_type', 'glob',
            '-i', os.path.join(src_dir, '*.png'),
            '-vf', f'scale={width}:{height}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        subprocess.run(cmd, check=True)

    def train_video(self):
        src_dir = os.path.join(self.root, "train", f"ours_{self.iteration}", "renders")
        seg_dir = os.path.join(self.root, f"seg_{self.iteration}", "train")

        if os.path.exists(src_dir):
            rgb_video = os.path.join(self.video_dir, f"train_{self.iteration}.mp4")
            self.create_video(src_dir, rgb_video)
        
        if os.path.exists(seg_dir):
            seg_video = os.path.join(self.video_dir, f"train_seg_{self.iteration}.mp4")
            self.create_video(seg_dir, seg_video)
        
    def novel_video(self):
        src_dir = os.path.join(self.root, "novel_views", f"ours_{self.iteration}", "renders")
        seg_dir = os.path.join(self.root, f"seg_{self.iteration}", "novel_views")

        if self.model == 'lseg':
            masks = [f for f in os.listdir(seg_dir) if f.endswith('.png')]
        elif self.model == 'sam':
            masks = [f for f in os.listdir(seg_dir) if f.endswith('_seg.png')]            

        if os.path.exists(src_dir):
            rgb_video = os.path.join(self.video_dir, f"novel_{self.iteration}.mp4")
            self.create_video(src_dir, rgb_video)
        
        if os.path.exists(seg_dir):
            seg_video = os.path.join(self.video_dir, f"novel_seg_{self.iteration}.mp4")
            self.create_video(seg_dir, seg_video, masks)

parser = argparse.ArgumentParser(description="Create videos from image sequences")
parser.add_argument("--root", type=str, required=True, help="Path including input images")
parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
parser.add_argument("--fps", type=int, default=10, help="Frames per second")
parser.add_argument("--foundation_model", "-f", required=True, choices=['lseg', 'sam'], help="Foundation model type")

if __name__ == "__main__":
    args = parser.parse_args()
    creator = VideoCreator(args.root, args.iteration, args.foundation_model, args.fps)
    creator.train_video()
    creator.novel_video()
