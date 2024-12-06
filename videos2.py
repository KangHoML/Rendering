import os
import cv2
import argparse
import subprocess

def create_video_from_images(image_folder, output_video_file, fps):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width = first_image.shape[:2]

    if width % 2 == 1:
        width -= 1
    if height % 2 == 1:
        height -= 1
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(image_folder, '*.png'),
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_file
    ]
    subprocess.run(cmd)

def main(args: argparse.Namespace) -> None:
   video_folder = os.path.join(args.data, f"videos_{args.iteration}")
   os.makedirs(video_folder, exist_ok=True)

   if args.foundation_model == 'lseg':
       # rendered results
       for res_dir in os.listdir(os.path.join(args.data, "novel_views")):
           if f"ours_{args.iteration}" in res_dir:
               if not any(x in res_dir for x in ['deletion', 'extraction', 'color_func']):
                   # Feature map video
                   fmap_folder = os.path.join(args.data, "novel_views", res_dir, 'feature_map')
                   fmap_video = os.path.join(video_folder, f'{res_dir}_fmap.mp4')
                   create_video_from_images(fmap_folder, fmap_video, args.fps)
               
               # Renders video
               render_folder = os.path.join(args.data, "novel_views", res_dir, 'renders')
               render_video = os.path.join(video_folder, f'{res_dir}.mp4')
               create_video_from_images(render_folder, render_video, args.fps)

       # Segmentation results
       for sub_dir in os.listdir(args.data):
           if f"seg_{args.iteration}" in sub_dir:
               seg_folder = os.path.join(args.data, sub_dir, "novel_views")
               seg_video = os.path.join(video_folder, f'{sub_dir}.mp4')
               create_video_from_images(seg_folder, seg_video, args.fps)

   elif args.foundation_model == 'sam':
       # rendered results
       for res_dir in os.listdir(os.path.join(args.data, "novel_views")):
           if f"ours_{args.iteration}" in res_dir:
               # Feature map video
               fmap_folder = os.path.join(args.data, "novel_views", res_dir, 'feature_map')
               fmap_video = os.path.join(video_folder, f'{res_dir}_fmap.mp4')
               create_video_from_images(fmap_folder, fmap_video, args.fps)
               
               # Renders video
               render_folder = os.path.join(args.data, "novel_views", res_dir, 'renders')
               render_video = os.path.join(video_folder, f'{res_dir}.mp4')
               create_video_from_images(render_folder, render_video, args.fps)

       # Segmentation results
       for sub_dir in os.listdir(args.data):
           if "seg_" in sub_dir:
               for seg_dir in os.listdir(os.path.join(args.data, sub_dir)):
                   seg_folder = os.path.join(args.data, sub_dir, seg_dir)
                   seg_video = os.path.join(video_folder, f'{sub_dir}_{seg_dir.split("_")[-1]}.mp4')
                   create_video_from_images(seg_folder, seg_video, args.fps)

parser = argparse.ArgumentParser(description="Create videos from image sequences")
parser.add_argument("--data", type=str, help="Path including input images")
parser.add_argument("--iteration", type=int, required=True)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--foundation_model", "-f", required=True, type=str)

if __name__ == "__main__":
   args = parser.parse_args()
   main(args)