import os
import shutil
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    
    data_root = 'data/custom'
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.makedirs(os.path.join())

    print("\n1. Preprocess the video")
    subprocess.run(["python3", "preprocess.py", "--video_path", args.video_path], check=True)

    print("\n2. Convert to COLMAP format")
    subprocess.run(["python3", "convert.py", "-s", "data/custom"], check=True)

    print("\n3. Run SAM encoder")
    current_dir = os.getcwd()
    os.chdir("encoders/sam_encoder")
    subprocess.run([
        "python3", "export_image_embeddings.py",
        "--checkpoint", "checkpoints/sam_vit_h_4b8939.pth",
        "--model-type", "vit_h",
        "--input", "../../data/custom/images",
        "--output", "../../data/custom/sam_embeddings"
    ], check=True)
    os.chdir(current_dir)

    print("\n4. Train the model")
    subprocess.run([
        "python3", "train.py",
        "-s", "data/custom",
        "-m", "output/custom",
        "-f", "sam",
        "--speedup",
        "--iterations", "5000"
    ], check=True)

    print("\nDone.")