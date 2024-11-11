import os
import cv2
import shutil
import numpy as np
import gradio as gr
from pathlib import Path

# 비디오 형식 처리
def process_video(input, num_frames):
    dir = "data/custom/input"
    
    vidcap = cv2.VideoCapture(input)
    tot_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(tot_frames // num_frames, 1)

    cnt_cur = 0
    cnt_trg = 0

    while cnt_cur < tot_frames:
        success = vidcap.grab()
        if not success: break

        if cnt_cur % interval == 0 and cnt_trg < num_frames:
            success, img = vidcap.retrieve()
            if success:
                output_path = os.path.join(dir, f"rgb_{cnt_trg:03d}.png")
                cv2.imwrite(output_path, img)
                cnt_trg += 1
        
        cnt_cur += 1
    
    vidcap.release()
    return f"Extracted {cnt_trg} frames from video"

# 이미지 폴더 처리
def process_folder(input, num_frames):
    dir = "data/custom/input"
    ext = {'.jpg', '.jpeg', '.png'}

    # 폴더 내 모든 이미지
    imgs = []
    for root, _, files in os.walk(input):
        for file in files:
            if Path(file).suffix.lower() in ext:
                imgs.append(os.path.join(root, file))
    
    # 균일한 간격으로 선택
    if len(imgs) > num_frames:
        indices = np.linspace(0, len(imgs)-1, num_frames, dtype=int)
        selected = [imgs[i] for i in indices]
    else:
        selected = imgs
    
    # 선택한 이미지 복사
    for idx, img in enumerate(selected):
        output_path = os.path.join(dir, f"rgb_{idx:03d}.jpg")
        shutil.copy2(img, output_path)
    
    return f"Extracted {len(selected)} images from folder"

def upload(input, num_frames=80):
    try:
        # 업로드 파일 경로
        dir = "data/custom/input"
        os.makedirs(dir, exist_ok=True)

        # 기존 파일 삭제
        for f in os.listdir(dir):
            file = os.path.join(dir, f)
            if os.path.isfile(file):
                os.unlink(file)

        # 예외처리
        if not input:
            return "Please upload a video file or select an image folder"
        
        file_path = Path(input.name)
        if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            result = process_video(input.name, num_frames)
        else:
            result = process_folder(input.name, num_frames)
        
        # convert.py 실행
        os.system(f"python convert.py -s data/custom")
        return f"{result}\nFiles processed and saved in {dir}"

    except Exception as e:
        return f"Error occured: {str(e)}"

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("# Feature-3DGS Upload Interface")
    gr.Markdown("""
    Upload options:
    1. Select a folder containing images (will sample 80 images uniformly)
    2. Upload a video file (will extract 80 frames at equal intervals)
    """)

    with gr.Row():
        file_input = gr.File(
            label="Upload Video or Select Image Folder",
            file_count="directory",  # 폴더 업로드 허용
            file_types=["video"]     # 비디오 파일도 허용
        )
    
    with gr.Row():
        process_btn = gr.Button("Process")
    
    output_text = gr.Textbox(label="Status")

    # 처리 버튼 클릭 시
    process_btn.click(
        fn=upload,
        inputs=[file_input],
        outputs=[output_text]
    )

# 서버 실행
demo.launch(share=True)