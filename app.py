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

# 이미지 
def process_image(input):
    dir = "data/custom/input"

    for idx, img_file in enumerate(input):
        output_path = os.path.join(dir, f"rgb_{idx:03d}.png")
        shutil.copy2(img_file.name, output_path)
    
    return f"Processed {len(input)} images"

def upload(input, num_frames=80):
    try:
        # 예외처리
        if not input:
            return "Please upload a video file or select an image folder"
        
        # 업로드 파일 경로
        dir = "data/custom/input"
        os.makedirs(dir, exist_ok=True)

        # 기존 파일 삭제
        for f in os.listdir(dir):
            file = os.path.join(dir, f)
            if os.path.isfile(file):
                os.unlink(file)
        
        file_path = Path(input.name)
        if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            if len(input) > 1:
                return "Please upload only one video file"
            result = process_video(input[0].name, num_frames)
        else:
            result = process_image(input.name)
        
        # convert.py 실행
        os.system(f"python convert.py -s data/custom")
        return f"{result}\nFiles processed and saved in {dir}"

    except Exception as e:
        return f"Error occured: {str(e)}"

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("# Feature-3DGS Upload Interface")
    gr.Markdown("""
    Select files to upload:
    - For images: Select multiple image files
    - For video: Select a single video file
    
    Supported formats:
    - Video (.mp4, .avi, .mov): Will extract 80 frames at equal intervals
    - Images (.jpg, .jpeg, .png): Will use all selected images
    """)

    file_input = gr.File(
        label="Upload Video or Select Image Folder",
        file_count="multiple",
        file_types=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )
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