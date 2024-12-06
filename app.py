import os
import shutil
import subprocess
import gradio as gr
from pathlib import Path

def preprocess(files, progress=gr.Progress()):
    progress(0, desc="Starting preprocessing...")

    # 기존 데이터 삭제 후 새로 생성
    if os.path.exists("data/custom"):
        shutil.rmtree("data/custom")
    os.makedirs("data/custom/input", exist_ok=True)
    
    progress(0.2, desc="Processing input files...")
    file_path = Path(files[0].name)
    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        subprocess.run([
                "python", "preprocess.py",
                "--video_path", file_path,
                "--output_dir", "data/custom"
            ])
    else:
        for i, img in enumerate(files):
            output_path = os.path.join("data/custom/input", f"rgb_{i:04d}.jpg")
            shutil.copy2(img.name, output_path)
    
    # COLMAP 실행
    progress(0.6, desc="Running COLMAP...")
    subprocess.run(["python", "convert.py", "-s", "data/custom"])
    
    progress(1.0, desc="Preprocessing complete!")
    return "✅ Preprocessing completed successfully"

def train(encoder_type, progress=gr.Progress()):
    progress(0, desc="Starting training...")

    # 기존 데이터 삭제 후 새로 생성
    if os.path.exists("output/custom"):
        shutil.rmtree("output/custom")
    os.makedirs("output/custom", exist_ok=True)

    # Encoding
    progress(0.2, desc="Running feature encoding...")
    cmd = None
    if encoder_type == "lseg":
        cmd = "cd encoders/lseg_encoder && python encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../data/custom/rgb_feature_langseg --test-rgb-dir ../../data/custom/images"
    elif encoder_type == "sam":
        cmd = "cd encoders/sam_encoder && python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input ../../data/custom/images --output ../../data/custom/sam_embeddings"
    subprocess.run(cmd, shell=True)

    # Training
    progress(0.4, desc="Training model...")
    subprocess.run([
        "python", "train.py",
        "-s", "data/custom",
        "-m", "output/custom",
        "-f", encoder_type,
        "--speedup",
        "--iterations", "5000"
    ])

    progress(1.0, desc="Training complete!")
    return "✅ Training completed successfully"

def render_video(encoder_type, progress=gr.Progress()):
    progress(0, desc="Starting rendering...")

    # novel_view rendering
    progress(0.3, desc="Rendering novel views...")
    subprocess.run([
        "python", "render.py",
        "-s", "data/custom",
        "-m", "output/custom",
        "-f", encoder_type,
        "--iteration", "5000",
        "--novel_view"
    ])

    # Run Segmentation
    progress(0.5, desc="Generating segmentation...")
    if encoder_type == "sam":
        subprocess.run([
            "python", "encoders/sam_encoder/segment.py",
            "--checkpoint", "encoders/sam_encoder/checkpoints/sam_vit_h_4b8939.pth",
            "--model-type", "vit_h",
            "--data", "output/custom",
            "--iteration", "5000"
        ])
    elif encoder_type == "lseg":
        subprocess.run([
            "python", "encoders/lseg_encoder/segmentation.py",
            "--data", "output/custom",
            "--iteration", "5000"
        ])

    # video 생성
    progress(0.7, desc="Creating videos...")
    subprocess.run([
        "python", "videos2.py",
        "--data", "output/custom",
        "-f", encoder_type,
        "--iteration", "5000"
    ])

    # RGB와 feature map video
    rgb_video = "output/custom/videos_5000/ours_5000.mp4"
    if encoder_type == "lseg":
        seg_video = "output/custom/videos_5000/seg_5000.mp4"
    elif encoder_type == "sam":
        seg_video = "output/custom/videos_5000/seg_5000_noprompt.mp4"

    if not os.path.exists(rgb_video) or not os.path.exists(seg_video):
        return None, None, "❌ Error generating videos"
    
    progress(1.0, desc="Rendering complete!")
    return rgb_video, seg_video, "✅ Videos generated successfully"

with gr.Blocks() as app:
    gr.Markdown("""
    # Feature 3DGS: Real-time segment-able 3D Renderer
    Upload images or a video to create an segment-able 3D scene representation.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_files = gr.File(
                label="Upload Images/Video",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]
            )
            encoder = gr.Radio(
                choices=["lseg", "sam"],
                label="Encoder Type",
                info="Select the foundation model to use"
            )
        
        with gr.Column(scale=1):
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    with gr.Row():
        process_btn = gr.Button("1️⃣ Preprocess Input", variant="primary")
        train_btn = gr.Button("2️⃣ Train Model", variant="primary")
        render_btn = gr.Button("3️⃣ Generate Videos", variant="primary")
    
    with gr.Row():
        with gr.Column():
            rgb_video = gr.Video(label="RGB Video")
        with gr.Column():
            seg_video = gr.Video(label="Segmentation Video")


    # Event handlers
    process_btn.click(
        preprocess,
        inputs=[input_files],
        outputs=[status_text]
    )
    
    train_btn.click(
        train,
        inputs=[encoder],
        outputs=[status_text]
    )
    
    render_btn.click(
        render_video,
        inputs=[encoder],
        outputs=[rgb_video, seg_video, status_text]
    )

app.launch(share=True)