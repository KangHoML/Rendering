import os
import re
import shutil
import subprocess
import gradio as gr

def upload(files):
    # file 없는 경우
    if files is None: return None
    
    # 기존 데이터 삭제 후 새로 생성
    if os.path.exists("data/custom"):
        shutil.rmtree("data/custom")
    os.makedirs("data/custom/input", exist_ok=True)

    # 이미지 서버에 저장
    imgs = []
    for i, img in enumerate(files):
        output_path = os.path.join("data/custom/input", f"rgb_{i:04d}.jpg")
        shutil.copy2(img.name, output_path)
        imgs.append(img.name)
    
    return imgs if imgs else None

def set_rasterizer(encoder_type):
    config_path = "submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/config.h"
    
    # 원본 config.h 내용 읽기
    with open(config_path, 'r') as f:
        config_content = f.read()

    if encoder_type == 'lseg':
        content = re.sub(
            r'#define NUM_SEMANTIC_CHANNELS \d+ .*',
            '#define NUM_SEMANTIC_CHANNELS 128 // LSeg: 512->128->512',
            config_content
        )
    elif encoder_type == 'sam':
        content = re.sub(
            r'#define NUM_SEMANTIC_CHANNELS \d+ .*',
            '#define NUM_SEMANTIC_CHANNELS 64 // SAM: 256->64->256',
            config_content
        )

    with open(config_path, 'w') as f:
        f.write(content)
    
    subprocess.run([
        "pip", "install", 
        "submodules/diff-gaussian-rasterization-feature"
    ])

def train(encoder_type, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting training...")

    # 기존 데이터 삭제 후 새로 생성
    if os.path.exists("output/custom"):
        shutil.rmtree("output/custom")
    os.makedirs("output/custom", exist_ok=True)
    
    # Rasterizer 설정
    progress(0.1, desc="Setting Rasterizer...")
    set_rasterizer(encoder_type)

    # COLMAP
    progress(0.2, desc="Running COLMAP...")
    subprocess.run(["python", "convert.py", "-s", "data/custom"])

    # Encoding
    progress(0.4, desc="Running feature encoding...") 
    cmd = None
    if encoder_type == "lseg":
        cmd = "cd encoders/lseg_encoder && python encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../data/custom/rgb_feature_langseg --test-rgb-dir ../../data/custom/images"
    elif encoder_type == "sam":
        cmd = "cd encoders/sam_encoder && python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input ../../data/custom/images --output ../../data/custom/sam_embeddings"
    subprocess.run(cmd, shell=True)

    # Training
    progress(0.6, desc="Training model...")
    subprocess.run([
        "python", "train.py",
        "-s", "data/custom",
        "-m", "output/custom",
        "-f", encoder_type,
        "--speedup",
        "--iterations", "5000",
    ])

    progress(1.0, desc="Training complete!")
    return "✅ Training completed successfully"  # status message만 반환

def render(encoder_type, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting rendering process...")

    # 렌더링
    progress(0.1, desc="Starting rendering...")
    subprocess.run([
        "python", "render.py",
        "-s", "data/custom",
        "-m", "output/custom",
        "-f", encoder_type, 
        "--iteration", "5000",
        "--novel_view",
        "--skip_test"
    ])

    # Run Segmentation
    progress(0.4, desc="Starting segmentation...")
    if encoder_type == "sam":
        subprocess.run([
            "python", "segment.py",
            "--checkpoint", "checkpoints/sam_vit_h_4b8939.pth",
            "--model-type", "vit_h",
            "--data", "../../output/custom",
            "--iteration", "5000",
            "--skip_test"
        ], cwd="encoders/sam_encoder")
    elif encoder_type == "lseg":
        subprocess.run([
            "python", "segmentation.py",
            "--weights", "demo_e200.ckpt",
            "--data", "../../output/custom",
            "--iteration", "5000"
        ], cwd="encoders/lseg_encoder")
        
    # Video 생성
    progress(0.7, desc="Starting video creation...")
    subprocess.run([
        "python", "videos.py",
        "--root", "output/custom", 
        "-f", encoder_type,
        "--iteration", "5000"
    ])

    # 비디오 경로
    train_rgb = "output/custom/videos_5000/train_5000.mp4"
    train_seg = "output/custom/videos_5000/train_seg_5000.mp4"
    novel_rgb = "output/custom/videos_5000/novel_5000.mp4"
    novel_seg = "output/custom/videos_5000/novel_seg_5000.mp4"

    progress(1.0, desc="Rendering complete!")
    return train_rgb, train_seg, novel_rgb, novel_seg, "✅ Rendering completed successfully"

# UI
with gr.Blocks(css="""
    .scrollable-gallery { max-height: 300px; overflow-y: auto; }
    """) as app:

    gr.Markdown("""
    # Feature 3DGS: Real-time segment-able 3D Renderer
    Upload images or a video to create an segment-able 3D scene representation.
    """)

    # 이미지 업로드
    with gr.Row():
        with gr.Column():
            with gr.Column():
                gallery = gr.File(label="Upload", file_types=["image"], file_count="multiple", height=300, elem_classes="scrollable-gallery")
                encoder = gr.Radio(choices=["lseg", "sam"], label="Encoder Type")
        with gr.Column():
            gallery_preview = gr.Gallery(label="Preview", show_label=False, columns=3, height=400)

    # Train & Render 버튼
    with gr.Row():
        train_btn = gr.Button("Train")
        render_btn = gr.Button("Render")
    status = gr.Textbox(label="Status")

    # 렌더링
    with gr.Tabs() as tabs:
        # Training Results Tab
        with gr.Tab("Training Results"):
            with gr.Row():
                train_rgb = gr.Video(label="RGB Render")
                train_seg = gr.Video(label="Segmentation")
        
        # Novel View Results Tab
        with gr.Tab("Novel View Results"):
            with gr.Row():
                novel_rgb = gr.Video(label="RGB Render")
                novel_seg = gr.Video(label="Segmentation")

    # Event handlers
    gallery.change(
        fn=upload,
        inputs=gallery,
        outputs=gallery_preview
    )

    # Train 버튼 클릭 시
    train_btn.click(
        fn=train,
        inputs=[encoder],
        outputs=[status]
    )

    # Render 버튼 클릭 시
    render_btn.click(
        fn=render,
        inputs=[encoder],
        outputs=[train_rgb, train_seg, novel_rgb, novel_seg, status]
    )

app.launch(share=True)