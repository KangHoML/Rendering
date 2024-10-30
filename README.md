## **Feature 3DGS**

### **Setting**

- CMakeLists.txt 파일 수정

    ```
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui ximgproc optflow)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    add_definitions(${OpenCV_DEFINITIONS})
    ```

- OpenCV 경로 지정
    ```
    cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
    cmake --build build -j24 --target install
    ```


### **Dataset**
- Dataset Structure
    ```
    <location>
        ├── input/  # 원본 이미지
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── images/  # 왜곡 제거 이미지
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── images_2/  # 1/2 scaling
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── images_4/  # 1/4 scaling
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── images_8/  # 1/8 scaling
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── sparse/  # SFM 정보 (convert.py)
        │   └── 0/
        │       ├── cameras.bin
        │       ├── images.bin
        │       └── points3D.bin
        └── rgb_feature_langseg/  # 또는 sam_embeddings/
            ├── image001_feature.npy
            ├── image002_feature.npy
            └── ...
    ```

- Replica Dataset
    https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view

### **Train**
- Semantic Channel 수정 및 Resterizor 재설치
    ```
    vim submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/config.h
    NUM_SEMANTIC_CHANNELS = 128 # lseg: 512 (speed: 128) / sam: 256 (speed: 64)
    cd submodules/diff-gaussian-rasterization-feature
    pip install .
    ```

- Train 코드
    ```
    python train.py -s data/Replica/office3 -m output/Replica/office3 -f lseg --speedup --iterations 7000
    ```

- 훈련된 모델 확인
    ```
    ./SIBR_viewers/install/bin/SIBR_remoteGaussian_app &
    python view.py -s data/Replica/office3 -m output/Replica/office3 -f lseg --ip "0.0.0.0"
    ```

### **Rendering**
- 렌더링
    ```
    python3 render.py -s data/Replica/office3 -m output/Replica/office3 -f lseg --iteration 7000
    ```

- Novel View
    ```
    python3 render.py -s data/Replica/office4 -m output/Replica/office4 -f lseg --iteration 7000 --novel_view
    ```

- Editting 포함
    ```
    python render.py -s data/Replica/office3 -m output/Replica/office3 -f lseg --iteration 7000 --edit_config configs/edit_color.yaml
    ```

- 비디오 생성
    ```
    python3 videos.py --data output/Replica/office3 --fps 10 -f lseg --iteration 7000 
    ```

### **Inference**
- 기존 Segmentation 활용 시
    
    [ADE20K](https://ade20k.csail.mit.edu/) 중 150개의 label을 활용하여 segment
    ```
    python -u segmentation.py --data ../../output/Replica/room0/ --iteration 6000
    ```

- Custom Segmentation 활용 시

    ```
    python -u segmentation.py --data ../../output/Replica/room0/ --iteration 6000 --label_src car,building,tree
    ```

- Segmantation Metrix
    ```
    cd encoders/lseg_encoder
    python -u segmentation_metric.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --student-feature-dir ../../output/Replica/room0/test/ours_30000/saved_feature/ --teacher-feature-dir ../../data/Replica/room0/rgb_feature_langseg/ --test-rgb-dir ../../output/Replica/room0/test/ours_30000/renders/ --workers 0 --eval-mode test
    ```
    
