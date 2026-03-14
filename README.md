# Stamp Detection - YOLOv9-S

Object detection of stamps on scanned documents.

| | |
|---|---|
| **Task** | Object Detection |
| **Class** | `stamp` (single class, id 0) |
| **Model** | YOLOv9-S (pretrained COCO) |
| **Framework** | [mapo80/YOLO](https://github.com/mapo80/YOLO) (PyTorch Lightning) |
| **Image size** | 320x320 |
| **Dataset** | [mapo80/stamps](https://huggingface.co/datasets/mapo80/stamps) (~6 GB) |

---

## Dataset

Published on HuggingFace: [mapo80/stamps](https://huggingface.co/datasets/mapo80/stamps)

| Split | Total | Positive | Negative | Bounding Boxes |
|-------|-------|----------|----------|---------------|
| Train | 20,857 | 16,503 | 4,354 | 28,101 |
| Val | 3,059 | 2,268 | 791 | 3,851 |
| Test | 3,059 | 2,266 | 793 | 3,825 |
| **Total** | **26,975** | **21,037** | **5,938** | **35,777** |

- **Positive**: documents with stamps (annotated bounding boxes)
- **Negative**: documents without stamps (empty label files, reduce false positives)

### Annotation format

YOLO txt — one `.txt` per image:
```
0 0.5234 0.4123 0.1456 0.1987
0 0.7812 0.6543 0.0874 0.1125
```
`class_id x_center y_center width height` (normalized 0–1). Empty file = no stamps.

### Sources

| # | Dataset | Author | Images | License | Link |
|---|---------|--------|--------|---------|------|
| 1 | Stamp Detection | Stampa | 1,271 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/stampa/stamp-detection-4jhgg) |
| 2 | Stamp Detection | JSam | 4,539 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/jsam/stamp-detection-okgih) |
| 3 | Stamp Recognition | Stamp Project | 4,488 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/stamp-project/stamp-recognition-sdoan/dataset/2) |
| 4 | Stamp | class | 2,289 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/class-cemy0/stamp-wnh42) |
| 5 | Stamp Detection | shujing liang | 2,341 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/shujing-liang/stamp-detection-f3yka) |
| 6 | YOLO-Stamp | Detect and Classify | 1,400 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/detect-and-classify/yolo-stamp-6jf5w) |
| 7 | stamp-individual | swp | 2,992 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/swp-3jks1/stamp-individual) |
| 8 | stamp-shape | swp | 3,000 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/swp-3jks1/stamp-shape/dataset/4) |
| 9 | Stamp Detectation | Marcos | 427 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/marcos-7aslt/stamp-detectation) |
| 10 | Detect Postage Stamp | jackwild | 395 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/jackwildgooglecom/detect-postage-stamp) |
| 11 | StaVer | Tatman / DFKI | 400 | Free | [Kaggle](https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset) |
| 12 | RVL-CDIP (neg) | Harley et al. | 1,060 | Free | [HuggingFace](https://huggingface.co/datasets/chainyo/rvl-cdip) |
| 13 | Tobacco3482 (neg) | Audriaz | 11 | Free | [Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) |
| 14 | FUNSD (neg) | Jaume et al. | 199 | Free | [HuggingFace](https://huggingface.co/datasets/nielsr/funsd) |

---

## Training on Cloud GPU

Copy-paste on a fresh Linux VM (RunPod, Lambda, Vast.ai):

```bash
# 1. System
apt-get update && apt-get install -y unzip wget git
pip install --upgrade pip

# 2. Clone this repo
git clone https://github.com/mapo80/stamps-detection.git
cd stamps-detection

# 3. Install YOLO framework
git clone https://github.com/mapo80/YOLO.git yolo-framework
cd yolo-framework
pip install -r requirements.txt
pip install -e .
cd ..

# 4. Download dataset (~6 GB)
python scripts/prepare_training_dataset.py --output ./data

# 5. Train
cd yolo-framework
python -m yolo.cli fit --config ../config_stamp_gpu.yaml
```

### Resume training

```bash
cd yolo-framework
python -m yolo.cli fit --config ../config_stamp_gpu.yaml \
  --ckpt_path ../runs/stamp_v9s/last.ckpt
```

### TensorBoard

```bash
tensorboard --logdir runs/stamp_v9s_logs --bind_all
```

### Override batch size

```bash
python -m yolo.cli fit --config ../config_stamp_gpu.yaml --data.batch_size=256
```

---

## Training on Apple Silicon (MPS)

```bash
cd yolo-framework
python -m yolo.cli fit --config ../config_stamp.yaml
```

---

## Licenses

Roboflow datasets (1–10): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Other sources: free for research.

**CC BY 4.0 Attribution** — This dataset uses data from: **Stampa**, **JSam**, **Stamp Project**, **class**, **shujing liang**, **DETECT AND CLASSIFY**, **swp**, **Marcos**, **jackwildgooglecom** on [Roboflow Universe](https://universe.roboflow.com/).
