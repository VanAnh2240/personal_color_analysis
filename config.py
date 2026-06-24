"""
config.py
"""

import os



BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")
RESULT_IMG = RESULT_DIR 
SRC_DIR    = os.path.join(BASE_DIR, "src")

os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Dataset 
LAPA_NUM_CLASSES = 11        
LAPA_CLASS_NAMES = [
    "background",   # 0
    "skin",         # 1
    "left_eyebrow", # 2
    "right_eyebrow",# 3
    "left_eye",     # 4
    "right_eye",    # 5
    "nose",         # 6
    "upper_lip",    # 7
    "inner_mouth",  # 8
    "lower_lip",    # 9
    "hair",         # 10
]

PIGMENT_REGIONS = {
    "skin":      1,
    "left_eye":  4,  
    "right_eye": 5,   
    "nose":      6,  
    "hair":      10, 
    "upper_lip": 7,   
    "lower_lip": 9,  
}

IMG_SIZE   = (224, 224) 
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ────────────────────────────────────────────
# Training
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SCHEDULER     = "cosine"  
NUM_WORKERS   = 4
SEED          = 42
K_FOLDS       = 3             

ACTIVE_MODEL  = "clipunet"
DEEPLAB_BACKBONE     = "resnet50"
DEEPLAB_OUTPUT_STRIDE = 16

CLIP_MODEL_NAME  = "ViT-B/16"
CLIP_EMBED_DIM   = 512
UNET_CHANNELS    = [256, 128, 64, 32]  


# K-Means color extraction
KMEANS_CLUSTERS    = 5      
KMEANS_MAX_ITER    = 300
KMEANS_N_INIT      = 10
KMEANS_COLOR_SPACE = "LAB"    

#Season classification
SEASON_RULES = {
    "Spring":  {"warm": True,  "value_min": 6.0, "chroma_min": 4.0},
    "Summer":  {"warm": False, "value_min": 5.0, "chroma_max": 6.0},
    "Autumn":  {"warm": True,  "value_max": 6.5, "chroma_max": 6.5},
    "Winter":  {"warm": False, "value_min": 4.0, "chroma_min": 6.0},
}
WARM_HUE_MIN = 0  
WARM_HUE_MAX = 90


# Checkpoint 
CKPT_DEEPLAB  = os.path.join(CKPT_DIR, "system_1_deeplabv3.pth")
CKPT_CLIPUNET = os.path.join(CKPT_DIR, "system_2_clipunet.pth")


# API APP
RESULT_IMG   = os.path.join(BASE_DIR, "app_img")