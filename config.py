import torch

DEVICE = "mps"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT = 512
WIDTH = 512
GUIDANCE_SCALE = 8
LOSS_SCALE = 200
NUM_INFERENCE_STEPS = 50
BATCH_SIZE = 1

STABLE_DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
SDCONCEPTS = ['<meeg>', '<midjourney-style>', '<moebius>', ' <Marc_Allante>', '<wlop-style>']