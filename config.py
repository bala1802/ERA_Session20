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

#LMS DSCRETE SCHEDULER
BETA_START = 0.00085
BETA_END = 0.012
BETA_SCHEDULE = "scaled_linear"
NUM_TRAIN_TIMESTEPS = 1000
