import torch
import torchvision.transforms as T
import torch.nn.functional as F

def blue_channel(images):
    error = torch.abs(images[:,2] - 0.9).mean() 
    return error

def elastic_transform(images):
    elastic_transformer = T.ElasticTransform(alpha=550.0,sigma=5.0)
    transformed_imgs = elastic_transformer(images)
    error = torch.abs(transformed_imgs - images).mean()
    return error

def symmetry(images):
    flipped_image = torch.flip(images, [3])
    error = F.mse_loss(images, flipped_image)
    print("Loss Calculated for the Symmetry : ", error)
    return error

def saturation(images):
    transformed_imgs = T.functional.adjust_saturation(images,saturation_factor = 10)
    error = torch.abs(transformed_imgs - images).mean()
    return error