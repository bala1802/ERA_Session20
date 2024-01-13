# ERA_Session20

## Objective:
The purpose of this repository is to understand the architecture of Generative Art & Stable Diffusion

## Repository:
```
.
├── LICENSE
├── README.md
├── config.py
├── diffusion_loss.py
├── image_generator.py
├── inference.ipynb
├── model.py
├── prediction.py
├── requirements.txt
├── symmetry_loss_analysis.py
├── tree.txt
└── utils.py
```

## How to execute this repository?

In `inference.ipynb`, 
    - add the prompt in the `prompt` variable
    - configure the required loss function and execute the prediction function

## Results

`prompt = A King riding a horse`

### 1. Without Loss Function

![Alt text](image.png)

### 2. Blue Channel

Computing the average absolute difference between the `blue channel` values of each pixel in the batch and the target value of `0.9`. This allows us to measure how far, on average the blue channel deviates from the desired value of `0.9` across all images in the batch

![Alt text](image-1.png)