# Pytorch Implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) & Module Implementation

# Preparation
Download and extract `data` folder from [link](https://drive.google.com/file/d/1u9EIGj0z6LspZcj2wpStIqdLYsNwC6Bl/view?usp=sharing), and place it in the root folder of this repo.

# Tests Pass Screenshots
### Using GPU
![Screenshot](https://i.ibb.co/VJTbsFg/image.png)

### Using CPU
![Screenshot](https://i.ibb.co/Ld9B4Y7/image.png)

## Special Thanks
We thank VITA Lab for providing clusters on `izar` for training our models in Miniproject 2!

## Follow up fixes
1. Fixed `Sequential` backpropagation and parameter saving to resolve potential issues that could arise when building such a structure.
```
self.model = Sequential(
    Sequential(
        Conv2d(3, 32, 3, stride=2, padding=2),
        ReLU(),
        Conv2d(32, 64, 3, stride=2, padding=2),
        ReLU()),
    Upsampling(2, 64, 32, kernel_size=4, stride=1),
    Sequential(
        ReLU(),
        Upsampling(2, 32, 3, stride=1, kernel_size=3),
        Sigmoid())
).to(self.device)
```
