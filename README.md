# Neural Style Transfer with VGG19

This project demonstrates the use of neural style transfer (NST) using a pre-trained VGG19 model to blend the content of one image with the style of another.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Explanation](#explanation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as a painting)—and blend them together so that the output image looks like the content image, but "painted" in the style of the style image. This project uses the VGG19 network(a pre-trained network) to perform NST.

## Features

- Load and preprocess content and style images.
- Extract features using a pre-trained VGG19 network.
- Compute content and style losses.
- Optimize the generated image to minimize these losses.
- Display the intermediate and final results.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib


Explanation
Preprocessing
Images are loaded, converted to float32, and resized while maintaining their aspect ratios.

VGG19 Model
The VGG19 network is used for feature extraction. We use specific layers to extract content and style features.

Loss Calculation
Content Loss: Measures how much the content in the generated image differs from the content image.
Style Loss: Measures how much the style in the generated image differs from the style image using the Gram matrix.
Total Variation Loss: Encourages spatial smoothness in the generated image to reduce noise.
Optimization
An Adam optimizer is used to iteratively update the generated image to minimize the total loss.

Results
Example output images will be shown here after the script runs. You can compare the content image, style image, and the final output image.

Acknowledgements
The VGG19 model used in this project is pre-trained on the ImageNet dataset.
The neural style transfer technique is based on the paper "A Neural Algorithm of Artistic Style" by Gatys et al.

