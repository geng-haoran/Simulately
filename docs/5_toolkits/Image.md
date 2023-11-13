---
sidebar_position: 1
---


# Image
> There are a lot of materials and toolkits for image processing. Here we provide some materials and some code snippets mainly using OpenCV and Pillow.

### Related Materials
- [OpenCV](https://opencv.org/): OpenCV is an open source computer vision and machine learning software library. [Tutorial](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html); [Github](https://github.com/opencv/opencv)
- [Pillow](https://pillow.readthedocs.io/en/stable/): Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

### Read and Write Images
- Using Pillow
  ```python
  from PIL import Image

    # Read an image
    image = Image.open('path_to_image.jpg')

    # Save an image
    image.save('path_to_save_image.jpg')
  ```
- Using OpenCV
  ```python
    import cv2

    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Save an image
    cv2.imwrite('path_to_save_image.jpg', image)
  ```

### Image Resizing
- Using Pillow
  ```python
  # Resize image
    resized_image = image.resize((new_width, new_height))

    # Thumbnail (maintains aspect ratio)
    image.thumbnail((new_width, new_height))
  ```
- Using OpenCV
  ```python
  # Resize image
    resized_image = cv2.resize(image, (new_width, new_height))
  ```

### Image Feature Extraction
- [DINO](https://github.com/facebookresearch/dino): Self-Supervised Vision Transformers with DINO. [Code](https://github.com/facebookresearch/dino); [Paper](https://arxiv.org/abs/2104.14294)
- [DINO v2](https://dinov2.metademolab.com/): DINOv2: Learning Robust Visual Features without Supervision. [Code](https://github.com/facebookresearch/dinov2); [Paper](https://arxiv.org/abs/2304.07193)

### Image Segmentation
- [Segment Anything Model (SAM)](https://segment-anything.com/): SAM is a promptable segmentation system with zero-shot generalization to unfamiliar objects and images, without the need for additional training. [Code](https://github.com/facebookresearch/segment-anything); [Paper](https://arxiv.org/abs/2304.02643)
- [Segmentation Models.pytorch](https://github.com/qubvel/segmentation_models.pytorch): The main features of this library are: (1) High level API (just two lines to create a neural network) (2) 9 models architectures for binary and multi class segmentation (including legendary Unet) (3) 124 available encoders (and 500+ encoders from timm) (4) All encoders have pre-trained weights for faster and better convergence (5) Popular metrics and losses for training routines. [Code](https://github.com/qubvel/segmentation_models.pytorch)

### Open-Vocabulary Image Detection
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): Open-Set Detection. Detect everything with language! [Code](https://github.com/IDEA-Research/GroundingDINO); [Paper](https://arxiv.org/abs/2303.05499);
- [Awesome Open Vocabulary Object Detection](https://github.com/witnessai/Awesome-Open-Vocabulary-Object-Detection): A summary of open-vocabulary object detection papers. [Code](https://github.com/witnessai/Awesome-Open-Vocabulary-Object-Detection)

### Open-Vocabulary Image Segmentation
- [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything):  A very interesting demo by combining Grounding DINO and Segment Anything which aims to detect and segment anything with text inputs! And we will continue to improve it and create more interesting demos based on this foundation.
[Code](https://github.com/IDEA-Research/Grounded-Segment-Anything)