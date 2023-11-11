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