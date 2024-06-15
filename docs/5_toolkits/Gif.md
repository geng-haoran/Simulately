---
sidebar_position: 2
---


# Gif
> Here we give some examples about how to use and edit Gif in Python.

### Related Materials
- [OpenCV](https://opencv.org/): OpenCV is an open source computer vision and machine learning software library. [Tutorial](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html); [Github](https://github.com/opencv/opencv)
- [Pillow](https://pillow.readthedocs.io/en/stable/): Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

### Generate Gif some several images:
Use the following Python code to load a series of images and save them as a GIF:
```python
from PIL import Image

# List of image file paths
image_files = ['image1.png', 'image2.png', 'image3.png']  # Replace with your image paths

# Open images and store them in a list
images = [Image.open(image_file) for image_file in image_files]

# Save as a GIF
output_path = 'output.gif'
images[0].save(
    output_path,
    save_all=True,
    append_images=images[1:],  # Appending all images except the first one
    duration=500,  # Duration of each frame in milliseconds
    loop=0  # Loop forever
)
```

### Compress the GIF:
Use the following Python code to open an existing GIF, compress it, and save the result:
  ```python
from PIL import Image

# Open the GIF file
input_path = 'input.gif'
output_path = 'output_compressed.gif'

with Image.open(input_path) as img:
    # Save the GIF with reduced quality
    img.save(output_path, optimize=True, quality=75)
  ``` 

### Further Compression Options:
For more aggressive GIF compression, you can use a combination of reducing the number of colors, resizing, or using a specialized library like gifsicle. Here's an example of resizing and reducing colors:
  ```python
from PIL import Image

def compress_gif(input_path, output_path, resize_factor=0.5, color_reduction=128):
    with Image.open(input_path) as img:
        # Resize the image
        width, height = img.size
        img = img.resize((int(width * resize_factor), int(height * resize_factor)), Image.ANTIALIAS)
        
        # Reduce the number of colors
        img = img.convert('P', palette=Image.ADAPTIVE, colors=color_reduction)
        
        # Save the GIF with optimization
        img.save(output_path, optimize=True, quality=75)
        
# Paths to the input and output GIFs
input_gif = 'input.gif'
output_gif = 'output_compressed.gif'

# Compress the GIF
compress_gif(input_gif, output_gif)

```