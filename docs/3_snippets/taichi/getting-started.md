---
sidebar_position: 0
---

# Getting Started
Welcome to the Getting Started for using Taichi. In this guide, we will walk through the installation, initialization, basic grammar, and some simple examples of Taichi. The provided code snippets are adapted from the [Taichi Docs](https://docs.taichi-lang.org/docs/) to help you get started quickly.

## Installation
Taichi is available as a PyPI package:
```bash
pip install taichi
```
Simply import Taichi as a regular Python library.
```python
import taichi as ti
```


## Initialize Taichi
```python
ti.init(arch=ti.gpu)
```
The argument `arch` in `ti.init()` specifies the backend that will execute the compiled code. This backend can be either `ti.cpu` or `ti.gpu`. If the `ti.gpu` option is specified, Taichi will attempt to use the GPU backends in the following order: `ti.cuda`, `ti.vulkan`, and `ti.opengl/ti.Metal`. If no GPU architecture is available, the CPU will be used as the backend.

Additionally, you can specify the desired GPU backend directly by setting `arch=ti.cuda`, for example. Taichi will raise an error if the specified architecture is not available. For further information, refer to the [Global Settings](https://docs.taichi-lang.org/docs/global_settings) section in the Taichi documentation.

## Define a Taichi field

Field is a fundamental and frequently utilized data structure in Taichi. It can be considered equivalent to NumPy's `ndarray` or PyTorch's `tensor`, but with added flexibility. For instance, a Taichi field can be [spatially sparse](https://docs.taichi-lang.org/docs/sparse) and can be easily [changed between different data layouts](https://docs.taichi-lang.org/docs/layout). Further advanced features of fields will be covered in [Fields tutorials](https://docs.taichi-lang.org/docs/field).

The function `ti.field(dtype, shape)` defines a Taichi field whose `shape` is of shape and whose elements are of type `dtype`.

```python
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))
```

## Kernels and functions

```python
@ti.func
def complex_sqr(z):  # complex square of a 2D vector
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02
```

The code above defines two functions, one decorated with `@ti.func` and the other with `@ti.kernel`. They are called *Taichi function and kernel* respectively. Taichi functions and kernels are not executed by Python's interpreter but taken over by Taichi's JIT compiler and deployed to your parallel multi-core CPU or GPU, which is determined by the `arch` argument in the `ti.init()` call.

**The main differences between Taichi functions and kernels are as follows:**

- Kernels serve as the entry points for Taichi to take over the execution. They can be called anywhere in the program, whereas Taichi functions can only be invoked within kernels or other Taichi functions. For instance, in the provided code example, the Taichi function `complex_sqr` is called within the kernel `paint`.
- It is important to note that the arguments and return values of kernels must be type hinted, while Taichi functions do not require type hinting. In the example, the argument `t` in the kernel `paint` is type hinted, while the argument `z` in the Taichi function `complex_sqr` is not.
- Taichi supports the use of nested functions, but nested kernels are not supported. Additionally, Taichi does not support recursive calls within Taichi functions.

More details can be found in [Kernels and Functions](https://docs.taichi-lang.org/docs/kernel_function).

## Parallel for loops

The key to achieving high performance in Taichi lies in efficient iteration. By utilizing parallelized looping, data can be processed more effectively.

```python
@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
```

The code snippet above showcases a for loop at the outermost scope within a Taichi kernel, which is automatically parallelized. The for loop operates on the `i` and `j` indices simultaneously, allowing for concurrent execution of iterations.

Taichi provides a convenient syntax for parallelizing tasks. Any for loop at the outermost scope within a kernel is **automatically** parallelized, eliminating the need for manual thread allocation, recycling, and memory management.

It is important to keep in mind that for loops nested within other constructs, such as `if/else` statements or other loops, are not automatically parallelized and are processed *sequentially*.

```python
@ti.kernel
def fill():
    total = 0
    for i in range(10): # Parallelized
        for j in range(5): # Serialized in each parallel thread
            total += i * j

    if total > 10:
        for k in range(5):  # Not parallelized because it is not at the outermost scope
```

<details>
<summary> <a> WARNING: The break statement is not supported in parallelized loops: </a> </summary>

```python
@ti.kernel
def foo():
    for i in x:
        ...
        break # Error!

@ti.kernel
def foo():
    for i in x:
        for j in range(10):
            ...
            break # OK!
```
</details>

## Display the result

To render the result on screen, Taichi provides a built-in GUI System. Use the `gui.set_image()` method to set the content of the window and `gui.show()` method to show the updated image.

```python
gui = ti.GUI("Julia Set", res=(n * 2, n))
# Sets the window title and the resolution

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i += 1
```

## Taichi examples

To view additional selected demos available in the Taichi Gallery:

```python
ti gallery
```
A new window will open and appear on the screen:

<div align="center">
    <a href="https://www.taichi-lang.org/" target="_blank"><img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/taichi-gallery.png"></img></a>
</div>

You can directly run the demo by simply clicking on it.

## Summary

- Taichi compiles and executes Taichi functions and kernels on the designated backend.
- For loops located at the outermost scope in a Taichi kernel are automatically parallelized.
- Taichi offers a flexible data container, known as field, and you can utilize indices to iterate over a field.
