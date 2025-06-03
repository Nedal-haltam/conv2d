# PPM Image Convolution Tool

This is a simple C program that reads a binary P6 PPM image, applies a 3x3 convolution kernel to it (e.g., edge detection), and writes the resulting image back in PPM format.

## Features

* Supports binary P6 PPM image format.
* Applies a user-defined 3x3 convolution kernel.
* Example: edge detection using the Laplacian kernel.
* Includes helper functions for reading, writing, and inverting images.

## Usage

### Build

Ensure you have `make` and a C compiler installed. Then run:

```bash
make
```

This compiles `main.c` and generates an executable named `convolve`.

### Run

```bash
./convolve input.ppm output.ppm
```

* `input.ppm`: Path to the source image (must be in binary P6 format).
* `output.ppm`: Path where the convolved image will be saved.

## Example

Apply edge detection to an image:

```bash
./convolve lena.ppm edges.ppm
```

## Convolution Kernel

The program currently uses the following Laplacian kernel:

```
-1 -1 -1
-1  8 -1
-1 -1 -1
```

You can modify the kernel in `main.c` inside the `main()` function.

## File Structure

* `main.c`: Contains all the logic for reading/writing PPM files, applying convolution, and managing memory.
* `Makefile`: Builds the project with standard `make` commands.

## Notes

* The program makes a second copy of the input image to store output data. Be mindful of memory usage on large images.
* Only RGB images are supported; no alpha channel.