# PPM Image Convolution Tool

This is a simple C program that reads an image, applies a 3x3 convolution kernel to it (e.g., edge detection), and writes the resulting image back the input format.

## Features

* Supports binary P6 PPM image format and PNG image format.
* Applies a user-defined 3x3 convolution kernel.
* Example: edge detection using the Laplacian kernel.
* Includes helper functions for reading, writing, and inverting images.


## Usage

### Build

Ensure you have `make` and a C compiler installed. Then run:

```bash
make
```

This compiles `main.c` and generates an executable named `conv`.

### Run
```bash
./conv input.ppm output.ppm
```
or 
```bash
./conv input.png output.png
```

## Example

Apply edge detection to an image:

```bash
./conv lena.ppm edges.ppm
```

## Convolution Kernel

The program currently uses the following Laplacian kernel:

```
-1 -1 -1
-1  8 -1
-1 -1 -1
```

The kernel is a global variable above the `convolve` function

## Notes

* The program makes a second copy of the input image to store output data. Be mindful of memory usage on large images.
* Only RGB images are supported; no alpha channel.