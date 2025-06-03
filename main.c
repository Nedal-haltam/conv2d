
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    int max_val;
    unsigned char* data;  // RGB pixel data
} PPMImage;

PPMImage* read_ppm(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Cannot open file");
        return NULL;
    }

    char format[3];
    if (fscanf(fp, "%2s", format) != 1 || strcmp(format, "P6") != 0) {
        printf("Invalid or unsupported PPM format.\n");
        fclose(fp);
        return NULL;
    }

    PPMImage* img = (PPMImage*)malloc(sizeof(PPMImage));

    // Skip comments
    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);

    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max_val);
    fgetc(fp);  // Consume one whitespace after header

    int size = img->width * img->height * 3;
    img->data = (unsigned char*)malloc(size);

    fread(img->data, 1, size, fp);
    fclose(fp);
    return img;
}

void invert_colors(PPMImage* img) {
    int size = img->width * img->height * 3;
    for (int i = 0; i < size; i++) {
        img->data[i] = 255 - img->data[i];
    }
}

int write_ppm(const char* filename, PPMImage* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot write to file");
        return 0;
    }

    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->max_val);
    fwrite(img->data, 1, img->width * img->height * 3, fp);
    fclose(fp);
    return 1;
}

void free_ppm(PPMImage* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

int save_ppm(const char* filename, unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot write PPM");
        return 0;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 3, width * height, fp);  // RGB only
    fclose(fp);
    return 1;
}

unsigned char clamp(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}

int kernel[3][3] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};
void convolve(PPMImage* input, PPMImage* output) {
    int w = input->width;
    int h = input->height;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int px = x + kx - 1; // Adjust for kernel offset
                    int py = y + ky - 1;
                    if (px < 0 || px >= w || py < 0 || py >= h) continue;

                    int idx = 3 * (py * w + px);
                    int k = kernel[ky][kx];
                    r_sum += input->data[idx]     * k;
                    g_sum += input->data[idx + 1] * k;
                    b_sum += input->data[idx + 2] * k;
                }
            }

            int out_idx = 3 * (y * w + x);
            output->data[out_idx]     = (unsigned char)(clamp(r_sum));
            output->data[out_idx + 1] = (unsigned char)(clamp(g_sum));
            output->data[out_idx + 2] = (unsigned char)(clamp(b_sum));
        }
    }
}


void HandlePNG(const char* input_png, const char* output_png) {
    int width, height, channels;
    // Load PNG image using stb_image
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3); // Force RGB (3 channels)
    if (!img) {
        fprintf(stderr, "Failed to load PNG: %s\n", stbi_failure_reason());
        exit(1);
    }

    PPMImage *ppm_img = (PPMImage*)malloc(sizeof(PPMImage));
    if (!ppm_img) exit(1);
    ppm_img->width = width;
    ppm_img->height = height;
    ppm_img->max_val = 255;
    ppm_img->data = img;

    PPMImage *out = (PPMImage*)malloc(sizeof(PPMImage));
    if (!out) exit(1);
    out->width = width;
    out->height = height;
    out->max_val = 255;
    out->data = (unsigned char*)malloc(width * height * 3); // Allocate memory for output image

    convolve(ppm_img, out);

    if (!stbi_write_png(output_png, out->width, out->height, 3, out->data, out->width * 3)) {
        fprintf(stderr, "Failed to write PNG: %s\n", stbi_failure_reason());
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input image> <output image>\n", argv[0]);
        return 1;
    }

    const char* extension = strrchr(argv[1], '.');

    if (strcmp(extension, ".ppm") == 0) {

        PPMImage* img = read_ppm(argv[1]);
        if (!img) return 1;
        PPMImage *out = (PPMImage*)malloc(sizeof(PPMImage));
        if (!out) return 1;
        out->width = img->width;
        out->height = img->height;
        out->max_val = 255;
        out->data = (unsigned char*)malloc(img->width * img->height * 3); // Allocate memory for output image
        if (!out->data) return 1;
        convolve(img, out);
        write_ppm(argv[2], out);
    }
    else if (strcmp(extension, ".png") == 0) 
    {
        HandlePNG(argv[1], argv[2]);
    }
    else {
        printf("Unsupported file format: %s\n", extension);
        return 1;
    }

    printf("Convolution applied and saved to %s\n", argv[2]);
    return 0;
}