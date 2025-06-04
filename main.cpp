
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    int max_val;
    unsigned char* data;
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

    
    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);

    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max_val);
    fgetc(fp);  

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
    fwrite(data, 3, width * height, fp);  
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
int kRows = 3;
int kCols = 3;
void conv2d(PPMImage* input, PPMImage* output) {
    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int px = x + kx - 1; 
                    int py = y + ky - 1;
                    if (!(px < 0 || px >= input->width || py < 0 || py >= input->height))
                    {
                        int k = kernel[ky][kx];
                        r_sum += input->data[(3 * (py * input->width + px)) + 0] * k;
                        g_sum += input->data[(3 * (py * input->width + px)) + 1] * k;
                        b_sum += input->data[(3 * (py * input->width + px)) + 2] * k;
                    }
                }
            }
            output->data[(3 * (y * input->width + x)) + 0] = clamp(r_sum);
            output->data[(3 * (y * input->width + x)) + 1] = clamp(g_sum);
            output->data[(3 * (y * input->width + x)) + 2] = clamp(b_sum);
        }
    }
}
#include <vector>
std::vector<int> conv1d(std::vector<int>& a, std::vector<int>& b)
{
    std::vector<int> result(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}
/*
variables: Input, kernel, Output
indexes: x, y, kx, ky
we can continue as if the output is in the most inner loop and the rgb_sum are just temporary, and we can clamp properly in the hardware



systolic steps:
    - for each variable: Dependence matrices, NSBVs
    - find the scheduling vectors for the all the possible variables configurations (i.e. pipelined/broadcasted for each variable)
    - for each scheduling vector obtain all its PDVs
    - for each PDV obtain its projection matrix
        - scheduling function
        - projected NSBVs to know along which axis is it pipelined/broadcasted
    - read the last part of the systolic-arrays paper in overleaf for:
        - delay registers
        - input feeding point
        - output initialization point and extraction point 
*/

void HandlePNG(const char* input_png, const char* output_png) {
    int width, height, channels;
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3);
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
    out->data = (unsigned char*)malloc(width * height * 3);

    conv2d(ppm_img, out);

    if (!stbi_write_png(output_png, out->width, out->height, 3, out->data, out->width * 3)) {
        fprintf(stderr, "Failed to write PNG: %s\n", stbi_failure_reason());
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {0, 2, 0};
    std::vector<int> result = conv1d(a, b);
    printf("Convolution result: ");
    for (int val : result) {
        printf("%d ", val);
    }
    return 0;
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
        out->data = (unsigned char*)malloc(img->width * img->height * 3);
        if (!out->data) return 1;
        conv2d(img, out);
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