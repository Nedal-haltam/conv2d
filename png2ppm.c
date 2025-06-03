#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s input.png output.ppm\n", argv[0]);
        return 1;
    }

    const char* input_png = argv[1];
    const char* output_ppm = argv[2];

    int width, height, channels;
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3); // Force RGB (3 channels)
    if (!img) {
        fprintf(stderr, "Failed to load PNG: %s\n", stbi_failure_reason());
        return 1;
    }

    if (!save_ppm(output_ppm, img, width, height)) {
        stbi_image_free(img);
        return 1;
    }

    printf("Saved PPM: %s\n", output_ppm);
    stbi_image_free(img);
    return 0;
}