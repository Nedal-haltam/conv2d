
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "raylib/include/raylib.h"

int K = 8;
#define MAX_ITER 30

typedef struct
{
    float r;
    float g;
    float b;
} Colorf;

typedef struct {
    Colorf color;
    int index;
} Point;

float color_distance(Colorf a, Colorf b) {
    return sqrtf((a.r - b.r)*(a.r - b.r) + (a.g - b.g)*(a.g - b.g) + (a.b - b.b)*(a.b - b.b));
}

// this returns the index of the closest centroid
int ClosestCluster(Colorf color, Colorf *centroids) {
    int index = 0;
    float min_dist = color_distance(color, centroids[0]);

    for (int i = 1; i < K; ++i) {
        float dist = color_distance(color, centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

void Cluster(Colorf* Centroids, Point* Points, int count)
{
    // Assign pixels to nearest centroid
    for (int i = 0; i < count; ++i) {
        Points[i].index = ClosestCluster(Points[i].color, Centroids);
    }

    // Recompute Centroids
    int* CentroidCount = (int*)malloc(sizeof(int) * K);
    Colorf* NewCentroids = (Colorf*)malloc(sizeof(Colorf) * K);
    for (int i = 0; i < K; i++)
    {
        CentroidCount[i] = 0;
        NewCentroids[i] = (Colorf){0};
    }

    for (int i = 0; i < count; ++i) {
        int cluster = Points[i].index;
        NewCentroids[cluster].r += Points[i].color.r;
        NewCentroids[cluster].g += Points[i].color.g;
        NewCentroids[cluster].b += Points[i].color.b;
        CentroidCount[cluster]++;
    }

    for (int i = 0; i < K; ++i) {
        if (CentroidCount[i] > 0) {
            Centroids[i].r = NewCentroids[i].r / CentroidCount[i];
            Centroids[i].g = NewCentroids[i].g / CentroidCount[i];
            Centroids[i].b = NewCentroids[i].b / CentroidCount[i];
        }
    }    
    free(CentroidCount);
    free(NewCentroids);
}

void Initialize(Point* Points, Colorf* centroids, uint32_t *image, int count)
{
    // decompose image -> colors
    for (int i = 0; i < count; ++i) {
        Points[i].color.r = (image[i] >> 8*0) & 0xFF;
        Points[i].color.g = (image[i] >> 8*1) & 0xFF;
        Points[i].color.b = (image[i] >> 8*2) & 0xFF;
    }
    
    // Initialize centroids randomly
    for (int i = 0; i < K; ++i) {
        int rand_index = rand() % count;
        centroids[i] = Points[rand_index].color;
    }
}

void kmeans_quantization(uint32_t *image, int Width, int Height) {
    int Pixels = Width * Height;
    Point* Points = (Point*)malloc(sizeof(Point) * Pixels);
    Colorf* centroids = (Colorf*)malloc(sizeof(Colorf)*K);
    for (int i = 0; i < K; i++)
    {
        centroids[i] = (Colorf){0};
    }

    Initialize(Points, centroids, image, Pixels);
    // run k-means
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        Cluster(centroids, Points, Pixels);
    }

    for (int i = 0; i < Pixels; ++i) {
        Colorf c = centroids[Points[i].index];
        image[i] = (((int)c.r) << 8*0) | (((int)c.g) << 8*1) | (((int)c.b) << 8*2) | 0xFF000000;
    }
    free(centroids);
    free(Points);
}

int w = 800;
int h = 600;
float scale = 0.5f;
#define BACKGROUND_COLOR ((Color){ .r = 0x20, .g = 0x20, .b = 0x20, .a = 0xFF})

Image meLoadImage(const char* FilePath)
{
    int width = 0, height = 0, n = 0;
    Image image = {
        .data = (void*)stbi_load(FilePath, &width, &height, &n, 4),
        .mipmaps = 1,
        .width = width,
        .height = height,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    return image;
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("Usage: %s input_Image output_Image\n", argv[0]);
        return 1;
    }
    const char* ImageFilePath = argv[1];
    const char* OutputFilePath = argv[2];
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
    InitWindow(w, h, "Color Quantization");
    SetTargetFPS(60);
    Image image = meLoadImage(ImageFilePath);
    Texture2D texture = LoadTextureFromImage(image);
    while (!WindowShouldClose())
    {
        w = GetScreenWidth();
        h = GetScreenHeight();
        if (IsKeyPressed(KEY_Q))
        {
            UnloadImage(image);
            image = meLoadImage(ImageFilePath);
            kmeans_quantization((uint32_t*)image.data, image.width, image.height);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(image);
        }
        if (IsKeyPressed(KEY_S))
        {
            if (!ExportImage(image, OutputFilePath))
            {
                fprintf(stderr, "Could not save image\n");
            }
        }
        if (IsKeyPressed(KEY_RIGHT))
            K++;
        if (IsKeyPressed(KEY_LEFT))
            K--;
        BeginDrawing();
        ClearBackground(BACKGROUND_COLOR);
        DrawTexturePro(texture, (Rectangle) { .x = 0, .y = 0, .width = texture.width, .height = texture.height}, (Rectangle) { .x = w / 2 - scale*texture.width / 2, .y = h / 2 - scale*texture.height / 2, .width = scale*texture.width, .height = scale*texture.height}, (Vector2) {.x = 0, .y = 0}, 0, WHITE);
        DrawFPS(0, 0);
        EndDrawing();
    }
    UnloadImage(image);
    UnloadTexture(texture);
    CloseWindow();
    return 0;
}
