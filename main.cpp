
#include <iostream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <deque>

bool verbose = false;
clock_t Clocker;
void StartClock()
{
    Clocker = clock();
}

double EndClock(bool Verbose = false)
{
    clock_t t = clock() - Clocker;
    double TimeTaken = (double)(t) / CLOCKS_PER_SEC;
    if (Verbose)
    {
        std::cout << "Time taken: " << std::fixed << std::setprecision(8) << TimeTaken << "s\n";
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);
    return TimeTaken;
}

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
        std::cout << "Unsupported PPM format. Only P6 is supported.\n";
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

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_DEPTH 3

int KERNEL2D_EDGE_DETECTOR[KERNEL_HEIGHT][KERNEL_WIDTH] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

int KERNEL2D_IDENTITY[KERNEL_HEIGHT][KERNEL_WIDTH] = {
    {0, 0, 0},
    {0, 1, 0},
    {0, 0, 0},
};

int (*k2d)[3] = KERNEL2D_EDGE_DETECTOR;

void conv2d(PPMImage* input, PPMImage* output)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int px = x + kx - 1; 
                    int py = y + ky - 1;
                    if (!(px < 0 || px >= input->width || py < 0 || py >= input->height))
                    {
                        int k = k2d[ky][kx];
                        // int k = KERNEL_IDENTITY[ky][kx];
                        r_sum += input->data[(3 * (py * input->width + px)) + 0] * k;
                        g_sum += input->data[(3 * (py * input->width + px)) + 1] * k;
                        b_sum += input->data[(3 * (py * input->width + px)) + 2] * k;
                    }
                }
            }
            output->data[(3 * (y * input->width + x)) + 0] = clamp(abs(r_sum));
            output->data[(3 * (y * input->width + x)) + 1] = clamp(abs(g_sum));
            output->data[(3 * (y * input->width + x)) + 2] = clamp(abs(b_sum));
        }
    }
}
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

void HandlePNG(const char* input_png, const char* output_png) {
    int width, height, channels;
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load PNG: " << stbi_failure_reason() << "\n";
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
        std::cerr << "Failed to write PNG: " << stbi_failure_reason() << "\n";
        exit(1);
    }
}

void WriteColoredPPM(const char* filename, int width, int height, uint32_t color) {
    PPMImage img;
    img.width = width;
    img.height = height;
    img.max_val = 255;
    img.data = (unsigned char*)malloc(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        img.data[3 * i + 0] = (color >> (8 * 3)) & 0xFF;
        img.data[3 * i + 1] = (color >> (8 * 2)) & 0xFF;
        img.data[3 * i + 2] = (color >> (8 * 1)) & 0xFF;
    }

    if (!write_ppm(filename, &img)) {
        std::cerr << "Failed to write random PPM image.\n";
    }
    
    free(img.data);
}

void HandlePPM(const char* input_ppm, const char* output_ppm)
{
    PPMImage* img = read_ppm(input_ppm);
    if (!img) exit(1);
    PPMImage *out = (PPMImage*)malloc(sizeof(PPMImage));
    if (!out) exit(1);
    out->width = img->width;
    out->height = img->height;
    out->max_val = 255;
    out->data = (unsigned char*)malloc(img->width * img->height * 3);
    if (!out->data) exit(1);
    conv2d(img, out);
    write_ppm(output_ppm, out);
}

using namespace cv;

void conv2d(const cv::Mat& input, cv::Mat& output) {
    output = Mat::zeros(input.size(), input.type());

    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    Vec3b pixel = input.at<Vec3b>(y + ky - 1, x + kx - 1);
                    int k = k2d[ky][kx];
                    r_sum += pixel[2] * k;
                    g_sum += pixel[1] * k;
                    b_sum += pixel[0] * k;
                }
            }
            output.at<Vec3b>(y, x)[2] = std::clamp(std::abs(r_sum), 0, 255);
            output.at<Vec3b>(y, x)[1] = std::clamp(std::abs(g_sum), 0, 255);
            output.at<Vec3b>(y, x)[0] = std::clamp(std::abs(b_sum), 0, 255);
        }
    }
}

// Example 3D kernel (temporal depth = 3)
int KERNEL3D_EDGE_DETECTOR[3][3][3] = {
    {
        {-1, -2, -1},
        {-2, -4, -2},
        {-1, -2, -1}
    },
    {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    },
    {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    }
};

int (*k3d)[3][3] = KERNEL3D_EDGE_DETECTOR;

void HandleMP4_2D(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video.\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video.\n";
        exit(1);
    }

    cv::Mat frame, filtered;
    if (verbose) std::cout << "Processing video...\n";

    StartClock();
    if (verbose) std::cout << "clock started\n";
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        conv2d(frame, filtered);

        writer.write(filtered);
    }
    EndClock(true);

    cap.release();
    writer.release();
    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}

void HandleMP4_3D(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open input video\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";
    if (verbose) std::cout << "Loading frames...\n";
    std::vector<Mat> input_frames;
    for (int i = 0; i < total_frames; ++i) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame, COLOR_BGR2GRAY);          // Convert to grayscale
        Mat frame_f;
        frame.convertTo(frame_f, CV_32F);               // Convert to float
        input_frames.push_back(frame_f);
    }
    cap.release();

    int kD = 3, kH = 3, kW = 3;                         // Kernel size
    int padD = kD / 2, padH = kH / 2, padW = kW / 2;

    int D = static_cast<int>(input_frames.size());
    int H = input_frames[0].rows;
    int W = input_frames[0].cols;

    std::vector<Mat> output_frames(D);
    for (int t = 0; t < D; ++t)
        output_frames[t] = Mat::zeros(H, W, CV_32F);

    if (verbose) std::cout << "Applying 3D convolution...\n";
    // 3D convolution
    StartClock();
    if (verbose) std::cout << "clock started\n";
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int z = padD; z < D - padD; ++z) {
        for (int y = padH; y < H - padH; ++y) {
            for (int x = padW; x < W - padW; ++x) {
                float sum = 0.0f;
                for (int dz = 0; dz < kD; ++dz)
                    for (int dy = 0; dy < kH; ++dy)
                        for (int dx = 0; dx < kW; ++dx)
                        {
                            int tz = z + dz - padD;
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            sum += input_frames[tz].at<float>(ty, tx) * k3d[dz][dy][dx];
                        }
                output_frames[z].at<float>(y, x) = sum;
            }
        }
    }
    EndClock(true);

    if (verbose) std::cout << "Writing output video...\n";
    // Convert float frames to 8-bit for VideoWriter
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), false);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    for (auto &f : output_frames) {
        Mat f8u;
        f.convertTo(f8u, CV_8U, 1.0, 128); // optional: scale/shift if negative values exist
        writer.write(f8u);
    }

    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}

void HandleMP4_3D_RGB(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open input video\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";
    if (verbose) std::cout << "Loading frames...\n";
    // Load all frames (color)
    std::vector<Mat> input_frames;
    for (int i = 0; i < total_frames; ++i) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        Mat frame_f;
        frame.convertTo(frame_f, CV_8UC3);  // convert to float with 3 channels
        input_frames.push_back(frame_f);
    }
    cap.release();

    int kD = 3, kH = 3, kW = 3;                         // Kernel size
    int padD = kD / 2, padH = kH / 2, padW = kW / 2;

    int D = static_cast<int>(input_frames.size());
    int H = input_frames[0].rows;
    int W = input_frames[0].cols;
    
    // Initialize output frames
    std::vector<Mat> output_frames(D);
    for (int t = 0; t < D; ++t)
        output_frames[t] = Mat::zeros(H, W, CV_8UC3);

    if (verbose) std::cout << "Applying 3D convolution on RGB frames...\n";
    // 3D convolution
    StartClock();
    if (verbose) std::cout << "clock started\n";
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int z = padD; z < D - padD; ++z) {
        for (int y = padH; y < H - padH; ++y) {
            for (int x = padW; x < W - padW; ++x) {
                Vec3f sum(0, 0, 0);
                for (int dz = 0; dz < kD; ++dz)
                    for (int dy = 0; dy < kH; ++dy)
                        for (int dx = 0; dx < kW; ++dx)
                        {
                            int tz = z + dz - padD;
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            Vec3f pixel = input_frames[tz].at<Vec3f>(ty, tx);
                            float k = k3d[dz][dy][dx];
                            sum[0] += pixel[0] * k; // Blue
                            sum[1] += pixel[1] * k; // Green
                            sum[2] += pixel[2] * k; // Red
                        }
                output_frames[z].at<Vec3f>(y, x) = sum;
            }
        }
    }
    EndClock(true);

    if (verbose) std::cout << "Writing output video...\n";
    // Write output video
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    for (auto &f : output_frames) {
        Mat f8u;
        f.convertTo(f8u, CV_8UC3, 1.0, 128);  // convert float to 8-bit
        writer.write(f8u);
    }

    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}

void HandleMP4_3D_RGB_Sliding(const char* input_path, const char* output_path)
{
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open input video\n";
        exit(1);
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    if (verbose) std::cout << "Video properties: " << width << "x" << height << " at " << fps << " FPS, total frames: " << total_frames << "\n";

    int kD = 3, kH = 3, kW = 3;
    int padH = kH / 2, padW = kW / 2;
    int padD = kD / 2;

    // Prepare VideoWriter
    VideoWriter writer(output_path, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::cout << "Failed to open output video\n";
        exit(1);
    }

    std::deque<Mat> buffer;  // sliding window of input frames

    if (verbose) std::cout << "Processing video with sliding 3D convolution...\n";

    int frame_idx = 0;
    Mat frame;
    StartClock();
    if (verbose) std::cout << "clock started\n";
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        Mat frame_f;
        frame.convertTo(frame_f, CV_32FC3);
        buffer.push_back(frame_f);

        // Wait until we have enough frames for the convolution
        if (buffer.size() < kD) continue;

        // Convolve the middle frame
        Mat out = Mat::zeros(height, width, CV_32FC3);
        int mid = kD / 2;

        
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int y = padH; y < height - padH; ++y) {
            for (int x = padW; x < width - padW; ++x) {
                Vec3f sum(0,0,0);
                for (int dz = 0; dz < kD; ++dz)
                    for (int dy = 0; dy < kH; ++dy)
                        for (int dx = 0; dx < kW; ++dx)
                        {
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            Vec3f pixel = buffer[dz].at<Vec3f>(ty, tx);
                            float k = k3d[dz][dy][dx];
                            sum[0] += pixel[0] * k;
                            sum[1] += pixel[1] * k;
                            sum[2] += pixel[2] * k;
                        }
                out.at<Vec3f>(y, x) = sum;
            }
        }
        // Convert to 8-bit and write
        Mat out8u;
        out.convertTo(out8u, CV_8UC3, 1.0, 128); // shift if needed
        writer.write(out8u);
        
        // Remove the oldest frame
        buffer.pop_front();
        frame_idx++;
    }
    EndClock(true);
    
    cap.release();
    writer.release();
    if (verbose) std::cout << "Output saved to " << output_path << "\n";
}

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    std::cout << "OpenMP is enabled.\n";
#endif

    if (argc != 3) 
    {
        std::cout << "Usage: " << argv[0] << " <input image> <output image>\n";
        return 1;
    }

    const char* extension = strrchr(argv[1], '.');
    if (strcmp(extension, ".ppm") == 0)
    {
        HandlePPM(argv[1], argv[2]);
    }
    else if (strcmp(extension, ".png") == 0) 
    {
        HandlePNG(argv[1], argv[2]);
    }
    else if (strcmp(extension, ".mp4") == 0)
    {
        std::string out_arg = argv[2];
        size_t dot = out_arg.rfind('.');
        std::string base = (dot == std::string::npos) ? out_arg : out_arg.substr(0, dot);
        std::string ext = (dot == std::string::npos) ? std::string() : out_arg.substr(dot);

        std::string out_2d = base + "_2d" + ext;
        std::string out_3d_gray = base + "_3d_gray" + ext;
        std::string out_3d_rgb = base + "_3d_rgb" + ext;

        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 2D per-frame convolution -> " << out_2d << "\n";
        HandleMP4_2D(argv[1], out_2d.c_str());
        std::cout << "---------------------------------------------------------------\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 3D temporal convolution (grayscale) -> " << out_3d_gray << "\n";
        HandleMP4_3D(argv[1], out_3d_gray.c_str());
        std::cout << "---------------------------------------------------------------\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 3D temporal convolution (RGB) -> " << out_3d_rgb << "\n";
        // HandleMP4_3D_RGB(argv[1], out_3d_rgb.c_str());
        HandleMP4_3D_RGB_Sliding(argv[1], out_3d_rgb.c_str());
        std::cout << "---------------------------------------------------------------\n";
    }
    else 
    {
        std::cerr << "Unsupported file format: " << extension << "\n";
        return 1;
    }
    return 0;
}