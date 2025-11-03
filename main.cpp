
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
#include "fftw-3.3.10/fftw-3.3.10/api/fftw3.h"

bool pseudo = false;
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

float clampf(float val) {
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
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

void conv2d(PPMImage& input, PPMImage& output)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int px = x + kx - 1; 
                    int py = y + ky - 1;
                    if (!(px < 0 || px >= input.width || py < 0 || py >= input.height))
                    {
                        int k = k2d[ky][kx];
                        // int k = KERNEL_IDENTITY[ky][kx];
                        r_sum += input.data[(3 * (py * input.width + px)) + 0] * k;
                        g_sum += input.data[(3 * (py * input.width + px)) + 1] * k;
                        b_sum += input.data[(3 * (py * input.width + px)) + 2] * k;
                    }
                }
            }
            output.data[(3 * (y * input.width + x)) + 0] = clamp(abs(r_sum));
            output.data[(3 * (y * input.width + x)) + 1] = clamp(abs(g_sum));
            output.data[(3 * (y * input.width + x)) + 2] = clamp(abs(b_sum));
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

void fftw_conv2d(const PPMImage* input, PPMImage* output) {
    int H = input->height;
    int W = input->width;
    int KH = KERNEL_HEIGHT;
    int KW = KERNEL_WIDTH;

    int outH = H;
    int outW = W;

    int fftH = H + KH - 1;
    int fftW = W + KW - 1;

    output->width = outW;
    output->height = outH;
    output->max_val = 255;
    output->data = (unsigned char*)malloc(outW * outH * 3);

    // Process 3 channels independently
    for (int c = 0; c < 3; ++c) {
        // Allocate FFTW arrays
        fftw_complex *A = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));
        fftw_complex *B = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));
        fftw_complex *C = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftH * (fftW / 2 + 1));

        double *imgPadded = (double*)fftw_malloc(sizeof(double) * fftH * fftW);
        double *kerPadded = (double*)fftw_malloc(sizeof(double) * fftH * fftW);
        double *result = (double*)fftw_malloc(sizeof(double) * fftH * fftW);

        // Zero-pad both arrays
        memset(imgPadded, 0, sizeof(double) * fftH * fftW);
        memset(kerPadded, 0, sizeof(double) * fftH * fftW);

        // Copy image channel to padded input
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                imgPadded[y * fftW + x] = (double)input->data[3 * (y * W + x) + c];

        // Copy kernel (flipped for true convolution)
        for (int y = 0; y < KH; ++y)
            for (int x = 0; x < KW; ++x)
                kerPadded[y * fftW + x] = k2d[KH - 1 - y][KW - 1 - x];

        // Plan FFTs
        fftw_plan planA = fftw_plan_dft_r2c_2d(fftH, fftW, imgPadded, A, FFTW_ESTIMATE);
        fftw_plan planB = fftw_plan_dft_r2c_2d(fftH, fftW, kerPadded, B, FFTW_ESTIMATE);
        fftw_plan planC = fftw_plan_dft_c2r_2d(fftH, fftW, C, result, FFTW_ESTIMATE);

        // Execute FFTs
        fftw_execute(planA);
        fftw_execute(planB);

        // Multiply frequency-domain terms (complex multiply)
        int nfreq = fftH * (fftW / 2 + 1);
        for (int i = 0; i < nfreq; ++i) {
            float a_re = A[i][0], a_im = A[i][1];
            float b_re = B[i][0], b_im = B[i][1];
            C[i][0] = a_re * b_re - a_im * b_im;
            C[i][1] = a_re * b_im + a_im * b_re;
        }

        // Inverse FFT
        fftw_execute(planC);

        // Normalize result (FFTW doesn't)
        for (int i = 0; i < fftH * fftW; ++i)
            result[i] /= (fftH * fftW);

        // Crop back to original size
        int yOffset = (KH - 1) / 2;
        int xOffset = (KW - 1) / 2;
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float val = result[(y + yOffset) * fftW + (x + xOffset)];
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output->data[3 * (y * outW + x) + c] = (unsigned char)(val);
            }
        }

        // Cleanup
        fftw_destroy_plan(planA);
        fftw_destroy_plan(planB);
        fftw_destroy_plan(planC);
        fftw_free(A); fftw_free(B); fftw_free(C);
        fftw_free(imgPadded); fftw_free(kerPadded); fftw_free(result);
    }
}

void HandlePNG(const char* input_png, const char* output_png) {
    int width, height, channels;
    unsigned char* img = stbi_load(input_png, &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load PNG: " << stbi_failure_reason() << "\n";
        exit(1);
    }

    PPMImage ppm_img;
    ppm_img.width = width;
    ppm_img.height = height;
    ppm_img.max_val = 255;
    ppm_img.data = img;

    PPMImage out;
    if (pseudo)
    {
        std::cout << "running pseudo conv2d\n";
        out.width = width;
        out.height = height;
        out.max_val = 255;
        out.data = (unsigned char*)malloc(width * height * 3);
        conv2d(ppm_img, out);
    }
    else
    {
        std::cout << "running fftw conv2d\n";
        fftw_conv2d(&ppm_img, &out);
    }
    if (!stbi_write_png(output_png, out.width, out.height, 3, out.data, out.width * 3)) {
        std::cerr << "Failed to write PNG: " << stbi_failure_reason() << "\n";
        exit(1);
    }
    free(out.data);
    stbi_image_free(img);
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
    PPMImage out;
    out.width = img->width;
    out.height = img->height;
    out.max_val = 255;
    out.data = (unsigned char*)malloc(img->width * img->height * 3);
    if (!out.data) exit(1);
    conv2d(*img, out);
    write_ppm(output_ppm, &out);
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
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        Mat frame_f;
        frame.convertTo(frame_f, CV_32F);
        input_frames.push_back(frame_f);
    }
    cap.release();

    int padD = KERNEL_DEPTH / 2, padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;

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
                for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
                    for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                        for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                        {
                            int tz = z + dz - padD;
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            sum += input_frames[tz].at<float>(ty, tx) * k3d[dz][dy][dx];
                        }
                output_frames[z].at<float>(y, x) = clampf((sum));
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
        f.convertTo(f8u, CV_8U, 1.0, 0);
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

    int padD = KERNEL_DEPTH / 2, padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;

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
                for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
                    for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                        for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
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
        f.convertTo(f8u, CV_8UC3, 1.0, 0);
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

    int padH = KERNEL_HEIGHT / 2, padW = KERNEL_WIDTH / 2;
    int padD = KERNEL_DEPTH / 2;

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
        if (buffer.size() < KERNEL_DEPTH) continue;

        // Convolve the middle frame
        Mat out = Mat::zeros(height, width, CV_32FC3);
        int mid = KERNEL_DEPTH / 2;

        
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int y = padH; y < height - padH; ++y) {
            for (int x = padW; x < width - padW; ++x) {
                Vec3f sum(0,0,0);
                for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
                    for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                        for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                        {
                            int ty = y + dy - padH;
                            int tx = x + dx - padW;
                            Vec3f pixel = buffer[dz].at<Vec3f>(ty, tx);
                            float k = k3d[dz][dy][dx];
                            sum[0] += pixel[0] * k;
                            sum[1] += pixel[1] * k;
                            sum[2] += pixel[2] * k;
                        }
                sum[0] = clampf(sum[0]);
                sum[1] = clampf(sum[1]);
                sum[2] = clampf(sum[2]);
                out.at<Vec3f>(y, x) = sum;
            }
        }
        // Convert to 8-bit and write
        Mat out8u;
        out.convertTo(out8u, CV_8UC3, 1.0, 0);
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

void fftw()
{
    double *in;
    fftw_complex *out;
    int N = 16;
    in = (double*) fftw_malloc(sizeof(double) * N);
    in[0] = 1.0;
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    fftw_destroy_plan(p);

    for (int i = 0; i < N; i++) {
        std::cout << "out[" << i << "] = " << out[i][0] << " + " << out[i][1] << "i\n";
    }

    fftw_free(in);
    fftw_free(out);
}

void usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " <input image> <output image> [options]\n";
    std::cout << "Supported input/output formats: .ppm, .png, .mp4\n";
    std::cout << "Options:\n";
    std::cout << "  -pseudo       Use pseudo convolution (spatial domain) instead of FFTW\n";
    std::cout << "  -verbose      Enable verbose output\n";
}

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    std::cout << "OpenMP is enabled.\n";
#endif

    const char* input_path = NULL;
    const char* output_path = NULL;
    int i = 1;
    while (i < argc)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            i++;
            if (i < argc)
            {
                input_path = argv[i];
            }
            else
            {
                std::cerr << "Error: Missing input file after -i\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            i++;
            if (i < argc)
            {
                output_path = argv[i];
            }
            else
            {
                std::cerr << "Error: Missing output file after -o\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "-pseudo") == 0)
        {
            pseudo = true;
        }
        else if (strcmp(argv[i], "-verbose") == 0)
        {
            verbose = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else 
        {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
        i++;
    }
    if (input_path == NULL || output_path == NULL)
    {
        std::cerr << "Error: Input and output file paths are required.\n";
        usage(argv[0]);
        return 1;
    }

    const char* extension = strrchr(input_path, '.');
    if (strcmp(extension, ".ppm") == 0)
    {
        HandlePPM(input_path, output_path);
    }
    else if (strcmp(extension, ".png") == 0) 
    {
        HandlePNG(input_path, output_path);
    }
    else if (strcmp(extension, ".mp4") == 0)
    {
        std::string out_arg = output_path;
        size_t dot = out_arg.rfind('.');
        std::string base = (dot == std::string::npos) ? out_arg : out_arg.substr(0, dot);
        std::string ext = (dot == std::string::npos) ? std::string() : out_arg.substr(dot);

        std::string out_2d = base + "_2d" + ext;
        std::string out_3d_gray = base + "_3d_gray" + ext;
        std::string out_3d_rgb = base + "_3d_rgb" + ext;

        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 2D per-frame convolution -> " << out_2d << "\n";
        HandleMP4_2D(input_path, out_2d.c_str());
        std::cout << "---------------------------------------------------------------\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 3D temporal convolution (grayscale) -> " << out_3d_gray << "\n";
        HandleMP4_3D(input_path, out_3d_gray.c_str());
        std::cout << "---------------------------------------------------------------\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Running 3D temporal convolution (RGB) -> " << out_3d_rgb << "\n";
        // HandleMP4_3D_RGB(input_path, out_3d_rgb.c_str()); // consumes too much memory
        HandleMP4_3D_RGB_Sliding(input_path, out_3d_rgb.c_str());
        std::cout << "---------------------------------------------------------------\n";
    }
    else 
    {
        std::cerr << "Unsupported file format: " << extension << "\n";
        return 1;
    }
    return 0;
}