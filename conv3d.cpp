
float clampf(float val) {
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
}

extern "C" void conv3d(void* input_ptr, float* kernel, void* output_ptr, int width, int height)
{
    float* input = static_cast<float*>(input_ptr);
    float* output = static_cast<float*>(output_ptr);

    const int KERNEL_WIDTH = 3;
    const int KERNEL_HEIGHT = 3;
    const int KERNEL_DEPTH = 3;
    const int CHANNELS = 3;

    int padH = KERNEL_HEIGHT / 2;
    int padW = KERNEL_WIDTH / 2;

    const int input_frame_stride = height * width * CHANNELS; 
    const int input_row_stride = width * CHANNELS;            

    for (int y = padH; y < height - padH; ++y)
    {
        for (int x = padW; x < width - padW; ++x)
        {
            int out_idx_start = y * input_row_stride + x * CHANNELS;
            float sum[3] = {0.0f, 0.0f, 0.0f};
            for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
            {
                for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                {
                    for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                    {
                        int y_in = y + dy - padH;
                        int x_in = x + dx - padW;
                        
                        int idx_in_start = dz * input_frame_stride + y_in * input_row_stride + x_in * CHANNELS;
                        int kidx = dz * (KERNEL_WIDTH * KERNEL_HEIGHT) + dy * KERNEL_WIDTH + dx;
                        
                        float k = kernel[kidx];
                        for (int c = 0; c < CHANNELS; ++c)
                        {
                            sum[c] += input[idx_in_start + c] * k;
                        }
                    }
                }
            }
            
            for (int c = 0; c < CHANNELS; ++c)
            {
                output[out_idx_start + c] = clampf(sum[c]);
            }
        }
    }
}

extern "C" void conv3d_3p(
    void* prev_ptr,   // Pointer to Frame T-1
    void* curr_ptr,   // Pointer to Frame T
    void* next_ptr,   // Pointer to Frame T+1
    float* kernel, 
    void* output_ptr, 
    int width, 
    int height)
{
    // 1. Store the 3 pointers in an array for easy access via 'dz'
    float* inputs[3];
    inputs[0] = static_cast<float*>(prev_ptr);
    inputs[1] = static_cast<float*>(curr_ptr);
    inputs[2] = static_cast<float*>(next_ptr);
    
    float* output = static_cast<float*>(output_ptr);

    const int KERNEL_WIDTH = 3;
    const int KERNEL_HEIGHT = 3;
    const int KERNEL_DEPTH = 3;
    const int CHANNELS = 3;

    int padH = KERNEL_HEIGHT / 2;
    int padW = KERNEL_WIDTH / 2;

    // Stride is just for ONE frame now
    const int input_row_stride = width * CHANNELS;            

    for (int y = padH; y < height - padH; ++y)
    {
        for (int x = padW; x < width - padW; ++x)
        {
            int out_idx_start = y * input_row_stride + x * CHANNELS;
            float sum[3] = {0.0f, 0.0f, 0.0f};
            
            for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
            {
                // Select the correct frame based on Z depth
                float* current_input_frame = inputs[dz];

                for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                {
                    for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                    {
                        int y_in = y + dy - padH;
                        int x_in = x + dx - padW;
                        
                        // Index is relative to the SPECIFIC frame now
                        int idx_pixel = y_in * input_row_stride + x_in * CHANNELS;
                        
                        int kidx = dz * (KERNEL_WIDTH * KERNEL_HEIGHT) + dy * KERNEL_WIDTH + dx;
                        float k = kernel[kidx];

                        for (int c = 0; c < CHANNELS; ++c)
                        {
                            sum[c] += current_input_frame[idx_pixel + c] * k;
                        }
                    }
                }
            }
            
            for (int c = 0; c < CHANNELS; ++c)
            {
                output[out_idx_start + c] = clampf(sum[c]);
            }
        }
    }
}