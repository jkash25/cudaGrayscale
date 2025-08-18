
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void rgb2gray(const unsigned char* rgbImage, unsigned char* grayImage, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = rgbImage[idx];
        unsigned char g = rgbImage[idx + 1];
        unsigned char b = rgbImage[idx + 2];

        grayImage[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char** argv) {
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }

    cv::Mat gray;

    // Start timer
    int64 startTime = cv::getTickCount();

    // CPU grayscale conversion
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    // Stop timer
    int64 end = cv::getTickCount();
    double elapsed_ms = (end - startTime) * 1000.0 / cv::getTickFrequency();

    std::cout << "CPU grayscale conversion took " << elapsed_ms << " ms" << std::endl;

    //cv::imwrite("cpu_output.jpg", gray);
    if (argc != 3) {
        std::cerr << "Usage: ./grayscale <input_image> <output_image>" << std::endl;
        return -1;
    }

    if (input.empty()) {
        std::cerr << "Error: could not load image " << argv[1] << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    cv::Mat output(height, width, CV_8UC1);

    //allocate gpu mem
    unsigned char *d_rgb, *d_gray;
    size_t rgbSize = width * height * channels * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_rgb, rgbSize);
    cudaMalloc(&d_gray, graySize);

    //copy to gpu
    cudaMemcpy(d_rgb, input.data, rgbSize, cudaMemcpyHostToDevice);

    dim3 block (16, 16);
    dim3 grid ((width + block.x -1 ) / block.x, (height * block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rgb2gray<<<grid, block>>>(d_rgb, d_gray, width, height, channels);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU kernel time: " << ms << " ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_gray, graySize, cudaMemcpyDeviceToHost);
    cv::imwrite(argv[2], output);
    cudaFree(d_rgb);
    cudaFree(d_gray);

    std::cout << "Saved grayscale image to: " << argv[2] << std::endl;
    std::cout << "GPU kernel improved performance by " << elapsed_ms/ms << "x" << std::endl;
    return 0;
}