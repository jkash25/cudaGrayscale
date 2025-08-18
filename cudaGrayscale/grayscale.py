
import cv2
import numpy as np
from numba import cuda
import math
import time

@cuda.jit
def rgb2gray(rgb_image, gray_image):
    x, y = cuda.grid(2)
    height, width, channels = rgb_image.shape
    if x < width and y < height:
        r = rgb_image[y, x, 0]
        g = rgb_image[y, x, 1]
        b = rgb_image[y, x, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray_image[y, x] = gray

input_img = cv2.imread("C:\\Users\\Jai K\\CS Stuff\\learncuda\\cudaGrayscale\\input.jpg")
height, width, channels = input_img.shape
gray_cpu = np.zeros((height, width), dtype=np.uint8)

# ---- CPU baseline ----
start = time.time()
gray_cpu = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
cpu_time = (time.time() - start) * 1000
print(f"CPU cvtColor time: {cpu_time:.2f} ms")

# ---- GPU with Numba ----
# Allocate GPU memory
d_input = cuda.to_device(input_img)
d_output = cuda.device_array((height, width), dtype=np.uint8)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(width / threadsperblock[0])
blockspergrid_y = math.ceil(height / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

rgb2gray[blockspergrid, threadsperblock](d_input, d_output)
cuda.synchronize()

start = time.time()
rgb2gray[blockspergrid, threadsperblock](d_input, d_output)
cuda.synchronize()
gpu_time = (time.time() - start) * 1000
print(f"GPU kernel time (Numba): {gpu_time: .2f} ms")

gray_gpu = d_output.copy_to_host()

cv2.imwrite("gray_cpu_numba.jpg", gray_cpu)
cv2.imwrite("gray_gpu_numba.jpg", gray_gpu)