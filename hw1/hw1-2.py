import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

dir_name = "./"  # 設定目錄名稱
img_name = "hw1-2.jpg"  # 設定圖片檔案名稱
# 使用 os.path.join 組合路徑，這樣可以確保在不同作業系統上都能正確組合
img_path = os.path.join(dir_name, img_name)

# 使用 cv2 的 imread 函數讀取圖片
# cv2.IMREAD_COLOR 代表以彩色模式讀取圖片
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 將彩色圖片轉換為灰階圖片
# 使用標準的公式進行轉換：Grayscale = 0.299 * R + 0.587 * G + 0.114 * B
# 在 OpenCV 中，圖片的順序是 BGR
img_to_gray = (
    0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
).astype(
    np.uint8
)  # 將計算結果轉換為 np.uint8 資料型態，這是因為圖片的像素值通常是 0-255 的整數

# 建立 Gaussian kernel (高斯核)
size = 3  # 定義高斯核的大小，這裡是 3x3
sigma = 3  # 定義高斯核的標準差
offset = size // 2  # 根據核的大小計算偏移量，用於確定核的中心位置
# 初始化高斯核，所有值都是 0，資料型態為浮點數
gaussian_kernel = np.zeros((size, size), dtype=np.float32)
# 遍歷高斯核的每個元素
for x in range(size):
    for y in range(size):
        # 根據高斯公式計算核中每個位置的值
        gaussian_kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(
            -(((x - offset) ** 2 + (y - offset) ** 2) / (2 * sigma**2))
        )
# 正規化高斯核，使其所有元素之和為 1
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

# Gaussian blur (高斯模糊)
pad_size = size // 2  # 計算需要填充的大小，為高斯核大小的一半
# 使用 np.pad 對圖片進行填充，為了確保在濾波時不超出邊界
pad_image = np.pad(
    img_to_gray, ((pad_size, pad_size), (pad_size, pad_size)), mode="constant"
)  # 使用 constant 模式，外部邊界會被填充為 0
h, w = img_to_gray.shape  # 獲取原始圖片的尺寸
blur_image = np.zeros((h, w), dtype=np.int32)
# 遍歷圖片的每個像素
for i in range(h):
    for j in range(w):
        # 對每個像素位置，使用高斯核進行卷積操作
        # 即：對每個高斯核的元素與圖片中對應元素相乘，然後相加
        # 使用 round() 函數確保結果為最接近的整數
        blur_image[i, j] = round(
            np.sum(pad_image[i : i + size, j : j + size] * gaussian_kernel)
        )
blur_image = blur_image.astype(np.uint8)  # 將模糊後的圖片資料型態轉為 uint8

#  顯示 blur_image
plt.figure(figsize=(12, 8))
plt.imshow(blur_image, cmap="gray")
plt.title("Gaussian Blurred Image")
plt.axis("off")
plt.show()


# Sobel 算子
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
size = 3  # Sobel 算子的大小
pad_size = size // 2  # 計算需要的填充量
# 對模糊後的圖片進行邊緣填充，確保使用 Sobel 算子時不超出邊界
pad_image = np.pad(
    blur_image, ((pad_size, pad_size), (pad_size, pad_size)), mode="constant"
)

# 計算每個像素的 X 方向梯度
gradient_ori_img_x = np.zeros((h, w), dtype=np.float32)
for i in range(h):
    for j in range(w):
        # 對每個像素使用 sobel_x 進行卷積操作
        # 即：對每個 sobel_x 的元素與圖片中對應元素相乘，然後相加
        gradient_ori_img_x[i, j] = np.sum(
            pad_image[i : i + size, j : j + size] * sobel_x
        )
# 將計算得到的梯度值轉換為正確的像素值範圍（0-255）
gradient_img_x = cv2.convertScaleAbs(gradient_ori_img_x)

# 計算每個像素的 Y 方向梯度
gradient_ori_img_y = np.zeros((h, w), dtype=np.float32)
for i in range(h):
    for j in range(w):
        # 對每個像素使用 sobel_y 進行卷積操作
        # 即：對每個 sobel_y 的元素與圖片中對應元素相乘，然後相加
        gradient_ori_img_y[i, j] = np.sum(
            pad_image[i : i + size, j : j + size] * sobel_y
        )
# 將計算得到的梯度值轉換為正確的像素值範圍（0-255）
gradient_img_y = cv2.convertScaleAbs(gradient_ori_img_y)

# 計算梯度幅度
# 表示圖片在每個像素位置的邊緣強度
# 通過將 X 方向和 Y 方向的梯度值進行平方，加在一起，然後取平方根得到
gradient_magnitude = np.sqrt(gradient_ori_img_x**2 + gradient_ori_img_y**2)
print(gradient_magnitude)  # 印出計算得到的梯度幅度

# 顯示 gradient_img_x
plt.figure(figsize=(12, 8))
plt.imshow(gradient_img_x, cmap="gray")
plt.title("Gradient Image (X-direction)")
plt.axis("off")
plt.show()

# 顯示 gradient_img_y
plt.figure(figsize=(12, 8))
plt.imshow(gradient_img_y, cmap="gray")
plt.title("Gradient Image (Y-direction)")
plt.axis("off")
plt.show()


# 計算 Ix^2、Iy^2 和 IxIy 用於 Structure Tensor H
Ix2 = gradient_ori_img_x**2
Iy2 = gradient_ori_img_y**2
Ixy = gradient_ori_img_x * gradient_ori_img_y
k = 0.04  # k 是 Harris 角點檢測中的常數，通常介於 0.04 到 0.06 之間
window_size = 3  # 定義滑動窗口的大小，這裡是 3x3
offset = window_size // 2  # 計算偏移量
R = np.zeros((h, w), dtype=np.float32)  # 初始化 R 矩陣以存儲每個像素的 R 值
# 遍歷每個像素，計算它的 R 值
for i in range(offset, h - offset):
    for j in range(offset, w - offset):
        # 對滑動窗口內的 Ix^2、Iy^2 和 IxIy 求和，並與高斯核進行權重平均
        Sx2 = np.sum(
            Ix2[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )
        Sy2 = np.sum(
            Iy2[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )
        Sxy = np.sum(
            Ixy[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )

        #  det(M) = ad - bc
        det = (Sx2 * Sy2) - (Sxy * Sxy)
        # trace(M) = a + b
        trace = Sx2 + Sy2
        # 使用 Harris 角點響應函數計算 R 值
        R[i, j] = det - k * (trace**2)

# 選取 R 值最大值的 0.5% 作為門檻值
threshold = 0.005 * R.max()
# 創建一個新的圖片，與 R 矩陣大小相同，所有像素值初始化為 0
corner_img = np.zeros_like(R, dtype=np.uint8)
# 在新圖片中，將 R 值大於門檻值的像素設為 255（白色）
corner_img[R > threshold] = 255

# 顯示 corner_img(window_size = 3x3)
plt.figure(figsize=(12, 8))
plt.imshow(corner_img, cmap="gray")
plt.title("Harris Response(window_size = 3x3)")
plt.axis("off")
plt.show()

# 顯示 final output
plt.figure(figsize=(12, 8))
# 根據門檻值找到所有大於門檻值的像素座標
corner_img_for_overlaid = np.where(R > threshold)
# 顯示原始灰階圖片
plt.imshow(img_to_gray, cmap="gray")
# 在圖片上標記角點：將檢測到的角點位置以紅色點標記在原始的灰階圖片上
plt.scatter(
    corner_img_for_overlaid[1], corner_img_for_overlaid[0], s=1, c="red", marker="o"
)
plt.title("Original Image with Corner Points Overlaid(window_size = 3x3)")
plt.axis("off")
plt.show()

# 選取 R 值最大值的 0.1% 作為門檻值
threshold = 0.001 * R.max()
# 創建一個新的圖片，與 R 矩陣大小相同，所有像素值初始化為 0
corner_img = np.zeros_like(R, dtype=np.uint8)
# 在新圖片中，將 R 值大於門檻值的像素設為 255（白色）
corner_img[R > threshold] = 255

# 顯示 corner_img(threshold = 0.001 * R.max())
plt.figure(figsize=(12, 8))
plt.imshow(corner_img, cmap="gray")
plt.title("Harris Response(threshold = 0.001 * R.max())")
plt.axis("off")
plt.show()

# 顯示 final output
plt.figure(figsize=(12, 8))
# 根據門檻值找到所有大於門檻值的像素座標
corner_img_for_overlaid = np.where(R > threshold)
# 顯示原始灰階圖片
plt.imshow(img_to_gray, cmap="gray")
# 在圖片上標記角點：將檢測到的角點位置以紅色點標記在原始的灰階圖片上
plt.scatter(
    corner_img_for_overlaid[1], corner_img_for_overlaid[0], s=1, c="red", marker="o"
)
plt.title("Original Image with Corner Points Overlaid(threshold = 0.001 * R.max())")
plt.axis("off")
plt.show()

# 計算 R 值落點分佈
bins_ranges = [-1e11, -1e10, -1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
hist, bin_edges = np.histogram(R, bins=bins_ranges)
for i in range(len(bin_edges) - 1):
    print(f"Range: ({bin_edges[i]:.0e}, {bin_edges[i + 1]:.0e}] -> Count: {hist[i]}")


window_size = 5  # 設置非極大值抑制的窗口大小，這裡是 5x5
offset = window_size // 2  # 計算偏移量
nms_img = np.zeros_like(R, dtype=np.float32)
for i in range(offset, h - offset):
    for j in range(offset, w - offset):
        # 如果當前像素的 R 值是其鄰域內 5x5 的最大值，則保留它
        if R[i, j] == np.max(
            R[i - offset : i + offset + 1, j - offset : j + offset + 1]
        ):
            nms_img[i, j] = R[i, j]

# 設定門檻值為 nms_img 中最大值的 0.5%
threshold = 0.005 * nms_img.max()
# 創建一個新的圖片，與 nms_img 大小相同，所有像素值初始化為 0
nms_after_threshold = np.zeros_like(nms_img, dtype=np.uint8)
# 在新圖片中，將 nms_img 值大於門檻值的像素設為255（白色）
nms_after_threshold[nms_img > threshold] = 255

# 顯示 nms_after_threshold
plt.figure(figsize=(12, 8))
plt.imshow(nms_after_threshold, cmap="gray")
plt.title("NMS Image")
plt.axis("off")
plt.show()


# 建立 Gaussian kernel (高斯核)
size = 7  # 定義高斯核的大小，這裡是 7x7
sigma = 3  # 定義高斯核的標準差
offset = size // 2  # 根據核的大小計算偏移量，用於確定核的中心位置
# 初始化高斯核，所有值都是 0，資料型態為浮點數
gaussian_kernel = np.zeros((size, size), dtype=np.float32)
# 遍歷高斯核的每個元素
for x in range(size):
    for y in range(size):
        # 根據高斯公式計算核中每個位置的值
        gaussian_kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(
            -(((x - offset) ** 2 + (y - offset) ** 2) / (2 * sigma**2))
        )
# 正規化高斯核，使其所有元素之和為 1
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

# 計算 Ix^2、Iy^2 和 IxIy 用於 Structure Tensor H
Ix2 = gradient_ori_img_x**2
Iy2 = gradient_ori_img_y**2
Ixy = gradient_ori_img_x * gradient_ori_img_y
k = 0.04  # k 是 Harris 角點檢測中的常數，通常介於 0.04 到 0.06 之間
window_size = 7  # 定義滑動窗口的大小，這裡是 7x7
offset = window_size // 2  # 計算偏移量
R = np.zeros((h, w), dtype=np.float32)  # 初始化 R 矩陣以存儲每個像素的 R 值
# 遍歷每個像素，計算它的 R 值
for i in range(offset, h - offset):
    for j in range(offset, w - offset):
        # 對滑動窗口內的 Ix^2、Iy^2 和 IxIy 求和，並與高斯核進行權重平均
        Sx2 = np.sum(
            Ix2[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )
        Sy2 = np.sum(
            Iy2[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )
        Sxy = np.sum(
            Ixy[i - offset : i + offset + 1, j - offset : j + offset + 1]
            * gaussian_kernel
        )

        #  det(M) = ad - bc
        det = (Sx2 * Sy2) - (Sxy * Sxy)
        # trace(M) = a + b
        trace = Sx2 + Sy2
        # 使用 Harris 角點響應函數計算 R 值
        R[i, j] = det - k * (trace**2)

# 選取 R 值最大值的 0.5% 作為門檻值
threshold = 0.005 * R.max()
# 創建一個新的圖片，與 R 矩陣大小相同，所有像素值初始化為 0
corner_img = np.zeros_like(R, dtype=np.uint8)
# 在新圖片中，將 R 值大於門檻值的像素設為 255（白色）
corner_img[R > threshold] = 255

# 顯示 corner_img(window_size = 7x7)
plt.figure(figsize=(12, 8))
plt.imshow(corner_img, cmap="gray")
plt.title("Harris Response(window_size = 7x7)")
plt.axis("off")
plt.show()

# 顯示 final output
plt.figure(figsize=(12, 8))
# 根據門檻值找到所有大於門檻值的像素座標
corner_img_for_overlaid = np.where(R > threshold)
# 顯示原始灰階圖片
plt.imshow(img_to_gray, cmap="gray")
# 在圖片上標記角點：將檢測到的角點位置以紅色點標記在原始的灰階圖片上
plt.scatter(
    corner_img_for_overlaid[1], corner_img_for_overlaid[0], s=1, c="red", marker="o"
)
plt.title("Original Image with Corner Points Overlaid(window_size = 7x7)")
plt.axis("off")
plt.show()
