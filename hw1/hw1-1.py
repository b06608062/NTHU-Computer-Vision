import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

dir_name = "./"  # 設定目錄名稱
img_name = "hw1-1.jpg"  # 設定圖片檔案名稱
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

# 獲取灰階圖片的形狀，以確定其高度和寬度
img_shape = img_to_gray.shape
N = img_shape[0]  # 圖片的高度
M = img_shape[1]  # 圖片的寬度
# 計算灰階直方圖
histogram = np.zeros(256, dtype=np.int32)
# 遍歷每一個像素
for i in range(N):
    for j in range(M):
        # 對於圖片中的每個像素值，增加其在直方圖中的計數
        histogram[img_to_gray[i, j]] += 1

# 計算累積分布函數（CDF）
cdf = np.zeros(256, dtype=np.int32)
cdf[0] = histogram[0]  # 設定 CDF 的初始值為直方圖的第一個值
# 從第二個值開始，每個值都是當前的直方圖值加上前一個 CDF 的值
for i in range(1, 256):
    cdf[i] = histogram[i] + cdf[i - 1]

# 計算直方圖均衡化的映射函數
equalized_mapping_function = np.zeros(256, dtype=np.int32)
total_pixel = N * M  # 計算總像素數，也就是圖片的高度乘以寬度
for i in range(256):
    # 為每個灰階值計算新的像素值：
    # 1. 使用 CDF 計算像素值 i 的累積率，即 cdf[i] / total_pixel
    # 2. 將累積概率乘以 255 (因為像素值範圍是 0-255)
    # 3. 使用 round() 函數確保得到的值是最接近的整數
    equalized_mapping_function[i] = round(255 * (cdf[i] / total_pixel))

# 使用映射函數均衡化灰階圖片
# 利用 Numpy 的陣列索引特性，使得每一個像素值在 img_to_gray 中都會被其對應的映射值所替換
img_after_equalized = equalized_mapping_function[img_to_gray]

# 計算均衡化後的灰階直方圖
# 遍歷每一個像素
histogram_after_equalized = np.zeros(256, dtype=np.int32)
for i in range(N):
    for j in range(M):
        # 對於圖片中的每個像素值，增加其在直方圖中的計數
        histogram_after_equalized[img_after_equalized[i, j]] += 1

# 創建一個新的圖片窗口，大小為 12x8
plt.figure(figsize=(12, 8))

# 第一個子圖：原始的灰階圖片
plt.subplot(2, 2, 1)
plt.imshow(img_to_gray, cmap="gray")  # 用灰階顏色映射來顯示圖片
plt.title("Original Grayscale Image")  # 為子圖設置標題
plt.axis("off")  # 不顯示坐標軸

# 第二個子圖：均衡化後的灰階圖片
plt.subplot(2, 2, 2)
plt.imshow(img_after_equalized, cmap="gray")
plt.title("Equalized Grayscale Image")
plt.axis("off")

# 第三個子圖：原始圖片的直方圖
plt.subplot(2, 2, 3)
plt.bar(range(256), histogram, color="blue", width=2)  # 以藍色的柱狀圖顯示，寬度為 2
plt.title("Original Grayscale Histogram")
plt.xlabel("Pixel Value")  # 設置 X 軸的標籤
plt.ylabel("Frequency")  # 設置 Y 軸的標籤

# 第四個子圖：均衡化後的灰階直方圖
plt.subplot(2, 2, 4)
plt.bar(range(256), histogram_after_equalized, color="blue", width=2)
plt.title("Equalized Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# 調整子圖之間的間距，使其看起來更整齊
plt.tight_layout()
# 顯示整個圖片窗口
plt.show()
