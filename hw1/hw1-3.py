import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

dir_name = "./"  # 設定目錄名稱
# 設定圖片檔案名稱
img1_name = "hw1-3-1.jpg"
img2_name = "hw1-3-2.jpg"
# 使用 os.path.join 組合路徑，這樣可以確保在不同作業系統上都能正確組合
img1_path = os.path.join(dir_name, img1_name)
img2_path = os.path.join(dir_name, img2_name)
# 使用 cv2 的 imread 函數讀取圖片
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
img1 = cv2.resize(
    img1, (img1.shape[1] * 2, img1.shape[0] * 2), interpolation=cv2.INTER_LINEAR
)
# 將彩色圖片轉換為灰階圖片
img1_to_gray = (
    0.299 * img1[:, :, 2] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 0]
).astype(np.uint8)
img2_to_gray = (
    0.299 * img2[:, :, 2] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 0]
).astype(np.uint8)

sift = cv2.SIFT_create()  # 初始化 SIFT 特徵檢測器
# 在第一張圖片上，使用 SIFT 特徵提取器來檢測關鍵點並計算描述子
kp1, des1 = sift.detectAndCompute(img1_to_gray, None)

height, width = img2_to_gray.shape  # 獲取第二張圖片的尺寸
# 創建三個遮罩，每個遮罩涵蓋圖片的三分之一區域
# 第一個遮罩涵蓋圖片的頂部三分之一
mask1 = np.zeros((height, width), dtype=np.uint8)
mask1[: height // 3, :] = 1
# 第二個遮罩涵蓋圖片的中部三分之一
mask2 = np.zeros((height, width), dtype=np.uint8)
mask2[height // 3 : 2 * height // 3, :] = 1
# 第三個遮罩涵蓋圖片的底部三分之一
mask3 = np.zeros((height, width), dtype=np.uint8)
mask3[2 * height // 3 :, :] = 1
# 使用這三個遮罩分別在第二張圖片上提取 SIFT 特徵
kp2_1, des2_1 = sift.detectAndCompute(img2_to_gray, mask1)
kp2_2, des2_2 = sift.detectAndCompute(img2_to_gray, mask2)
kp2_3, des2_3 = sift.detectAndCompute(img2_to_gray, mask3)


def BF_matcher(des1, des2, threshold=0.5):
    matches = []  # 用於存儲匹配組的列表
    for i, d1 in enumerate(des1):  # 遍歷 des1 中的每個描述符
        distances = []  # 存儲當前描述符 d1 和 des2 中每個描述符之間的距離
        for j, d2 in enumerate(des2):  # 遍歷 des2 中的每個描述符
            # 計算 d1 和 d2 之間的歐幾里得距離，並與其索引一起存儲
            distances.append((np.linalg.norm(d1 - d2), j))
        # 依照匹配組距離，由小到大排序 distances
        distances.sort(key=lambda x: x[0])
        min = distances[0]  # distances 中最小距離之匹配組
        # 如果最小距離小於第二短距離的閾值倍數
        if min[0] < threshold * distances[1][0]:
            # 將匹配組加到列表中
            matches.append((i, min[1], min[0]))
    # 返回前 20 組距離最小的匹配
    return sorted(matches, key=lambda x: x[2])[:20]


# 找 des1 和三個描述符集 des2_1、des2_2 和 des2_3 之間的匹配
matches1 = BF_matcher(des1, des2_1)
matches2 = BF_matcher(des1, des2_2)
matches3 = BF_matcher(des1, des2_3)


def create_dmatches(matches, shift):
    dmatches = []  # 用於存儲 DMatch 物件的列表
    for match in matches:
        query_idx, train_idx, _ = match  # 取得 query_idx 和 train_idx 索引
        dmatch = cv2.DMatch()  # 創建一個新的 DMatch 物件
        dmatch.queryIdx = query_idx  # 設定查詢索引
        dmatch.trainIdx = train_idx + shift  # 設定訓練索引，並加上指定的偏移量
        dmatches.append(dmatch)  # 將 DMatch 物件加到列表中
    return dmatches


# 創建並合併三組 DMatch 列表
dmatches = (
    create_dmatches(matches1, 0)  # 第一組匹配，無需偏移
    + create_dmatches(matches2, len(kp2_1))  # 第二組匹配，偏移量為第一組關鍵點的數量
    + create_dmatches(matches3, len(kp2_1) + len(kp2_2))  # 第三組匹配，偏移量為前兩組關鍵點的總數
)

# 使用 cv2.drawMatchesKnn() 繪製匹配結果
# 這個函數將兩張圖片並排顯示，並用線條連接匹配的關鍵點
img_matches = cv2.drawMatchesKnn(
    img1_to_gray,  # 第一張圖片
    kp1,  # 第一張圖片的關鍵點
    img2_to_gray,  # 第二張圖片
    kp2_1 + kp2_2 + kp2_3,  # 串聯第二張圖片的關鍵點
    [dmatches],  # 匹配的 DMatch 列表
    None,
    matchColor=(0, 0, 255),  # 使用紅色線條表示匹配
)

# 將 img_matches 從 BGR 色彩空間轉換為 RGB 色彩空間
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
# 顯示 img_matches
plt.figure(figsize=(12, 8))
plt.imshow(img_matches_rgb)
plt.title("SIFT(2.0x)")
plt.axis("off")
plt.show()
