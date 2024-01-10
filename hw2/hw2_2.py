import cv2
import numpy as np
import os


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global corner_list
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_list) < 4:
            corner_list.append((x, y))


def Find_Homography(src, tar):
    A = []  # 初始化矩陣 A，用於構建線性方程組

    # 遍歷點對，構建方程組中的矩陣 A
    for p, p_ in zip(src, tar):
        x, y = p  # 源點座標
        x_, y_ = p_  # 目標點座標
        # 對於每個點對，根據單應性矩陣的定義構建兩行線性方程
        line1 = [x, y, 1, 0, 0, 0, -x_ * x, -x_ * y, -x_]
        A.append(line1)  # 添加到矩陣 A
        line2 = [0, 0, 0, x, y, 1, -y_ * x, -y_ * y, -y_]
        A.append(line2)  # 添加到矩陣 A

    A = np.array(A)  # 將 A 轉換為 numpy 陣列形式

    # 使用奇異值分解（SVD）解線性方程組
    U, D, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)  # 解是 VT 的最後一行，並將其重新塑形為 3x3 矩陣

    return H  # 返回單應性矩陣 H


def inverse_mapping(img_src, img_tar, H):
    # 計算變換矩陣 H 的逆矩陣 H_inv
    H_inv = np.linalg.inv(H)
    # 獲取目標影像 tar 的長寬尺寸
    tar_Y, tar_X, _ = img_tar.shape
    # 創建目標影像的座標網格
    X, Y = np.meshgrid(np.arange(tar_X), np.arange(tar_Y))
    # 利用逆變換矩陣 H_inv 對座標點進行映射
    coords = np.dot(H_inv, np.stack((X.ravel(), Y.ravel(), np.ones(tar_X * tar_Y))))
    # 將齊次座標轉換成笛卡爾座標，即將其除以第三個座標值
    coords /= coords[2]
    # 取出映射後的 x, y 座標
    coords = coords[:2, :]
    # 獲取原始影像 src 的長寬尺寸
    src_Y, src_X, _ = img_src.shape
    # 創建一個布林陣列 mask，用於標記映射後座標點是否落在原始影像的範圍內
    mask = (
        (coords[0, :] >= 0)
        & (coords[0, :] < src_X)
        & (coords[1, :] >= 0)
        & (coords[1, :] < src_Y)
    ).reshape(tar_Y, tar_X)

    # 分別計算 x, y 的整數座標和小數部分
    x = coords[0, :].reshape(tar_Y, tar_X)
    y = coords[1, :].reshape(tar_Y, tar_X)
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    # 確保座標不會超出原始影像的邊界
    x2 = np.minimum(x1 + 1, src_X - 1)
    y2 = np.minimum(y1 + 1, src_Y - 1)

    # 計算雙線性插值的權重
    dx = x - x1
    dy = y - y1
    w1 = (1 - dx) * (1 - dy)
    w2 = dx * (1 - dy)
    w3 = (1 - dx) * dy
    w4 = dx * dy

    # 從 mask 中篩選出有效的 y1, x1, y2, x2 座標
    valid_x1 = x1[mask]
    valid_y1 = y1[mask]
    valid_x2 = x2[mask]
    valid_y2 = y2[mask]

    # 利用雙線性插值根據權重和有效座標計算目標影像 tar 的像素值
    img_tar[mask] = (
        w1[mask][..., np.newaxis] * img_src[valid_y1, valid_x1]
        + w2[mask][..., np.newaxis] * img_src[valid_y1, valid_x2]
        + w3[mask][..., np.newaxis] * img_src[valid_y2, valid_x1]
        + w4[mask][..., np.newaxis] * img_src[valid_y2, valid_x2]
    )


def compute_vanishing_point(corner_list):
    # 從角點列表中解包四個角點 A, B, C, D
    [A, B, C, D] = corner_list
    # 計算直線 AB 的座標表示的外積，得到直線的參數
    v1 = np.cross([A[0], A[1], 1], [B[0], B[1], 1])
    # 計算直線 CD 的座標表示的外積，得到直線的參數
    v2 = np.cross([C[0], C[1], 1], [D[0], D[1], 1])
    # 計算兩條直線的外積，得到兩直線的交點，即消失點
    vp = np.cross(v1, v2)
    # 將座標轉換成笛卡爾座標，即將其除以第三個座標值
    vp = vp / vp[2]

    # 返回消失點的整數座標 (x, y)
    return (int(vp[0]), int(vp[1]))


if __name__ == "__main__":
    dir_name = "./assets/"
    img_src = cv2.imread(os.path.join(dir_name, "post.png"))
    src_Y, src_X, _ = img_src.shape
    img_tar = cv2.imread(os.path.join(dir_name, "display.jpg"))
    tar_Y, tar_X, _ = img_tar.shape
    cv2.namedWindow("Interative window")
    cv2.setMouseCallback("Interative window", mouse_callback)
    corner_list = []
    corner_list_src = [(0, 0), (src_X, 0), (0, src_Y), (src_X, src_Y)]

    while True:
        fig = img_tar.copy()
        key = cv2.waitKey(1) & 0xFF

        if len(corner_list) == 4:
            # implement the inverse homography mapping and bi-linear qinterpolation
            # print(corner_list_src)
            # print(corner_list)
            H = Find_Homography(corner_list_src, corner_list)
            # print(H)

            inverse_mapping(img_src, fig, H)
            for p in corner_list:
                cv2.circle(fig, p, 5, (0, 0, 255), -1)

            # 計算消失點
            vp = compute_vanishing_point(corner_list)
            # 如果消失點的座標在目標圖像的範圍內
            if (0 <= vp[0] < tar_X) & (0 <= vp[1] < tar_Y):
                # 在圖像上繪製一個紅色的圓來標示消失點
                cv2.circle(fig, vp, 5, (0, 0, 255), -1)

        # quit
        if key == ord("q"):
            break

        # reset the corner_list
        if key == ord("r"):
            corner_list = []

        # show the corner list
        if key == ord("p"):
            print(corner_list)

        cv2.imshow("Interative window", fig)
        if len(corner_list) == 4:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    output_dir = "./output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(os.path.join(output_dir, "homography.png"), fig)
    cv2.destroyAllWindows()
