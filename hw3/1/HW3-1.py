import numpy as np
import os
import cv2
import random
import time


def load_image(image_name):
    return cv2.imread(os.path.join("./1", image_name), cv2.IMREAD_COLOR)


img = load_image("1-image.jpg")
img1 = load_image("1-book1.jpg")
img2 = load_image("1-book2.jpg")
img3 = load_image("1-book3.jpg")


def color_2_gray(img):
    return (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(
        np.uint8
    )


img_to_gray = color_2_gray(img)
img1_to_gray = color_2_gray(img1)
img2_to_gray = color_2_gray(img2)
img3_to_gray = color_2_gray(img3)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img_to_gray, None)
kp1, des1 = sift.detectAndCompute(img1_to_gray, None)
kp2, des2 = sift.detectAndCompute(img2_to_gray, None)
kp3, des3 = sift.detectAndCompute(img3_to_gray, None)


def BF_matcher(des1, des2, threshold):
    min_dists = {}
    for i, d1 in enumerate(des1):
        distances = []
        for j, d2 in enumerate(des2):
            distances.append((j, np.linalg.norm(d1 - d2)))
        distances.sort(key=lambda x: x[1])
        j, min = distances[0]
        if min < threshold * distances[1][1]:
            if j not in min_dists or min < min_dists[j][1]:
                min_dists[j] = (i, min)
    matches = np.array([(i, j) for j, (i, _) in min_dists.items()])

    return matches


threshold = 0.8
start_time = time.perf_counter()
matches1 = BF_matcher(des1, des, threshold)
print(
    f"matches1: time = {round(time.perf_counter() - start_time)}s, len = {len(matches1)}"
)
start_time = time.perf_counter()
matches2 = BF_matcher(des2, des, threshold)
print(
    f"matches2: time = {round(time.perf_counter() - start_time)}s, len = {len(matches2)}"
)
start_time = time.perf_counter()
matches3 = BF_matcher(des3, des, threshold)
print(
    f"matches3: time = {round(time.perf_counter() - start_time)}s, len = {len(matches3)}"
)


def create_dmatches(matches):
    dmatches = []
    for match in matches:
        query_idx, train_idx = match
        dmatch = cv2.DMatch()
        dmatch.queryIdx = query_idx
        dmatch.trainIdx = train_idx
        dmatches.append(dmatch)
    return dmatches


dmatches1 = create_dmatches(matches1)
dmatches2 = create_dmatches(matches2)
dmatches3 = create_dmatches(matches3)


def draw_matches(img1, kp1, img2, kp2, dmatches):
    return cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        [dmatches],
        None,
        matchColor=(0, 0, 255),
    )


book1_matches = draw_matches(img1, kp1, img, kp, dmatches1)
book2_matches = draw_matches(img2, kp2, img, kp, dmatches2)
book3_matches = draw_matches(img3, kp3, img, kp, dmatches3)
output_dir = "./1/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def save(img, image_name):
    cv2.imwrite(os.path.join(output_dir, image_name), img)


save(book1_matches, f"wo_RANSAC_book1_t={threshold}.jpg")
save(book2_matches, f"wo_RANSAC_book2_t={threshold}.jpg")
save(book3_matches, f"wo_RANSAC_book3_t={threshold}.jpg")


def compute_homography(src_pts, dst_pts):
    A = []
    for p, p_ in zip(src_pts, dst_pts):
        x, y = p
        x_, y_ = p_
        line1 = [x, y, 1, 0, 0, 0, -x_ * x, -x_ * y, -x_]
        A.append(line1)
        line2 = [0, 0, 0, x, y, 1, -y_ * x, -y_ * y, -y_]
        A.append(line2)
    A = np.array(A)
    U, D, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)

    return H


def RANSAC(dmatches, kp1, kp2, R, iterations=3000):
    max_inliers = 0
    best_homography = None
    best_inliers = None
    pt1s = [np.append(np.float32(kp1[d.queryIdx].pt), 1) for d in dmatches]
    pt2s = [np.float32(kp2[d.trainIdx].pt) for d in dmatches]
    for _ in range(iterations):
        random_select = random.sample(dmatches, 4)
        src_pts = np.float32([kp1[s.queryIdx].pt for s in random_select])
        dst_pts = np.float32([kp2[s.trainIdx].pt for s in random_select])
        H = compute_homography(src_pts, dst_pts)
        inliers = 0
        current_inliers = []
        for idx, d in enumerate(dmatches):
            pt1 = pt1s[idx]
            pt2 = pt2s[idx]
            transformed_pt1 = (np.dot(H, pt1) / np.dot(H, pt1)[2])[:2]
            distance = np.linalg.norm(transformed_pt1 - pt2)
            if distance < R:
                inliers += 1
                current_inliers.append(d)
        if inliers > max_inliers:
            max_inliers = inliers
            best_homography = H
            best_inliers = current_inliers

    return best_homography, best_inliers


def draw_frame_left_image(img, bps):
    for i in range(len(bps)):
        cv2.line(img, bps[i], bps[(i + 1) % len(bps)], color=(255, 0, 0), thickness=5)
    return 0


def draw_frame_right_image(img, bps, H):
    transformed_bps = []
    for bp in bps:
        transformed_bp = np.dot(H, np.array([bp[0], bp[1], 1]))
        transformed_bp = (transformed_bp / transformed_bp[2])[:2]
        transformed_bps.append(transformed_bp)
    transformed_bps = np.array(transformed_bps, dtype=int)
    for i in range(len(transformed_bps)):
        cv2.line(
            img,
            tuple(transformed_bps[i]),
            tuple(transformed_bps[(i + 1) % len(transformed_bps)]),
            color=(255, 0, 0),
            thickness=5,
        )

    return img


R = 5
H1, best_inliers1 = RANSAC(dmatches1, kp1, kp, R)
b1 = np.array([(100, 221), (125, 1349), (991, 1334), (1000, 225)])
draw_frame_left_image(img1, b1)
book1_matches = draw_matches(
    img1, kp1, draw_frame_right_image(img.copy(), b1, H1), kp, best_inliers1
)
H2, best_inliers2 = RANSAC(dmatches2, kp2, kp, R)
b2 = np.array([(65, 124), (73, 1353), (1044, 1346), (1042, 112)])
draw_frame_left_image(img2, b2)
book2_matches = draw_matches(
    img2, kp2, draw_frame_right_image(img.copy(), b2, H2), kp, best_inliers2
)
H3, best_inliers3 = RANSAC(dmatches3, kp3, kp, R)
b3 = np.array([(129, 183), (128, 1398), (977, 1390), (997, 178)])
draw_frame_left_image(img3, b3)
book3_matches = draw_matches(
    img3, kp3, draw_frame_right_image(img.copy(), b3, H3), kp, best_inliers3
)
save(book1_matches, f"RANSAC_book1_t={threshold}_R={R}.jpg")
save(book2_matches, f"RANSAC_book2_t={threshold}_R={R}.jpg")
save(book3_matches, f"RANSAC_book3_t={threshold}_R={R}.jpg")


def draw_deviation_vectors(img, kp1, kp2, inliers, H):
    for d in inliers:
        pt1 = np.float32(kp1[d.queryIdx].pt)
        pt2 = np.float32(kp2[d.trainIdx].pt)
        transformed_pt1 = (
            np.dot(H, np.append(pt1, 1)) / np.dot(H, np.append(pt1, 1))[2]
        )[:2]
        cv2.circle(
            img, (int(transformed_pt1[0]), int(transformed_pt1[1])), 1, (255, 0, 0)
        )
        cv2.circle(img, (int(pt2[0]), int(pt2[1])), 1, (255, 0, 0))
        cv2.arrowedLine(
            img,
            (int(transformed_pt1[0]), int(transformed_pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            (0, 0, 255),
            1,
            tipLength=1,
        )


draw_deviation_vectors(img, kp1, kp, best_inliers1, H1)
draw_deviation_vectors(img, kp2, kp, best_inliers2, H2)
draw_deviation_vectors(img, kp3, kp, best_inliers3, H3)
save(img, f"book_deviation_t={threshold}_R={R}.jpg")
