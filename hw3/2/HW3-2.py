import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import time

dir = "./2"
output_dir = "./2/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
img_name = "2-image"
img = cv2.imread(os.path.join(dir, img_name + ".jpg"), cv2.IMREAD_COLOR)


def save(img, image_name):
    cv2.imwrite(os.path.join(output_dir, image_name), img)


def kmeans(img):
    K = [5, 7, 9]
    H, W, _ = img.shape
    n_pixels = H * W
    sample_pool = img.reshape((-1, 3))

    for k in K:
        best_ssd = float("inf")
        best_segmented_img = None

        for it in range(3):
            random_indices = np.random.randint(0, n_pixels, size=k)
            centroids = sample_pool[random_indices].astype(np.float64)
            nearest_centroids = np.zeros((H, W), dtype=int)
            while True:
                distances = np.sqrt(
                    ((img - centroids[:, np.newaxis, np.newaxis]) ** 2).sum(axis=3)
                )
                new_nearest_centroids = np.argmin(distances, axis=0)
                if np.array_equal(new_nearest_centroids, nearest_centroids):
                    break
                nearest_centroids = new_nearest_centroids
                for i in range(k):
                    points = img[nearest_centroids == i]
                    if points.size > 0:
                        centroids[i] = np.mean(points, axis=0)
                print(f"K = {k}, iteration = {it + 1}\n", centroids)
            ssd = np.sum((img - centroids[nearest_centroids]) ** 2) / n_pixels
            if ssd < best_ssd:
                best_ssd = ssd
                best_segmented_img = centroids[nearest_centroids].astype(np.uint8)

        save(best_segmented_img, f"{img_name}_kmeans_K={k}.jpg")


def initialize_kmeans_plusplus(img, k):
    H, W, _ = img.shape
    n_pixels = H * W
    sample_pool = img.reshape((-1, 3))
    first_sample_idx = np.random.choice(n_pixels)
    centroids = np.array([sample_pool[first_sample_idx]], dtype=np.float64)
    for _ in range(1, k):
        distances = np.min(
            np.sqrt(((sample_pool - centroids[:, np.newaxis]) ** 2).mean(axis=2)),
            axis=0,
        )
        probabilities = distances**2
        probabilities /= probabilities.sum()
        next_sample_idx = np.random.choice(n_pixels, p=probabilities)
        centroids = np.append(centroids, [sample_pool[next_sample_idx]], axis=0)

    return centroids


def kmeans_plusplus(img):
    K = [5, 7, 9]
    H, W, _ = img.shape
    for k in K:
        centroids = initialize_kmeans_plusplus(img, k)
        nearest_centroids = np.zeros((H, W), dtype=int)
        while True:
            distances = np.sqrt(
                ((img - centroids[:, np.newaxis, np.newaxis]) ** 2).sum(axis=3)
            )
            new_nearest_centroids = np.argmin(distances, axis=0)
            if np.array_equal(new_nearest_centroids, nearest_centroids):
                break
            nearest_centroids = new_nearest_centroids
            for i in range(k):
                points = img[nearest_centroids == i]
                if points.size > 0:
                    centroids[i] = np.mean(points, axis=0)
            print(f"K = {k}\n", centroids)

        segmented_img = centroids[nearest_centroids].astype(np.uint8)
        save(segmented_img, f"{img_name}_kmeans_plusplus_K={k}.jpg")


def mean_shift(img, bandwidth):
    H, W, _ = img.shape
    data_points = img.reshape(-1, 3)
    tree = KDTree(data_points)
    mean_shift_points = np.copy(data_points)
    epoch = 0
    while True:
        next_mean_shift_points = np.zeros_like(mean_shift_points)
        for i, point in enumerate(mean_shift_points):
            in_window = tree.query_ball_point(point, bandwidth)
            points_in_window = data_points[in_window]
            if len(points_in_window) > 0:
                next_mean_shift_points[i] = np.mean(points_in_window, axis=0)
            else:
                next_mean_shift_points[i] = point
        if np.linalg.norm(next_mean_shift_points - mean_shift_points) < 1e-3:
            break
        print(mean_shift_points.dtype)
        mean_shift_points = next_mean_shift_points
        epoch += 1
        print(f"Epoch = {epoch}\n")
    print("Down!")

    return mean_shift_points.reshape(H, W, 3)


def downsample(img, scale_factor):
    H, W, _ = img.shape

    return cv2.resize(
        img,
        (int(W * scale_factor), int(H * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )


def process_point(args):
    point, data_points, tree, bandwidth = args
    in_window = tree.query_ball_point(point, bandwidth)
    points_in_window = data_points[in_window]

    return np.mean(points_in_window, axis=0) if len(points_in_window) > 0 else point


def mean_shift_optimized(img, bandwidth):
    H, W, _ = img.shape
    data_points = img.reshape(-1, 3)
    tree = KDTree(data_points)
    mean_shift_points = np.copy(data_points)
    with Pool() as pool:
        epoch = 0
        while True:
            args = [
                (point, data_points, tree, bandwidth) for point in mean_shift_points
            ]
            next_mean_shift_points = np.array(pool.map(process_point, args))
            if np.linalg.norm(next_mean_shift_points - mean_shift_points) < 1e-3:
                break
            mean_shift_points = next_mean_shift_points
            epoch += 1
            print(f"Epoch = {epoch}")
        print("Down!")

    return mean_shift_points.reshape(H, W, 3).astype(np.uint8)


def plot_rgb_feature_space(data, title):
    if data.ndim == 3:
        data_points = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    else:
        data_points = data.reshape(-1, 3)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data_points[:, 0], data_points[:, 1], data_points[:, 2], c=data_points / 255.0
    )
    ax.set_title(title)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    plt.savefig(os.path.join(output_dir, title))


def prune_modes(cluster_img, threshold):
    data_points = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    distances = cdist(data_points, data_points)
    assigned = np.zeros(len(data_points), dtype=bool)
    clusters = []
    for i, _ in enumerate(data_points):
        if assigned[i]:
            continue
        neighbours = np.where((distances[i] <= threshold) & ~assigned)[0]
        cluster = np.mean(data_points[neighbours], axis=0)
        clusters.append(cluster)
        assigned[neighbours] = True
    print(f"Total {len(clusters)} clusters")

    return np.array(clusters, dtype=np.uint8)


def assign_pixels_to_clusters(img, cluster_centers, title):
    H, W, _ = img.shape
    data_points = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    tree = KDTree(cluster_centers)
    labels = tree.query(data_points, k=1)[1]
    assigned_cluster_img = np.array([cluster_centers[label] for label in labels])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data_points[:, 0],
        data_points[:, 1],
        data_points[:, 2],
        c=assigned_cluster_img / 255.0,
    )
    ax.set_title(title)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    plt.savefig(os.path.join(output_dir, title))

    return cv2.cvtColor(assigned_cluster_img.reshape(H, W, 3), cv2.COLOR_RGB2BGR)


def run_mean_shift():
    scale_factor = 0.1
    bandwidth = 30
    downsample_image = downsample(img, scale_factor)
    start_time = time.perf_counter()
    cluster_img = mean_shift_optimized(downsample_image, bandwidth)
    print(f"mean_shift: time = {round(time.perf_counter() - start_time)}s")
    plot_rgb_feature_space(img, f"{img_name}_ori_rgb_fs.jpg")
    plot_rgb_feature_space(cluster_img, f"{img_name}_rgb_fs_ms_bw={bandwidth}.jpg")
    cluster_centers = prune_modes(cluster_img, bandwidth)
    plot_rgb_feature_space(
        cluster_centers, f"{img_name}_rgb_fs_ms_p_bw={bandwidth}.jpg"
    )
    assigned_cluster_img = assign_pixels_to_clusters(
        img, cluster_centers, f"{img_name}_rgb_fs_ms_p_final_bw={bandwidth}.jpg"
    )
    save(assigned_cluster_img, f"{img_name}_mean_shift_bw={bandwidth}.jpg")


def mean_shift_plus(img, bandwidth):
    H, W, _ = img.shape
    data_points = img.reshape(H * W, 3)
    tree = KDTree(data_points)
    is_visited = np.zeros(data_points.shape[0], dtype=bool)
    cluster_centers = []
    epoch = 0
    while not is_visited.all():
        unvisited_indices = np.where(~is_visited)[0]
        random_index = np.random.choice(unvisited_indices)
        point = data_points[random_index]
        while True:
            in_window = tree.query_ball_point(point, bandwidth)
            points_in_window = data_points[in_window]
            mean_shift = np.mean(points_in_window, axis=0)
            is_visited[in_window] = True
            if np.linalg.norm(mean_shift - point) < 1e-3:
                break
            point = mean_shift
        has_merged = False
        for center in cluster_centers:
            if np.linalg.norm(center - point) < 1e-3:
                has_merged = True
                break
        if not has_merged:
            cluster_centers.append(point)
        epoch = epoch + 1
        print(f"Epoch = {epoch}")
    print("Down!")
    print(f"Total {len(cluster_centers)} clusters")
    tree = KDTree(cluster_centers)
    labels = tree.query(data_points, k=1)[1]

    cluster_img = np.array([cluster_centers[label] for label in labels])
    return cluster_img.reshape(H, W, 3).astype(np.uint8)


def run_mean_shift_plus():
    bandwidth = 10
    start_time = time.perf_counter()
    clustered_img = mean_shift_plus(img, bandwidth)
    print(f"mean_shift: time = {round(time.perf_counter() - start_time)}s")
    plot_rgb_feature_space(clustered_img, f"{img_name}_rgb_fs_ms_plus_bw={bandwidth}")
    save(clustered_img, f"{img_name}_mean_shift_plus_bw={bandwidth}.jpg")


def mean_shift_with_spatial(img, bandwidth):
    H, W, _ = img.shape
    X, Y = np.meshgrid(range(W), range(H))
    data_points = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2], X, Y), axis=-1)
    data_points = data_points.reshape(H * W, 5).astype(np.float16)
    data_points[:, :3] /= 255.0
    data_points[:, 3:] /= np.array([W, H])
    tree = KDTree(data_points)
    mean_shift_points = np.copy(data_points)
    with Pool() as pool:
        epoch = 0
        while True:
            args = [
                (point, data_points, tree, bandwidth) for point in mean_shift_points
            ]
            next_mean_shift_points = np.array(pool.map(process_point, args))
            if np.linalg.norm(next_mean_shift_points - mean_shift_points) < 1e-3:
                break
            mean_shift_points = next_mean_shift_points
            epoch += 1
            print(f"Epoch = {epoch}")
    print("Down!")

    return mean_shift_points, (mean_shift_points[:, :3] * 255).reshape(H, W, 3).astype(
        np.uint8
    )


def prune_modes_with_spatial(result, threshold):
    data_points = result
    distances = cdist(data_points, data_points)
    assigned = np.zeros(len(data_points), dtype=bool)
    clusters = []
    for i, _ in enumerate(data_points):
        if assigned[i]:
            continue
        neighbours = np.where((distances[i] <= threshold) & ~assigned)[0]
        cluster = np.mean(data_points[neighbours], axis=0)
        clusters.append(cluster)
        assigned[neighbours] = True
    print(f"Total {len(clusters)} clusters")

    return np.array(clusters, dtype=np.float16)


def assign_pixels_to_clusters_with_spatial(img, cluster_centers):
    H, W, _ = img.shape
    X, Y = np.meshgrid(range(W), range(H))
    data_points = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2], X, Y), axis=-1)
    data_points = data_points.reshape(H * W, 5).astype(np.float16)
    data_points[:, :3] /= 255.0
    data_points[:, 3:] /= np.array([W, H])
    tree = KDTree(cluster_centers)
    labels = tree.query(data_points, k=1)[1]
    assigned_cluster_img = (
        np.array([cluster_centers[label] for label in labels])[:, :3] * 255
    )

    return assigned_cluster_img.reshape(H, W, 3).astype(np.uint)


def run_mean_shift_with_spatial():
    scale_factor = 0.1
    downsample_image = downsample(img, scale_factor)
    bandwidth = 0.3
    start_time = time.perf_counter()
    mean_shift_result, cluster_img = mean_shift_with_spatial(
        downsample_image,
        bandwidth,
    )
    print(f"mean_shift_with_spatial: time = {round(time.perf_counter() - start_time)}s")
    upsampled_image = cv2.resize(
        cluster_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    save(upsampled_image, f"{img_name}_mean_shift_with_spatial_upsampled.jpg")
    cluster_centers = prune_modes_with_spatial(mean_shift_result, bandwidth)
    assigned_cluster_img = assign_pixels_to_clusters_with_spatial(img, cluster_centers)
    save(assigned_cluster_img, f"{img_name}_mean_shift_with_spatial_pruned.jpg")


if __name__ == "__main__":
    kmeans(img)
    kmeans_plusplus(img)
    run_mean_shift()
    run_mean_shift_plus()
    run_mean_shift_with_spatial()
