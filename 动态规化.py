import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import os
from sklearn.neighbors import KDTree
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import time

# ========== Step 1: Voxelization ==========
def voxelize_points(points, voxel_size):
    print(" 正在进行体素化...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    voxel_centers = []
    voxel_map = {}
    voxel_density = {}

    for voxel in tqdm(voxel_grid.get_voxels(), desc="体素中心计算"):
        idx = voxel.grid_index
        center = np.array(idx) * voxel_size + voxel_grid.origin + voxel_size / 2
        key = tuple(np.round(center, 3))
        voxel_centers.append(center)
        voxel_map[key] = []
        voxel_density[key] = 0

    for pt in tqdm(points, desc="体素点填充"):
        key = tuple(np.round(np.floor(pt / voxel_size) * voxel_size + voxel_size / 2, 3))
        if key in voxel_map:
            voxel_map[key].append(pt)
            voxel_density[key] += 1

    return np.array(voxel_centers), voxel_map, voxel_density

# ========== Step 2: Compute Directions ==========
# 全局缓存变量
_GLOBAL_voxel_centers = None
_GLOBAL_neighbors_all = None

def compute_single_direction_shared(i):
    center = _GLOBAL_voxel_centers[i]
    neighbors = _GLOBAL_voxel_centers[_GLOBAL_neighbors_all[i]]
    cov = np.cov(neighbors.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return tuple(np.round(center, 3)), (eigvals, eigvecs)

def compute_local_directions_fast(voxel_centers, k=10):
    """
    批量计算局部主方向（替代 compute_local_directions_optimized）
    voxel_centers: (N,3) 体素中心坐标
    k: 每个点的邻居数
    """
    print(f"开始批量计算局部主方向，共 {len(voxel_centers)} 个体素，k={k}")
    t0 = time.time()

    # 1. 建立 KDTree 查邻居
    tree = cKDTree(voxel_centers)
    neighbors_idx = tree.query(voxel_centers, k=k)[1]  # shape = (N, k)

    # 2. 获取所有邻居坐标
    all_neighbors = voxel_centers[neighbors_idx]  # (N, k, 3)

    # 3. 去中心化
    means = np.mean(all_neighbors, axis=1, keepdims=True)  # (N, 1, 3)
    centered = all_neighbors - means  # (N, k, 3)

    # 4. 批量计算协方差矩阵
    # covs[i] = (centered[i].T @ centered[i]) / (k - 1)
    covs = np.einsum('nij,nik->nkj', centered, centered) / (k - 1)  # (N, 3, 3)

    # 5. 批量求特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(covs)  # 自动按升序返回

    # 6. 存成字典，键为体素中心坐标（四舍五入避免浮点误差）
    directions = {
        tuple(np.round(center, 3)): (eigvals[i], eigvecs[i])
        for i, center in enumerate(voxel_centers)
    }

    print(f"批量主方向计算完成，用时 {time.time() - t0:.2f} 秒")
    return directions
# ========== Step 3: Compute Features and Energy ==========
def compute_trace(eigvals):
    return np.sum(eigvals)

def compute_E(eigvals):
    A, B, C = eigvals
    return ((A * B - A * B) + (A * C - A * C) + (B * C - B * C)) / (A + B + C + 1e-6)

def compute_energy(data_term, smoothness_term, eta=0.6):
    return eta * data_term + smoothness_term

def compute_smoothness(dir1, dir2):
    cos_theta = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-6)
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    return np.sin(angle)

# ========== Step 4: Build Energy Graph ==========
def nearest_voxel_center(seed, voxel_centers):
    tree = KDTree(voxel_centers)
    dist, idx = tree.query([seed], k=1)
    nearest = voxel_centers[idx[0][0]]
    return tuple(np.round(nearest, 3))
def process_edges_for_voxel(i, voxel_centers, principal_dirs, eigvals_array, k, eta):
    # 子进程中自己构建 KDTree，避免序列化问题
    tree = KDTree(voxel_centers)

    center = voxel_centers[i]
    neighbors_idx = tree.query(center.reshape(1, -1), k=k)[1][0]

    dir1 = principal_dirs[i]
    eigvals1 = eigvals_array[i]

    trace = compute_trace(eigvals1)
    E_val = compute_E(eigvals1)
    D = 1.0 / (trace + E_val + 1e-6)

    row, col, weight = [], [], []
    for j in neighbors_idx:
        if i == j:
            continue
        dir2 = principal_dirs[j]
        S = compute_smoothness(dir1, dir2)
        T = compute_energy(D, S, eta)

        row.append(i)
        col.append(j)
        weight.append(T)

    return row, col, weight

def build_sparse_energy_graph(voxel_centers, directions, k=15, eta=0.8, n_jobs=8):
    N = len(voxel_centers)
    principal_dirs = np.array([directions[tuple(np.round(c, 3))][1][:, -1] for c in voxel_centers])
    eigvals_array = np.array([directions[tuple(np.round(c, 3))][0] for c in voxel_centers])

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_edges_for_voxel)(
            i, voxel_centers, principal_dirs, eigvals_array, k, eta
        ) for i in range(N)
    )

    rows, cols, weights = [], [], []
    for r, c, w in results:
        rows.extend(r)
        cols.extend(c)
        weights.extend(w)

    graph = csr_matrix((weights, (rows, cols)), shape=(N, N))
    return graph


# ========== Step 5: Select Seed Points ==========
def process_single_grid(grid_pts,
                        local_top_percent,
                        cluster_radius,
                        min_cluster_size):
    seeds = []

    if len(grid_pts) < 10:
        return seeds

    grid_pts = np.array(grid_pts, dtype=np.float32)
    sorted_idx = np.argsort(-grid_pts[:, 2])
    top_n = max(int(len(grid_pts) * local_top_percent), min_cluster_size)
    top_pts = grid_pts[sorted_idx[:top_n]]

    if len(top_pts) < min_cluster_size:
        return seeds

    xy = top_pts[:, :2]
    tree = KDTree(xy)
    visited = np.zeros(len(top_pts), dtype=bool)
    labels = -np.ones(len(top_pts), dtype=int)
    cluster_id = 0

    for i in range(len(top_pts)):
        if visited[i]:
            continue
        idx = tree.query_radius([xy[i]], r=cluster_radius)[0]
        if len(idx) < min_cluster_size:
            continue

        queue = list(idx)
        visited[queue] = True
        labels[queue] = cluster_id

        while queue:
            current = queue.pop()
            neighbors = tree.query_radius([xy[current]], r=cluster_radius)[0]
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    labels[n] = cluster_id
                    queue.append(n)

        cluster_id += 1

    for cid in range(cluster_id):
        cluster_pts = top_pts[labels == cid]
        if len(cluster_pts) == 0:
            continue
        topmost = cluster_pts[np.argmax(cluster_pts[:, 2])]
        seeds.append(tuple(np.round(topmost, 3)))

    return seeds

def extract_seed_points_by_local_grid(points,
                                      grid_size=1.0,
                                      local_top_percent=0.5,
                                      cluster_radius=0.3,
                                      min_cluster_size=5,
                                      n_jobs=8):
    """
    将XY平面划分为网格，每个网格内部提取top百分比的高点，
    并在XY上聚类选出每个聚类z最大的点作为种子点。
    使用 joblib 并行加速。
    """
    start_time = time.time()
    points = points.astype(np.float32)
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    grid_dict = defaultdict(list)

    for pt in points:
        gx = int((pt[0] - min_x) // grid_size)
        gy = int((pt[1] - min_y) // grid_size)
        grid_dict[(gx, gy)].append(pt)

    grid_items = list(grid_dict.items())

    # 用 tqdm + joblib 并行处理
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_grid)(
            grid_pts,
            local_top_percent,
            cluster_radius,
            min_cluster_size
        ) for _, grid_pts in tqdm(grid_items, desc="Processing grids (parallel)")
    )

    # 合并所有网格提取出的种子点
    seeds = [pt for seed_list in results for pt in seed_list]

    end_time = time.time()
    print(f"Seed extraction completed in {end_time - start_time:.2f} seconds.")
    print(f"Total seed points: {len(seeds)}")
    return seeds

# ========== Step 6: Find Trunk Path for Each Seed ==========
def is_path_valid(path, angle_threshold=30):
    if len(path) < 15:
        return False

    vecs = np.diff(np.array(path), axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    directions = vecs / (norms + 1e-6)
    avg_dir = np.mean(directions, axis=0)
    avg_norm = np.linalg.norm(avg_dir)

    if avg_norm < 1e-3:
        return False  # 避免方向平均值太小（不稳定）

    avg_dir /= avg_norm
    end_dir = directions[-1]

    cos_theta = np.dot(end_dir, avg_dir)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return angle < angle_threshold

def find_trunk_path(G, start_voxel, angle_threshold=30):
    try:
        lengths, paths = nx.single_source_dijkstra(G, source=start_voxel)

        best_path = []
        best_score = -1

        for end_voxel, candidate_path in paths.items():
            if len(candidate_path) < 15:
                continue
            if not is_path_valid(candidate_path, angle_threshold=angle_threshold):
                continue
            if len(candidate_path) > best_score:
                best_score = len(candidate_path)
                best_path = candidate_path

        return best_path
    except Exception as e:
        print(f"[Warning] Failed to find path from {start_voxel}: {e}")
        return []


# ========== Step 7: Visualization ==========
def visualize_trunks(points, paths):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    trunk_pcds = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    for i, path in enumerate(paths):
        trunk_points = np.array(path)
        trunk_pcd = o3d.geometry.PointCloud()
        trunk_pcd.points = o3d.utility.Vector3dVector(trunk_points)
        trunk_pcd.paint_uniform_color(colors[i % len(colors)])
        trunk_pcds.append(trunk_pcd)

    o3d.visualization.draw_geometries([pcd] + trunk_pcds)


# ========== Step 8: Save Trunk Paths ==========
def save_paths_as_pointclouds(paths, output_dir=r"E:\0\NIBIO_MLS\plot89_txt\axis", voxel_size=0.05):
    os.makedirs(output_dir, exist_ok=True)
    for i, path in enumerate(paths):
        trunk_points = np.array(path)
        if len(trunk_points) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(trunk_points)
        # 给路径上色
        color = np.array([[1.0, 0.0, 0.0]])  # 红色
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(trunk_points), 1)))
        o3d.io.write_point_cloud(f"{output_dir}/trunk_path_{i+1}.ply", pcd)



# ========== 主函数调用 ==========
if __name__ == "__main__":
    input_path = r" "  # 也可以是 .txt：每行 x y z（可带更多列，只取前三列）

    if input_path.lower().endswith((".las", ".laz")):
        import laspy
        las = laspy.read(input_path)
        points = np.vstack([las.x, las.y, las.z]).T
    else:
        with open(input_path, "r") as f:
            first = f.readline()
        delimiter = "," if "," in first else None   # 逗号分隔或空格/Tab 分隔都可以
        points = np.loadtxt(input_path, delimiter=delimiter, usecols=(0, 1, 2))

    voxel_size = 0.1
    voxel_centers, voxel_map, voxel_density = voxelize_points(points, voxel_size)
    directions = compute_local_directions_fast(voxel_centers, k=10)
    graph = build_sparse_energy_graph(voxel_centers, directions, k=12, eta=0.85, n_jobs=8)
    voxel_index_map = {tuple(np.round(center, 3)): idx for idx, center in enumerate(voxel_centers)}
    index_voxel_map = {idx: center for idx, center in enumerate(voxel_centers)}
    seed_voxels = extract_seed_points_by_local_grid(
        points,
        grid_size=1.0,  # 每个网格 4x4 米
        local_top_percent=0.5,  # 局部区域内取前30%高点
        cluster_radius=0.3,
        min_cluster_size=10
    )

    print(f"Selected {len(seed_voxels)} seed points.")
    all_paths = []
    tree = KDTree(voxel_centers)  # 提前建立 KDTree，避免多次重复建立
    start_time = time.time()

    for i, seed in enumerate(tqdm(seed_voxels, desc=" Finding trunk paths")):
        start_voxel = nearest_voxel_center(seed, voxel_centers)
        if start_voxel not in voxel_index_map:
            print(f"[{i + 1}]  Nearest voxel {start_voxel} not in graph. Skipping.")
            continue

        path = find_trunk_path(graph, start_voxel, angle_threshold=35)  # 或换成更强的评分版函数

        if len(path) > 1:
            print(f"[{i + 1}]  Path found, length = {len(path)}")
            all_paths.append(path)
        else:
            print(f"[{i + 1}]  No valid path from seed {seed}.")

    print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")

    # 保存路径
    save_paths_as_pointclouds(all_paths)

    # 可视化
    visualize_trunks(points, all_paths)
