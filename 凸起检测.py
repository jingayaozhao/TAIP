import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import laspy
from matplotlib import cm
import matplotlib as mpl
import hdbscan
from sklearn.decomposition import PCA
import itertools
import csv
from datetime import datetime

# ========== 参数 ==========
input_folder = r" "
output_root = r" "  # 根目录

# 实验参数设置
min_cluster_size_list = [100, 150, 200, 250, 300, 400]
min_samples_list = [3, 5, 8, 12]
epsilon = 0.1  # 固定

os.makedirs(output_root, exist_ok=True)

# ========== 第一步：合并所有PLY（只需执行一次） ==========
print("=" * 60)
print("加载所有点云...")
all_points = []
ply_files = [f for f in os.listdir(input_folder) if f.endswith(".ply")]

for ply in ply_files:
    ply_path = os.path.join(input_folder, ply)
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] > 0:
        all_points.append(pts)

if len(all_points) == 0:
    raise RuntimeError("输入目录中没有点云数据")

points = np.vstack(all_points)
print(f"合并完成，共 {points.shape[0]} 个点")
print("=" * 60)

# ========== 生成参数组合 ==========
param_combinations = list(itertools.product(min_cluster_size_list, min_samples_list))
print(f"\n共 {len(param_combinations)} 组参数组合待测试：")
for i, (mcs, ms) in enumerate(param_combinations):
    print(f"  [{i + 1:2d}] min_cluster_size={mcs}, min_samples={ms}, epsilon={epsilon}")

# ========== 统计结果存储 ==========
experiment_results = []

# ========== 批量处理循环 ==========
for exp_idx, (min_cluster_size, min_samples) in enumerate(param_combinations):

    print("\n" + "=" * 60)
    print(f"[实验 {exp_idx + 1}/{len(param_combinations)}]")
    print(f"参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
    print("=" * 60)

    # 生成输出文件夹名称
    dataset_name = os.path.basename(input_folder).replace("_axis", "")
    output_folder_name = f"{dataset_name}_{min_cluster_size}_{min_samples}"
    output_folder = os.path.join(output_root, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"输出目录: {output_folder}")

    # ========== 第二步：HDBSCAN 聚类 ==========
    print("开始 HDBSCAN 聚类...")
    xyz = points[:, :3]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method='leaf',
        metric='euclidean'
    )

    labels = clusterer.fit_predict(xyz)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    print(f"检测到 {n_clusters} 个簇（可能的树），噪声点数: {noise_count}")

    # ========== 二点五步：簇级别几何特征提取 ==========
    print("开始计算簇几何特征...")

    clusters_info = []
    unique_labels = [l for l in set(labels) if l != -1]

    for lbl in unique_labels:
        pts = points[labels == lbl]
        if pts.shape[0] < 50:
            continue

        # PCA 主方向
        pca = PCA(n_components=3)
        pca.fit(pts)

        direction = pca.components_[0]
        direction = direction / np.linalg.norm(direction)

        zmin = pts[:, 2].min()
        zmax = pts[:, 2].max()
        center = pts.mean(axis=0)

        clusters_info.append({
            "label": lbl,
            "points": pts,
            "dir": direction,
            "zmin": zmin,
            "zmax": zmax,
            "center": center
        })

    print(f"有效参与合并的簇数量: {len(clusters_info)}")

    # ========== 二点六步：构建可合并簇对 ==========
    print("构建簇合并关系图...")

    merge_pairs = []
    angle_threshold = 0.95
    z_gap_threshold = 2.0
    line_dist_thresh = 0.5

    for i in range(len(clusters_info)):
        for j in range(i + 1, len(clusters_info)):
            A = clusters_info[i]
            B = clusters_info[j]

            # 1. 主方向一致性
            cos_angle = abs(np.dot(A["dir"], B["dir"]))
            if cos_angle < angle_threshold:
                continue

            # 2. Z方向连续性
            gap = max(0, max(A["zmin"], B["zmin"]) - min(A["zmax"], B["zmax"]))
            if gap > z_gap_threshold:
                continue

            # 3. 共线性判断（点到线距离）
            vec = B["center"] - A["center"]
            dist_to_line = np.linalg.norm(np.cross(vec, A["dir"]))
            if dist_to_line > line_dist_thresh:
                continue

            merge_pairs.append((i, j))

    print(f"发现 {len(merge_pairs)} 组可合并簇关系")

    # ========== 二点七步：Union-Find 合并簇 ==========
    parent = list(range(len(clusters_info)))


    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x


    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra


    for a, b in merge_pairs:
        union(a, b)

    # 生成最终合并后的簇
    merged_clusters = {}
    for idx in range(len(clusters_info)):
        root = find(idx)
        merged_clusters.setdefault(root, []).append(idx)

    # 构建合并后的点云簇列表
    final_clusters = []
    print("开始生成合并后的轴线簇...")

    for group_indices in merged_clusters.values():
        merged_pts = np.vstack([clusters_info[i]["points"] for i in group_indices])
        final_clusters.append(merged_pts)

    print(f"合并完成，最终轴线簇数量: {len(final_clusters)}")

    # ========== 绘图参数 ==========
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.major.size'] = 4

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # 背景
    sample_indices = np.random.choice(points.shape[0],
                                      min(50000, points.shape[0]),
                                      replace=False)
    background_points = points[sample_indices]
    ax.scatter(background_points[:, 0],
               background_points[:, 1],
               background_points[:, 2],
               c="#D3D3D3", s=0.5, alpha=0.03, edgecolors='none')

    # ========== 第四步：逐簇处理 ==========
    valid_tree_count = 0
    tree_heights = []

    # ========== 处理合并后的轴线簇 ==========
    for idx, cluster_points in enumerate(final_clusters):
        print(f"\n▶ 处理 tree_{idx}（合并后簇），点数：{cluster_points.shape[0]}")

        # ========== PCA 垂直性判别 ==========
        pca = PCA(n_components=3)
        try:
            pca.fit(cluster_points)
            direction = pca.components_[0]
            verticality = abs(direction[2])
        except:
            print(f"  PCA 失败，跳过")
            continue

        if verticality < 0.7:
            print(f"   非垂直结构（verticality={verticality:.2f}）")
            continue

        # ========== 凸包 ==========
        try:
            hull = ConvexHull(cluster_points)
        except Exception as e:
            print(f"  凸包失败：{e}")
            continue

        # ========== 输出 LAS ==========
        outfile = os.path.join(output_folder, f"tree_{idx}.las")
        hdr = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(hdr)

        las.x = cluster_points[:, 0]
        las.y = cluster_points[:, 1]
        las.z = cluster_points[:, 2]
        las.write(outfile)

        valid_tree_count += 1
        min_corner = cluster_points.min(axis=0)
        max_corner = cluster_points.max(axis=0)
        tree_height = max_corner[2] - min_corner[2]
        tree_heights.append(tree_height)

        print(f"  ✅ 保存完成：tree_{idx}")

    print(f"\n处理结束，共输出 {valid_tree_count} 棵有效树")

    # ========== 图表设置 ==========
    ax.set_xlabel('X Coordinate (m)', fontsize=12, labelpad=10, weight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, labelpad=10, weight='bold')
    ax.set_zlabel('Height (m)', fontsize=12, labelpad=10, weight='bold')

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, color='gray')

    ax.view_init(elev=25, azim=45)
    try:
        ax.set_box_aspect([1, 1, 0.8])
    except:
        pass

    ax.set_title(f'HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}\n'
                 f'Clusters: {n_clusters}, Valid Trees: {valid_tree_count}',
                 fontsize=13, pad=20, weight='bold')

    if tree_heights:
        norm = mpl.colors.Normalize(vmin=min(tree_heights), vmax=max(tree_heights))
        sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.12, shrink=0.7, aspect=20)
        cbar.set_label('Tree Height (m)', fontsize=11, weight='bold', labelpad=12)

    plt.tight_layout()

    output_fig_png = os.path.join(output_folder, 'convex_hull_visualization.png')
    output_fig_pdf = os.path.join(output_folder, 'convex_hull_visualization.pdf')

    plt.savefig(output_fig_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_fig_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"\n✅ 可视化图已保存：")
    print(f"   PNG: {output_fig_png}")
    print(f"   PDF: {output_fig_pdf}")

    plt.close(fig)  # 关闭图形，释放内存

    # ========== 记录实验结果 ==========
    experiment_results.append({
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "epsilon": epsilon,
        "initial_clusters": n_clusters,
        "noise_points": noise_count,
        "valid_clusters_for_merge": len(clusters_info),
        "merge_pairs": len(merge_pairs),
        "final_clusters": len(final_clusters),
        "valid_trees": valid_tree_count,
        "output_folder": output_folder_name
    })

# ========== 输出实验汇总报告 ==========
print("\n" + "=" * 60)
print("所有实验完成！生成汇总报告...")
print("=" * 60)

# 保存为CSV
summary_csv = os.path.join(output_root, "experiment_summary.csv")
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ["min_cluster_size", "min_samples", "epsilon",
                  "initial_clusters", "noise_points", "valid_clusters_for_merge",
                  "merge_pairs", "final_clusters", "valid_trees", "output_folder"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in experiment_results:
        writer.writerow(result)

print(f"\n✅ 实验汇总表已保存: {summary_csv}")

# 打印汇总表格
print("\n" + "-" * 100)
print(
    f"{'min_cluster_size':^18} | {'min_samples':^12} | {'初始簇数':^10} | {'噪声点':^10} | {'合并后簇数':^12} | {'有效树':^8}")
print("-" * 100)
for r in experiment_results:
    print(f"{r['min_cluster_size']:^18} | {r['min_samples']:^12} | {r['initial_clusters']:^10} | "
          f"{r['noise_points']:^10} | {r['final_clusters']:^12} | {r['valid_trees']:^8}")
print("-" * 100)

print(f"\n批量实验全部完成！共处理 {len(param_combinations)} 组参数组合")
print(f"   结果保存在: {output_root}")