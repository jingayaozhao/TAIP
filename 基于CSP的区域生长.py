import numpy as np
import laspy
import os
from scipy.spatial import KDTree
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime


class TreeSegmenter:
    def __init__(self, las_file_path, trunks_dir, output_dir="tree_las_files", voxel_size=0.05):
      
        self.output_dir = output_dir
        self.trees_dir = os.path.join(self.output_dir, "individual_trees")
        os.makedirs(self.trees_dir, exist_ok=True)

        # 进度追踪文件
        self.progress_file = os.path.join(self.output_dir, "segmentation_progress.json")
        self.distance_cache_dir = os.path.join(self.output_dir, "distance_cache")
        os.makedirs(self.distance_cache_dir, exist_ok=True)

        # 核心参数（保持原有优化配置）
        self.voxel_size = voxel_size
        self.graph_radius = 0.2
        self.trunk_height = 1.3
        self.slice_half_thickness = 0.05
        self.z_min = self.trunk_height - self.slice_half_thickness
        self.z_max = self.trunk_height + self.slice_half_thickness
        self.tolerance = 1e-2
        self.max_hops = 2
        self.hop_penalty = 1.2

        # 竞争机制参数
        self.max_normalized_distance = 10.0  # DBH标准化距离阈值

        # 加载原始点云
        self._load_original_point_cloud(las_file_path)

        # 降采样
        self._voxel_downsample()
        print(
            f"降采样完成：原始{self.n_original_points}点 → 降采样后{self.n_points}点（保留{self.n_points / self.n_original_points:.1%}）")

        # 加载树干
        self._load_and_slice_trunks(trunks_dir)

        # 结果存储
        self.dbh = {}
        self.progress_data = self._load_progress()
        self.completed_trees = set(self.progress_data.get('completed_trees', []))

        # 全局分配数组（用于追踪已分配的点）
        self.global_assignments = np.full(self.n_original_points, -1, dtype=int)

    def _load_progress(self):
        """加载处理进度"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            print(f"\n检测到历史进度：已完成{len(progress.get('completed_trees', []))}棵树")
            return progress
        return {'completed_trees': [], 'dbh': {}}

    def _save_progress(self, tree_id):
        """实时保存处理进度"""
        self.completed_trees.add(tree_id)
        self.progress_data['completed_trees'] = list(self.completed_trees)
        self.progress_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.progress_data['total_trees'] = len(self.trunk_clusters)
        self.progress_data['dbh'] = {str(k): v for k, v in self.dbh.items()}

        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)

    def _get_distance_cache_path(self, tree_id):
        """获取距离场缓存文件路径"""
        return os.path.join(self.distance_cache_dir, f"tree_{tree_id}_distances.npy")

    def _load_cached_distance(self, tree_id):
        """加载缓存的距离场"""
        cache_path = self._get_distance_cache_path(tree_id)
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None

    def _save_distance_cache(self, tree_id, distances):
        """保存距离场到缓存"""
        cache_path = self._get_distance_cache_path(tree_id)
        np.save(cache_path, distances)

    def _load_original_point_cloud(self, las_file_path):
        """加载原始点云"""
        self.original_las = laspy.read(las_file_path)
        self.original_pc = np.vstack((self.original_las.x, self.original_las.y, self.original_las.z)).T
        self.n_original_points = self.original_pc.shape[0]
        self.original_header = self.original_las.header
        self.point_format = self.original_las.point_format

    def _voxel_downsample(self):
        """体素降采样"""
        voxel_indices = np.floor(self.original_pc / self.voxel_size).astype(int)
        voxel_groups = {}
        for idx, (point, voxel_idx) in enumerate(zip(self.original_pc, voxel_indices)):
            key = tuple(voxel_idx)
            if key not in voxel_groups:
                voxel_groups[key] = []
            voxel_groups[key].append((point, idx))

        self.downsampled_pc = []
        self.downsample_to_original = []
        self.original_to_downsample = np.full(self.n_original_points, -1, dtype=int)

        for group in voxel_groups.values():
            points = np.array([p[0] for p in group])
            center = np.mean(points, axis=0)
            self.downsampled_pc.append(center)

            original_indices = [p[1] for p in group]
            downsample_idx = len(self.downsampled_pc) - 1
            self.downsample_to_original.append(original_indices)
            for orig_idx in original_indices:
                self.original_to_downsample[orig_idx] = downsample_idx

        self.downsampled_pc = np.array(self.downsampled_pc)
        self.n_points = self.downsampled_pc.shape[0]
        self.downsampled_kdtree = KDTree(self.downsampled_pc)

    def _load_and_slice_trunks(self, trunks_dir):
        """加载树干切片和完整树干点"""
        trunk_files = [f for f in os.listdir(trunks_dir) if f.endswith(('.las', '.laz'))]
        if not trunk_files:
            raise ValueError(f"树干文件夹 {trunks_dir} 中未找到LAS文件")

        original_kdtree = KDTree(self.original_pc)
        self.trunk_clusters = {}
        self.full_trunk_clusters = {}

        for tree_id, filename in enumerate(trunk_files):
            file_path = os.path.join(trunks_dir, filename)
            trunk_las = laspy.read(file_path)
            full_trunk_pc = np.vstack((trunk_las.x, trunk_las.y, trunk_las.z)).T

            # 胸高切片点
            slice_mask = (full_trunk_pc[:, 2] >= self.z_min) & (full_trunk_pc[:, 2] <= self.z_max)
            trunk_slice_pc = full_trunk_pc[slice_mask]
            if len(trunk_slice_pc) < 3:
                print(f"警告：树干 {filename} 胸高切片点不足（{len(trunk_slice_pc)}个）")
                continue

            # 完整树干点
            original_full_indices = []
            for point in full_trunk_pc:
                matches = original_kdtree.query_ball_point(point, r=self.tolerance)
                original_full_indices.extend(matches)
            original_full_indices = np.unique(original_full_indices)

            # 胸高切片点
            original_slice_indices = []
            for point in trunk_slice_pc:
                matches = original_kdtree.query_ball_point(point, r=self.tolerance)
                original_slice_indices.extend(matches)
            original_slice_indices = np.unique(original_slice_indices)

            # 映射到降采样点云
            downsample_slice_indices = np.unique(self.original_to_downsample[original_slice_indices])
            downsample_slice_indices = downsample_slice_indices[downsample_slice_indices != -1]
            downsample_full_indices = np.unique(self.original_to_downsample[original_full_indices])
            downsample_full_indices = downsample_full_indices[downsample_full_indices != -1]

            if len(downsample_slice_indices) > 0 and len(downsample_full_indices) > 5:
                self.trunk_clusters[tree_id] = downsample_slice_indices
                self.full_trunk_clusters[tree_id] = downsample_full_indices
                print(
                    f"加载树干 {tree_id}：{filename}（胸高{len(downsample_slice_indices)}个，完整{len(downsample_full_indices)}个）")
            else:
                print(f"警告：树干 {filename} 完整树干点不足（{len(downsample_full_indices)}个）")

        if not self.trunk_clusters:
            raise RuntimeError("未加载任何有效树干切片点")
        print(f"\n总计加载有效树干：{len(self.trunk_clusters)}棵")

    def calculate_dbh(self):
        """计算胸径"""
        # 如果已有缓存的DBH，优先使用
        if 'dbh' in self.progress_data and self.progress_data['dbh']:
            for tree_id_str, dbh_val in self.progress_data['dbh'].items():
                tree_id = int(tree_id_str)
                if tree_id in self.trunk_clusters:
                    self.dbh[tree_id] = dbh_val
            print("从缓存加载胸径数据")

        # 计算缺失的DBH
        for tree_id, slice_indices in self.trunk_clusters.items():
            if tree_id not in self.dbh:
                slice_points = self.downsampled_pc[slice_indices][:, :2]
                centroid = np.mean(slice_points, axis=0)
                distances = np.linalg.norm(slice_points - centroid, axis=1)
                radius = np.mean(distances)
                self.dbh[tree_id] = radius * 2 if radius > 0.01 else 0.0

        print("胸径计算完成:", {f"树{tid}": f"{d:.2f}m" for tid, d in self.dbh.items()})

    def _custom_multi_source_dijkstra(self, source_indices):
        """带路径跳跃的Dijkstra算法"""
        distances = np.full(self.n_points, np.inf, dtype=np.float32)
        heap = []

        for idx in source_indices:
            if 0 <= idx < self.n_points:
                distances[idx] = 0.0
                heapq.heappush(heap, (0.0, idx, 0))

        while heap:
            current_dist, u, hops = heapq.heappop(heap)
            if current_dist > distances[u]:
                continue

            # 一级邻域
            neighbors = self.downsampled_kdtree.query_ball_point(self.downsampled_pc[u], r=self.graph_radius)
            for v in neighbors:
                if u == v:
                    continue
                edge_weight = np.linalg.norm(self.downsampled_pc[u] - self.downsampled_pc[v])
                new_dist = current_dist + edge_weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(heap, (new_dist, v, hops))

            # 二级邻域（跳跃）
            if hops < self.max_hops:
                far_neighbors = self.downsampled_kdtree.query_ball_point(
                    self.downsampled_pc[u], r=self.graph_radius * 2
                )
                for v in far_neighbors:
                    if u == v or v in neighbors:
                        continue
                    edge_weight = np.linalg.norm(self.downsampled_pc[u] - self.downsampled_pc[v])
                    new_dist = current_dist + edge_weight * self.hop_penalty
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(heap, (new_dist, v, hops + 1))

        return distances

    def _compute_single_distance_field(self, tree_id):
        """
        计算单棵树的距离场（支持缓存）
        返回: normalized_distances
        """
        # 尝试加载缓存
        cached_distances = self._load_cached_distance(tree_id)
        if cached_distances is not None:
            print(f"    从缓存加载树{tree_id}的距离场")
            return cached_distances

        # 计算距离场
        source_indices = self.full_trunk_clusters[tree_id]
        raw_distances = self._custom_multi_source_dijkstra(source_indices)

        # DBH标准化
        dbh = self.dbh[tree_id] if self.dbh[tree_id] > 0 else 1e-6
        normalized_distances = raw_distances / (dbh ** (2 / 3))

        # 保存到缓存
        self._save_distance_cache(tree_id, normalized_distances)

        return normalized_distances

    def _save_single_tree(self, tree_id, point_mask):
        """保存单棵树的点云（原始点云版本）"""
        if np.sum(point_mask) == 0:
            print(f"    警告：树{tree_id}无匹配点，跳过保存")
            return False

        try:
            # 创建新LAS对象
            try:
                new_header = laspy.LasHeader(point_format=self.point_format, version="1.4")
                new_las = laspy.LasData(new_header)
            except:
                new_las = laspy.create(point_format=self.point_format.id)

            # 复制坐标
            new_las.x = self.original_las.x[point_mask]
            new_las.y = self.original_las.y[point_mask]
            new_las.z = self.original_las.z[point_mask]

            # 复制其他属性
            for dim in self.point_format.dimensions:
                if dim.name not in ['x', 'y', 'z']:
                    try:
                        setattr(new_las, dim.name, getattr(self.original_las, dim.name)[point_mask])
                    except:
                        pass

            # 保存文件
            filename = f"tree_{tree_id}_dbh_{self.dbh[tree_id]:.2f}m.las"
            filepath = os.path.join(self.trees_dir, filename)
            new_las.write(filepath)
            print(f"    已保存: {filename} (点数: {np.sum(point_mask)})")
            return True

        except Exception as e:
            print(f"    保存树{tree_id}失败: {str(e)}")
            return False

    def process_and_save_trees_realtime(self):
        
        print(f"\n开始实时竞争式树木分割（共{len(self.trunk_clusters)}棵树）...")
        print(f"{'=' * 60}\n")

        # 获取待处理的树（排除已完成的）
        pending_trees = [tid for tid in self.trunk_clusters.keys() if tid not in self.completed_trees]

        if not pending_trees:
            print("所有树木已完成处理！")
            return

        print(f"待处理树木：{len(pending_trees)}棵")
        print(f"已完成树木：{len(self.completed_trees)}棵\n")

        # 逐树处理
        for idx, tree_id in enumerate(pending_trees, 1):
            print(f"\n{'=' * 60}")
            print(f"处理进度：[{idx}/{len(pending_trees)}] 树{tree_id}（DBH: {self.dbh[tree_id]:.2f}m）")
            print(f"{'=' * 60}")

            try:
                # 步骤1：计算当前树的距离场
                print(f"  步骤1/4：计算树{tree_id}的距离场...")
                current_tree_distances = self._compute_single_distance_field(tree_id)
                valid_count = np.sum(current_tree_distances < self.max_normalized_distance)
                print(f"    完成（有效范围点数: {valid_count}）")

                # 步骤2：计算所有已完成树的距离场（用于竞争）
                print(f"  步骤2/4：加载已完成树的距离场（共{len(self.completed_trees)}棵）...")
                competitor_distances = {}
                for completed_id in self.completed_trees:
                    competitor_distances[completed_id] = self._load_cached_distance(completed_id)
                    if competitor_distances[completed_id] is None:
                        print(f"    警告：树{completed_id}的缓存缺失，将重新计算")
                        competitor_distances[completed_id] = self._compute_single_distance_field(completed_id)

                # 步骤3：竞争分配（当前树 vs 所有已完成树）
                print(f"  步骤3/4：竞争分配点云...")
                downsample_mask = np.zeros(self.n_points, dtype=bool)

                for point_idx in range(self.n_points):
                    current_dist = current_tree_distances[point_idx]

                    # 只考虑有效范围内的点
                    if current_dist >= self.max_normalized_distance:
                        continue

                    # 与所有已完成树竞争
                    is_winner = True
                    for comp_id, comp_distances in competitor_distances.items():
                        if comp_distances[point_idx] < current_dist:
                            is_winner = False
                            break

                    if is_winner:
                        downsample_mask[point_idx] = True

                assigned_downsample_count = np.sum(downsample_mask)
                print(f"    当前树赢得{assigned_downsample_count}个降采样点")

                # 步骤4：映射到原始点云并立即保存
                print(f"  步骤4/4：映射到原始点云并保存...")
                original_mask = np.zeros(self.n_original_points, dtype=bool)
                downsample_indices = np.where(downsample_mask)[0]

                for down_idx in downsample_indices:
                    orig_indices = self.downsample_to_original[down_idx]
                    original_mask[orig_indices] = True
                    self.global_assignments[orig_indices] = tree_id

                assigned_original_count = np.sum(original_mask)
                print(f"    降采样{assigned_downsample_count}个 → 原始{assigned_original_count}个")

                # 立即保存
                if self._save_single_tree(tree_id, original_mask):
                    self._save_progress(tree_id)
                    print(f"     树{tree_id}处理完成并保存")
                else:
                    print(f"     树{tree_id}保存失败")

                # 释放内存
                del current_tree_distances
                del competitor_distances

            except Exception as e:
                print(f"     树{tree_id}处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 保存未分配点
        print(f"\n{'=' * 60}")
        print("保存未分配点...")
        print(f"{'=' * 60}")
        self._save_unassigned_points(self.global_assignments)

        print(f"\n{'=' * 60}")
        print(f"全部树木处理完成！")
        print(f"结果保存至: {os.path.abspath(self.trees_dir)}")
        print(f"{'=' * 60}")

    def _save_unassigned_points(self, global_assignments):
        """保存未分配的点"""
        unassigned_mask = (global_assignments == -1)
        unassigned_count = np.sum(unassigned_mask)

        if unassigned_count > 0:
            print(f"未分配点: {unassigned_count}个...")
            try:
                try:
                    new_header = laspy.LasHeader(point_format=self.point_format, version="1.4")
                    unassigned_las = laspy.LasData(new_header)
                except:
                    unassigned_las = laspy.create(point_format=self.point_format.id)

                unassigned_las.x = self.original_las.x[unassigned_mask]
                unassigned_las.y = self.original_las.y[unassigned_mask]
                unassigned_las.z = self.original_las.z[unassigned_mask]

                filepath = os.path.join(self.trees_dir, "unassigned_points.las")
                unassigned_las.write(filepath)
                print(f" 未分配点已保存: unassigned_points.las")
            except Exception as e:
                print(f" 未分配点保存失败: {str(e)}")

    def visualize_saved_trees(self, sample_ratio=0.2):
       
        print("\n生成可视化图表...")

        # 读取已保存的树文件
        tree_files = [f for f in os.listdir(self.trees_dir) if f.startswith('tree_') and f.endswith('.las')]

        if not tree_files:
            print("未找到已保存的树文件")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab20(np.linspace(0, 1, len(tree_files)))

        for i, filename in enumerate(tree_files):
            filepath = os.path.join(self.trees_dir, filename)
            tree_las = laspy.read(filepath)
            tree_pc = np.vstack((tree_las.x, tree_las.y, tree_las.z)).T

            # 采样
            if len(tree_pc) > 10000:
                sample_idx = np.random.choice(len(tree_pc), int(len(tree_pc) * sample_ratio), replace=False)
                tree_pc = tree_pc[sample_idx]

            # 提取树ID和DBH
            tree_id = int(filename.split('_')[1])
            dbh_str = filename.split('dbh_')[1].replace('m.las', '')

            ax.scatter(
                tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2],
                c=[colors[i]], label=f'树{tree_id} (DBH: {dbh_str}m)',
                s=2, alpha=0.7
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'树木分割结果（实时保存版本，共{len(tree_files)}棵树）')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()

    def run_pipeline(self):
        """主流程（实时保存版本）"""
        print("=" * 60)
        print("开始单木分割流程（实时保存版本）")
        print("=" * 60)

        # 1. 计算胸径
        self.calculate_dbh()

        # 2. 实时竞争分配并保存（核心改动）
        self.process_and_save_trees_realtime()

        # 3. 可视化（可选）
        try:
            self.visualize_saved_trees()
        except Exception as e:
            print(f"可视化失败（可忽略）: {str(e)}")

        print("\n流程全部完成！")

# 示例用法
if __name__ == "__main__":
    las_file_path = r" "
    trunks_dir = r" "

    segmenter = TreeSegmenter(
        las_file_path=las_file_path,
        trunks_dir=trunks_dir,
        output_dir=r" ",
        voxel_size=0.05
    )

    segmenter.run_pipeline()