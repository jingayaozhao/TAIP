import os
import numpy as np
import laspy
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ========== 通用函数 ==========

def load_las_points(filename):
    """加载LAS点云文件"""
    las = laspy.read(filename)
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    header = las.header
    return pts, header


def load_axis_points(filename):
    """
    自动识别轴线文件格式（.las / .ply）
    返回: (points, header)
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".las":
        las = laspy.read(filename)
        pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        return pts, las.header

    elif ext == ".ply":
        ply_data = PlyData.read(filename)
        x = np.asarray(ply_data['vertex']['x'])
        y = np.asarray(ply_data['vertex']['y'])
        z = np.asarray(ply_data['vertex']['z'])
        pts = np.column_stack((x, y, z)).astype(np.float32)
        return pts, None

    else:
        raise ValueError(f"不支持的轴线文件格式: {ext}")


def save_points_as_las(points, ref_header, filename):
    """保存点云为LAS文件"""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = ref_header.scales
    header.offsets = ref_header.offsets
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(filename)


class SpatialIndex:
    """空间索引类，用于加速点云查询"""

    def __init__(self, points, leaf_size=32):
        """
        构建KD树空间索引

        参数:
            points: 点云数据 (N, 3)
            leaf_size: KD树叶子节点大小
        """
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.tree = cKDTree(self.points, leafsize=leaf_size)
        self._bbox = None

    @property
    def bbox(self):
        """获取点云包围盒"""
        if self._bbox is None:
            self._bbox = (self.points.min(axis=0), self.points.max(axis=0))
        return self._bbox

    def query_ball(self, center, radius):
        """查询球形范围内的点索引"""
        indices = self.tree.query_ball_point(center, radius)
        return np.array(indices, dtype=np.int32)

    def query_cylinder_fast(self, axis_point, axis_dir, radius, length_range=None):
        """
        快速查询圆柱体范围内的点
        """
        axis_dir = axis_dir / np.linalg.norm(axis_dir)

        if length_range is not None:
            t_min, t_max = length_range
            center = axis_point + axis_dir * (t_min + t_max) / 2
            half_length = (t_max - t_min) / 2
            search_radius = np.sqrt(half_length ** 2 + radius ** 2) * 1.1
        else:
            center = axis_point
            search_radius = radius * 3

        candidate_indices = self.tree.query_ball_point(center, search_radius)
        if len(candidate_indices) == 0:
            return np.empty((0, 3), dtype=np.float32)

        candidates = self.points[candidate_indices]

        v = candidates - axis_point
        proj_len = np.dot(v, axis_dir)

        if length_range is not None:
            t_min, t_max = length_range
            axial_mask = (proj_len >= t_min) & (proj_len <= t_max)
        else:
            axial_mask = np.ones(len(candidates), dtype=bool)

        if np.any(axial_mask):
            proj_vec = np.outer(proj_len[axial_mask], axis_dir)
            dist_vec = v[axial_mask] - proj_vec
            radial_dist = np.linalg.norm(dist_vec, axis=1)
            radial_mask = radial_dist < radius

            final_indices = np.where(axial_mask)[0][radial_mask]
            return candidates[final_indices]

        return np.empty((0, 3), dtype=np.float32)


def extract_cylinder_points_indexed(spatial_index, axis_point, axis_dir, radius, buffer=0.2):
    """
    使用空间索引提取圆柱体内的点
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    effective_radius = radius + buffer

    bbox_min, bbox_max = spatial_index.bbox
    bbox_half_diag = np.linalg.norm(bbox_max - bbox_min) / 2

    return spatial_index.query_cylinder_fast(
        axis_point, axis_dir, effective_radius,
        length_range=(-bbox_half_diag * 1.5, bbox_half_diag * 1.5)
    )


def extract_cylinder_segment(spatial_index, center_point, axis_dir, radius,
                             seg_length, buffer=0.2):
    """
    提取圆柱段内的点

    参数:
        spatial_index: 空间索引
        center_point: 圆柱段中心点
        axis_dir: 轴线方向（归一化）
        radius: 圆柱半径
        seg_length: 段长度
        buffer: 半径缓冲

    返回:
        段内的点云
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    effective_radius = radius + buffer
    half_len = seg_length / 2

    return spatial_index.query_cylinder_fast(
        center_point, axis_dir, effective_radius,
        length_range=(-half_len, half_len)
    )


def fit_circle_2d(points_2d):
    """
    2D圆拟合（最小二乘法）

    参数:
        points_2d: 2D点 (N, 2)

    返回:
        (center_2d, radius) 或 (None, None)
    """
    if len(points_2d) < 10:
        return None, None

    # 使用代数最小二乘法拟合圆
    # 圆方程: (x-a)^2 + (y-b)^2 = r^2
    # 展开: x^2 + y^2 = 2ax + 2by + (r^2 - a^2 - b^2)
    # 令 c = r^2 - a^2 - b^2
    # 线性形式: 2ax + 2by + c = x^2 + y^2

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2

    try:
        result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        a, b_val, c = result
        center = np.array([a, b_val])
        r_sq = c + a ** 2 + b_val ** 2

        if r_sq <= 0:
            return None, None

        radius = np.sqrt(r_sq)

        # 验证拟合质量
        distances = np.sqrt((x - a) ** 2 + (y - b_val) ** 2)
        std_dev = np.std(distances - radius)

        # 如果标准差太大，说明拟合质量差
        if std_dev > radius * 0.5:
            return None, None

        return center, radius

    except Exception:
        return None, None


def fit_segment_circle(points, axis_point, axis_dir):
    """
    对一段点云拟合圆（在垂直于轴线的平面上）

    返回:
        (center_3d, radius, quality_score) 或 (None, None, 0)
    """
    if len(points) < 20:
        return None, None, 0

    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 将点投影到垂直于轴线的平面
    v = points - axis_point
    proj_along_axis = np.outer(v @ axis_dir, axis_dir)
    v_perp = v - proj_along_axis

    # 创建局部2D坐标系
    # 找两个垂直于axis_dir的正交向量
    if abs(axis_dir[2]) < 0.9:
        u1 = np.cross(axis_dir, [0, 0, 1])
    else:
        u1 = np.cross(axis_dir, [1, 0, 0])
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(axis_dir, u1)
    u2 = u2 / np.linalg.norm(u2)

    # 投影到2D
    points_2d = np.column_stack([v_perp @ u1, v_perp @ u2])

    # 拟合圆
    center_2d, radius = fit_circle_2d(points_2d)

    if center_2d is None:
        return None, None, 0

    # 转换回3D
    center_3d = axis_point + center_2d[0] * u1 + center_2d[1] * u2

    # 计算质量分数（基于点到圆的距离分布）
    distances = np.linalg.norm(points_2d - center_2d, axis=1)
    inlier_ratio = np.mean(np.abs(distances - radius) < radius * 0.3)

    return center_3d, radius, inlier_ratio


def fit_axis_and_radius_optimized(points, axis_dir, n_slices=20):
    """
    沿轴线方向切片拟合
    """
    if len(points) < 50:
        return None, None, None

    axis_dir_norm = axis_dir / np.linalg.norm(axis_dir)
    proj_len = points @ axis_dir_norm

    proj_min, proj_max = proj_len.min(), proj_len.max()
    if proj_max - proj_min < 0.1:
        return None, None, None

    slice_edges = np.linspace(proj_min, proj_max, n_slices + 1)

    centers = []
    radii = []

    slice_indices = np.digitize(proj_len, slice_edges) - 1
    slice_indices = np.clip(slice_indices, 0, n_slices - 1)

    for i in range(n_slices):
        mask = slice_indices == i
        count = np.sum(mask)

        if count < 30:
            continue

        slice_points = points[mask]
        slice_center = slice_points.mean(axis=0)

        v = slice_points - slice_center
        proj_to_axis = np.outer(v @ axis_dir_norm, axis_dir_norm)
        v_perp = v - proj_to_axis

        dist_sq = np.sum(v_perp * v_perp, axis=1)
        r = np.sqrt(np.median(dist_sq))

        centers.append(slice_center)
        radii.append(r)

    if len(centers) < 2:
        return None, None, None

    centers = np.array(centers, dtype=np.float32)

    pca = PCA(n_components=1)
    pca.fit(centers)
    new_axis_dir = pca.components_[0].flatten()

    if np.dot(new_axis_dir, axis_dir_norm) < 0:
        new_axis_dir = -new_axis_dir

    axis_point = centers.mean(axis=0)
    mean_radius = np.median(radii) if radii else 0.1

    return axis_point, new_axis_dir, mean_radius


def refine_trunk_optimized(points, spatial_index, axis_point0, axis_dir0,
                           init_radius, max_iter=10, tol=1e-3):
    """
    迭代优化树干提取
    """
    axis_point = np.array(axis_point0, dtype=np.float32)
    axis_dir = np.array(axis_dir0, dtype=np.float32)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    radius = init_radius

    for it in range(max_iter):
        cand_pts = extract_cylinder_points_indexed(
            spatial_index, axis_point, axis_dir, radius, buffer=0.05
        )

        if len(cand_pts) < 50:
            print(f"迭代 {it}: 候选点不足({len(cand_pts)}个)，停止迭代")
            break

        new_axis_point, new_axis_dir, new_radius = fit_axis_and_radius_optimized(
            cand_pts, axis_dir
        )

        if new_axis_point is None:
            print(f"迭代 {it}: 拟合失败，停止迭代")
            break

        if np.dot(new_axis_dir, axis_dir) < 0:
            new_axis_dir = -new_axis_dir

        delta_axis = np.linalg.norm(new_axis_point - axis_point)
        delta_dir = np.linalg.norm(new_axis_dir - axis_dir)
        delta_r = abs(new_radius - radius)

        print(f"迭代 {it}: Δaxis={delta_axis:.4f}, Δdir={delta_dir:.4f}, "
              f"Δr={delta_r:.4f} 半径={new_radius:.3f}")

        axis_point, axis_dir, radius = new_axis_point, new_axis_dir, new_radius

        if delta_axis < tol and delta_dir < tol and delta_r < tol:
            print(">>> 参数收敛，停止迭代 <<<")
            break

    refined_pts = extract_cylinder_points_indexed(
        spatial_index, axis_point, axis_dir, radius, buffer=0.2
    )

    return axis_point, axis_dir, radius, refined_pts


def grow_trunk_single_direction(spatial_index, start_point, axis_dir, radius,
                                direction=1, step=0.3, max_length=50.0,
                                buffer=0.25, min_pts=30,
                                quality_threshold=0.3, consecutive_fail=3):
    """
    沿单一方向生长树干

    参数:
        spatial_index: 空间索引
        start_point: 起始点
        axis_dir: 轴线方向（归一化）
        radius: 初始半径
        direction: 生长方向 (1=沿axis_dir, -1=反方向)
        step: 步长
        max_length: 最大生长长度
        buffer: 半径缓冲
        min_pts: 最小点数阈值
        quality_threshold: 质量阈值
        consecutive_fail: 连续失败次数阈值

    返回:
        (trunk_points, end_point, end_radius)
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    grow_dir = axis_dir * direction

    trunk_segments = []
    curr_point = start_point.copy()
    curr_radius = radius
    fail_count = 0
    total_distance = 0

    # 记录半径历史用于平滑
    radius_history = [radius]

    n_steps = int(max_length / step)

    for i in range(n_steps):
        # 移动到下一个位置
        next_point = curr_point + grow_dir * step
        total_distance += step

        # 提取该段的点
        seg_pts = extract_cylinder_segment(
            spatial_index, next_point, axis_dir,
            curr_radius * 1.5, step * 2, buffer
        )

        if len(seg_pts) < min_pts:
            fail_count += 1
            if fail_count >= consecutive_fail:
                print(f"  方向{direction}: 第{i}段点数过少({len(seg_pts)}个)，连续失败{fail_count}次，停止")
                break
            continue

        # 拟合该段的圆
        center_3d, new_radius, quality = fit_segment_circle(seg_pts, next_point, axis_dir)

        if center_3d is None or quality < quality_threshold:
            fail_count += 1
            if fail_count >= consecutive_fail:
                print(f"  方向{direction}: 第{i}段拟合质量差(quality={quality:.2f})，连续失败{fail_count}次，停止")
                break
            continue

        # 检查半径变化是否合理（不应该突然变化太大）
        radius_change_ratio = new_radius / curr_radius if curr_radius > 0 else 1.0
        if radius_change_ratio > 2.0 or radius_change_ratio < 0.3:
            fail_count += 1
            if fail_count >= consecutive_fail:
                print(f"  方向{direction}: 第{i}段半径变化异常(ratio={radius_change_ratio:.2f})，停止")
                break
            continue

        # 成功，重置失败计数
        fail_count = 0

        # 更新半径（使用平滑）
        radius_history.append(new_radius)
        if len(radius_history) > 5:
            radius_history.pop(0)
        curr_radius = np.median(radius_history)

        # 更新轴线位置（使用拟合的圆心）
        # 将圆心投影回轴线上
        v = center_3d - start_point
        proj_len = np.dot(v, axis_dir)
        curr_point = start_point + proj_len * axis_dir

        # 收集这一段的点
        trunk_segments.append(seg_pts)

        # 额外检查：如果点分布异常（如树冠区域），也应停止
        if len(seg_pts) > 0:
            # 检查点的径向分布
            v = seg_pts - next_point
            proj_to_axis = np.outer(v @ axis_dir, axis_dir)
            v_perp = v - proj_to_axis
            radial_dist = np.linalg.norm(v_perp, axis=1)

            # 如果大部分点离轴线很远，可能进入了树冠区域
            if np.median(radial_dist) > curr_radius * 2:
                print(f"  方向{direction}: 第{i}段可能进入树冠区域，停止")
                break

    if trunk_segments:
        trunk_pts = np.vstack(trunk_segments)
        return trunk_pts, curr_point, curr_radius
    else:
        return np.empty((0, 3), dtype=np.float32), start_point, radius


def grow_trunk_bidirectional(spatial_index, axis_point, axis_dir, radius,
                             step=0.3, max_length_up=40.0, max_length_down=30.0,
                             buffer=0.25, min_pts=30, quality_threshold=0.3):
    """
    双向生长树干

    先基于轴线拟合出初始树干区域，然后向上和向下两个方向延伸

    参数:
        spatial_index: 空间索引
        axis_point: 轴线上的参考点
        axis_dir: 轴线方向（应该大致指向上方）
        radius: 初始树干半径
        step: 生长步长
        max_length_up: 向上最大生长长度
        max_length_down: 向下最大生长长度
        buffer: 半径缓冲
        min_pts: 最小点数阈值
        quality_threshold: 质量阈值

    返回:
        所有树干点
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 确保axis_dir大致指向上方（Z轴正方向）
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

    all_trunk_pts = []

    # 先提取轴线附近的初始树干点
    init_pts = extract_cylinder_segment(
        spatial_index, axis_point, axis_dir, radius, step * 4, buffer
    )
    if len(init_pts) > 0:
        all_trunk_pts.append(init_pts)

    print(f"  初始树干点数: {len(init_pts)}")

    # 向上生长（沿axis_dir正方向）
    print("  向上生长...")
    up_pts, up_end, up_radius = grow_trunk_single_direction(
        spatial_index, axis_point, axis_dir, radius,
        direction=1, step=step, max_length=max_length_up,
        buffer=buffer, min_pts=min_pts, quality_threshold=quality_threshold
    )
    if len(up_pts) > 0:
        all_trunk_pts.append(up_pts)
        print(f"  向上生长获得 {len(up_pts)} 个点")

    # 向下生长（沿axis_dir负方向）
    print("  向下生长...")
    down_pts, down_end, down_radius = grow_trunk_single_direction(
        spatial_index, axis_point, axis_dir, radius,
        direction=-1, step=step, max_length=max_length_down,
        buffer=buffer, min_pts=min_pts, quality_threshold=quality_threshold
    )
    if len(down_pts) > 0:
        all_trunk_pts.append(down_pts)
        print(f"  向下生长获得 {len(down_pts)} 个点")

    if all_trunk_pts:
        return np.vstack(all_trunk_pts)
    else:
        return np.empty((0, 3), dtype=np.float32)


def estimate_tree_height_range(spatial_index, axis_point, axis_dir, radius, buffer=1.0):
    """
    估计树的高度范围

    在轴线附近搜索点云，估计树的大致高度范围
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 确保方向向上
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

    # 搜索一个较大的圆柱范围
    search_radius = radius * 3
    bbox_min, bbox_max = spatial_index.bbox
    max_search_length = bbox_max[2] - bbox_min[2]  # 使用Z轴范围作为参考

    pts = spatial_index.query_cylinder_fast(
        axis_point, axis_dir, search_radius,
        length_range=(-max_search_length, max_search_length)
    )

    if len(pts) < 10:
        return None, None

    # 计算沿轴线的投影
    v = pts - axis_point
    proj_len = v @ axis_dir

    return proj_len.min(), proj_len.max()


def prefilter_region_extended(points, axis_points, axis_dir, margin_horizontal=5.0,
                              margin_vertical=50.0):
    """
    根据轴线点预过滤点云区域（扩展版）

    在轴线方向上使用更大的扩展范围，以便能够延伸提取完整树干

    参数:
        points: 完整点云
        axis_points: 轴线点
        axis_dir: 轴线方向
        margin_horizontal: 水平方向边界扩展
        margin_vertical: 垂直方向边界扩展

    返回:
        过滤后的点云
    """
    if len(axis_points) == 0:
        return points

    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 确保方向向上
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

    # 计算轴线点的中心
    axis_center = axis_points.mean(axis=0)

    # 计算水平方向的范围（使用更小的margin）
    # 找到两个垂直于axis_dir的方向
    if abs(axis_dir[2]) < 0.9:
        h1 = np.cross(axis_dir, [0, 0, 1])
    else:
        h1 = np.cross(axis_dir, [1, 0, 0])
    h1 = h1 / np.linalg.norm(h1)
    h2 = np.cross(axis_dir, h1)
    h2 = h2 / np.linalg.norm(h2)

    # 计算所有点相对于轴线中心的位置
    v = points - axis_center

    # 沿轴线方向的投影
    proj_axial = v @ axis_dir

    # 水平方向的投影
    proj_h1 = v @ h1
    proj_h2 = v @ h2
    proj_horizontal = np.sqrt(proj_h1 ** 2 + proj_h2 ** 2)

    # 计算轴线点的范围
    axis_v = axis_points - axis_center
    axis_proj = axis_v @ axis_dir
    axis_min, axis_max = axis_proj.min(), axis_proj.max()

    # 扩展轴线方向的范围
    axial_min = axis_min - margin_vertical
    axial_max = axis_max + margin_vertical

    # 筛选条件
    mask = (proj_axial >= axial_min) & (proj_axial <= axial_max) & \
           (proj_horizontal <= margin_horizontal)

    return points[mask]


# ========== 主程序 ==========

def process_single_tree(forest_pts, spatial_index, axis_file, axes_folder,
                        output_folder, forest_header):
    """处理单棵树的树干提取（增强版）"""
    fname = os.path.basename(axis_file)
    full_path = os.path.join(axes_folder, fname)

    try:
        axis_points, _ = load_axis_points(full_path)
    except Exception as e:
        print(f"{fname}: 读取失败 -> {e}")
        return False

    if len(axis_points) < 3:
        print(f"{fname}: 轴线点不足（需≥3个点），跳过")
        return False

    # 初始轴线参数
    axis_point0 = axis_points.mean(axis=0)

    pca = PCA(n_components=1)
    pca.fit(axis_points)
    axis_dir0 = pca.components_[0].flatten()
    axis_dir0 = axis_dir0 / np.linalg.norm(axis_dir0)

    # 确保轴线方向大致向上
    if axis_dir0[2] < 0:
        axis_dir0 = -axis_dir0

    init_radius = 0.3

    print(f"\n{'=' * 50}")
    print(f"开始处理 {fname}")
    print(f"轴线点数: {len(axis_points)}")
    print(f"轴线高度范围: {axis_points[:, 2].min():.2f} - {axis_points[:, 2].max():.2f}")

    # 预过滤点云区域 - 使用扩展的垂直范围
    local_pts = prefilter_region_extended(
        forest_pts, axis_points, axis_dir0,
        margin_horizontal=5.0, margin_vertical=50.0
    )

    if len(local_pts) < 100:
        print(f"{fname}: 局部点云过少({len(local_pts)}个点)，跳过")
        return False

    print(f"局部点云点数: {len(local_pts)}")
    print(f"局部点云高度范围: {local_pts[:, 2].min():.2f} - {local_pts[:, 2].max():.2f}")

    # 为局部点云创建空间索引
    local_index = SpatialIndex(local_pts, leaf_size=16)

    # 迭代优化初始参数
    axis_point, axis_dir, radius, trunk_pts = refine_trunk_optimized(
        local_pts, local_index, axis_point0, axis_dir0, init_radius, max_iter=10
    )

    if axis_point is None:
        print(f"{fname}: 初始拟合失败")
        return False

    print(f"初始拟合完成: 半径={radius:.3f}")

    # 估计树的高度范围
    height_min, height_max = estimate_tree_height_range(
        local_index, axis_point, axis_dir, radius
    )
    if height_min is not None:
        print(f"估计树高范围: {height_min:.2f} - {height_max:.2f} (相对于轴线点)")

    # 双向生长树干
    trunk_full = grow_trunk_bidirectional(
        local_index,
        axis_point,
        axis_dir,
        radius,
        step=0.3,
        max_length_up=50.0,  # 向上最大延伸50米
        max_length_down=30.0,  # 向下最大延伸30米
        buffer=0.3,
        min_pts=30,
        quality_threshold=0.25
    )

    # 合并初始树干点
    if len(trunk_pts) > 0 and len(trunk_full) > 0:
        trunk_full = np.vstack([trunk_pts, trunk_full])
    elif len(trunk_pts) > 0:
        trunk_full = trunk_pts

    # 去重
    if len(trunk_full) > 0:
        trunk_full = np.unique(trunk_full, axis=0)

        # 计算提取的树干高度范围
        trunk_z_min, trunk_z_max = trunk_full[:, 2].min(), trunk_full[:, 2].max()
        trunk_height = trunk_z_max - trunk_z_min
        print(f"提取树干高度: {trunk_z_min:.2f} - {trunk_z_max:.2f} (高度差: {trunk_height:.2f}m)")

    # 保存结果
    out_name = os.path.join(output_folder, f"trunk_{os.path.splitext(fname)[0]}.las")

    if len(trunk_full) > 0:
        save_points_as_las(trunk_full, forest_header, out_name)
        print(f"{fname} 提取树干点数: {len(trunk_full)} → 保存至 {out_name}")
        return True
    else:
        print(f"{fname} 未提取到有效树干点")
        return False


def main():
    """批量处理所有HDBSCAN参数组合的轴线文件夹"""
    import time

    start_time = time.time()

    # 1. 读取森林点云
    print("正在加载森林点云...")
    load_start = time.time()
    forest_pts, forest_header = load_las_points(
        r"E:\dataset\Extended version of WYTHAM and LAUTx\p12\p12_1_non_ground.las"
    )
    print(f"点云加载完成，共 {len(forest_pts):,} 个点，耗时 {time.time() - load_start:.2f}秒")

    # 2. 构建全局空间索引
    print("正在构建空间索引...")
    index_start = time.time()
    global_index = SpatialIndex(forest_pts, leaf_size=32)
    print(f"空间索引构建完成，耗时 {time.time() - index_start:.2f}秒")

    # 3. 设置根路径
    root_axes_folder = r" "
    root_output_folder = r" "
    os.makedirs(root_output_folder, exist_ok=True)

    # 4. 遍历所有组合子文件夹
    subfolders = [
        f for f in os.listdir(root_axes_folder)
        if os.path.isdir(os.path.join(root_axes_folder, f))
    ]

    print(f"\n找到 {len(subfolders)} 个组合文件夹：")
    for f in subfolders:
        print(" -", f)

    total_success = 0
    total_files = 0
    process_start = time.time()

    for combo_folder in subfolders:
        axes_folder = os.path.join(root_axes_folder, combo_folder)
        output_folder = os.path.join(root_output_folder, combo_folder + "_trunks")
        os.makedirs(output_folder, exist_ok=True)

        axis_files = [
            f for f in os.listdir(axes_folder)
            if f.lower().endswith((".las", ".ply"))
        ]

        print(f"\n{'#' * 60}")
        print(f"处理组合: {combo_folder}, 轴线文件数: {len(axis_files)}")
        print(f"{'#' * 60}")

        success_count = 0

        for i, fname in enumerate(axis_files):
            print(f"\n[{i + 1}/{len(axis_files)}] ", end="")
            if process_single_tree(forest_pts, global_index, fname, axes_folder,
                                   output_folder, forest_header):
                success_count += 1

        print(f"\n组合 {combo_folder} 完成: 成功 {success_count}/{len(axis_files)}")
        total_success += success_count
        total_files += len(axis_files)

    total_time = time.time() - start_time
    process_time = time.time() - process_start

    print("\n" + "=" * 60)
    print("全部组合处理完成！")
    print(f"  总文件数: {total_files}")
    print(f"  成功处理: {total_success}")
    print(f"  失败: {total_files - total_success}")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  处理耗时: {process_time:.2f}秒")
    if total_files > 0:
        print(f"  平均每棵树: {process_time / total_files:.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    main()