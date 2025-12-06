import numpy as np
import open3d as o3d
from scipy.ndimage import minimum_filter, maximum_filter
import time
class MDetector:
    """
    M-detector Python 实现 (Numpy 接口优化版)
    输入: Numpy Array (N, 4) [x, y, z, intensity]
    输出: Numpy Array (Bg, 4), (Fg, 4)
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        
        # --- 这里的参数请根据雷达型号修改 ---
        # 例如 RS-16: 水平0.2, 垂直2.0
        self.hor_res = np.deg2rad(config.get('hor_resolution_deg', 0.2))
        self.ver_res = np.deg2rad(config.get('ver_resolution_deg', 2.0))
        self.fov_up = np.deg2rad(config.get('fov_up', 15.0))
        self.fov_down = np.deg2rad(config.get('fov_down', -15.0))

        self.depth_map_width = int(np.ceil(2 * np.pi / self.hor_res))
        self.depth_map_height = int(np.ceil((self.fov_up - self.fov_down) / self.ver_res))
        
        self.history = []
        self.max_history_length = config.get('max_depth_map_num', 5)
        self.timings = {}
        self.frame_count = 0 

        # 阈值参数
        self.k_depth = config.get('k_depth', 0.005) 
        
        # Case 1 (Entering)
        self.enter_min_thr1 = config.get('enter_min_thr1', 1.0)
        self.map_cons_thr1 = config.get('map_cons_depth_thr1', 0.3)
        self.occ_map_thr1 = config.get('occluded_map_thr1', 2)
        
        # Case 2 (Moving Away)
        self.occ_depth_thr2 = config.get('occ_depth_thr2', 0.15)
        self.map_cons_thr2 = config.get('map_cons_depth_thr2', 0.2)
        self.occ_times_thr2 = config.get('occluded_times_thr2', 2)
        self.v_min_thr2 = config.get('v_min_thr2', 0.5)
        
        # Case 3 (Occluding)
        self.occ_depth_thr3 = config.get('occ_depth_thr3', 0.15)
        self.map_cons_thr3 = config.get('map_cons_depth_thr3', 0.2)
        self.occ_times_thr3 = config.get('occluding_times_thr3', 2)
        self.v_min_thr3 = config.get('v_min_thr3', 0.5)

        self.search_radius = 1 

    def _project_to_spherical(self, points):
        """投影：将点云投影到球坐标深度图"""
        depth = np.linalg.norm(points, axis=1)
        depth = np.maximum(depth, 1e-6) # 避免除0
        
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        elevation = np.arcsin(np.clip(points[:, 2] / depth, -1.0, 1.0))
        
        u = ((np.pi - azimuth) / self.hor_res).astype(int)
        v = ((self.fov_up - elevation) / self.ver_res).astype(int)
        
        u = np.clip(u, 0, self.depth_map_width - 1)
        v = np.clip(v, 0, self.depth_map_height - 1)
        return u, v, depth

    def _create_history_frame(self, u, v, depth, timestamp):
        min_depth_map = np.full((self.depth_map_height, self.depth_map_width), np.inf, dtype=np.float32)
        max_depth_map = np.full((self.depth_map_height, self.depth_map_width), -np.inf, dtype=np.float32)
        
        # 使用 numpy 高效更新
        np.minimum.at(min_depth_map, (v, u), depth)
        np.maximum.at(max_depth_map, (v, u), depth)
        
        max_depth_map[max_depth_map == -np.inf] = 0
        
        return {
            "min_depth": min_depth_map, 
            "max_depth": max_depth_map,
            "timestamp": timestamp
        }

    def _get_adaptive_threshold(self, depth, base_threshold):
        return self.k_depth * depth + base_threshold

    def _check_neighborhood_consistency(self, u, v, depth, target_map, threshold_map):
        """Scipy 加速的邻域检查"""
        kernel_size = 2 * self.search_radius + 1
        map_min = minimum_filter(target_map, size=kernel_size, mode='nearest')
        map_max = maximum_filter(target_map, size=kernel_size, mode='nearest')
        
        neigh_min = map_min[v, u]
        neigh_max = map_max[v, u]
        
        is_consistent = (depth > (neigh_min - threshold_map)) & \
                        (depth < (neigh_max + threshold_map))
        return is_consistent

    def _find_dynamic_points_case1(self, u, v, depth):
        # Case 1: Entering
        occlusion_counts = np.zeros(len(depth), dtype=np.uint8)
        enter_thr = self._get_adaptive_threshold(depth, self.enter_min_thr1)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr1)

        for frame in self.history:
            hist_min = frame["min_depth"][v, u]
            is_closer = depth < (hist_min - enter_thr)
            is_consistent = self._check_neighborhood_consistency(u, v, depth, frame["min_depth"], cons_thr)
            occlusion_counts += (is_closer & (~is_consistent))
            
        return np.where(occlusion_counts >= self.occ_map_thr1)[0]

    def _find_dynamic_points_case2(self, u, v, depth, current_time):
        # Case 2: Moving Away
        if len(self.history) < 2: return np.array([], dtype=np.int64)
        
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr2)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr2)
        
        last_frame = self.history[-1]
        hist_max_last = last_frame["max_depth"][v, u]
        
        delta_t = current_time - last_frame["timestamp"]
        vel_limit = self.v_min_thr2 * delta_t
        final_thr = np.minimum(base_thr, vel_limit + 1e-3)
        
        is_further = (depth > (hist_max_last + final_thr)) & (hist_max_last > 0.1)
        
        recursive_pass = np.zeros(len(depth), dtype=np.uint8)
        for prev_frame in self.history[:-1]:
            hist_max_prev = prev_frame["max_depth"][v, u]
            dt_hist = last_frame["timestamp"] - prev_frame["timestamp"]
            if dt_hist <= 0: continue
            
            v_lim_hist = self.v_min_thr2 * dt_hist
            f_thr_hist = np.minimum(base_thr, v_lim_hist + 1e-3)
            
            is_prev_moving = (hist_max_last > (hist_max_prev + f_thr_hist)) & (hist_max_prev > 0.1)
            recursive_pass += is_prev_moving

        is_consistent_any = np.zeros(len(depth), dtype=bool)
        for frame in self.history:
             is_consistent_any |= self._check_neighborhood_consistency(
                 u, v, depth, frame["max_depth"], cons_thr
             )

        final_mask = is_further & (recursive_pass >= 1) & (~is_consistent_any)
        return np.where(final_mask)[0]

    def _find_dynamic_points_case3(self, u, v, depth, current_time):
        # Case 3: Occluding
        occlusion_counts = np.zeros(len(depth), dtype=np.uint8)
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr3)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr3)

        for frame in self.history:
            hist_min = frame["min_depth"][v, u]
            delta_t = current_time - frame["timestamp"]
            vel_limit = self.v_min_thr3 * delta_t
            final_thr = np.minimum(base_thr, vel_limit + 1e-3)
            
            is_occluding = depth < (hist_min - final_thr)
            is_consistent = self._check_neighborhood_consistency(u, v, depth, frame["min_depth"], cons_thr)
            occlusion_counts += (is_occluding & (~is_consistent))

        return np.where(occlusion_counts >= self.occ_times_thr3)[0]

    def process_frame(self, points_np, timestamp=None):
        """
        核心接口：Numpy 进，Numpy 出
        :param points_np: (N, 3) XYZ 或 (N, 4) XYZI 的 Numpy 数组
        :param timestamp: (Optional) 这一帧的时间戳，如果不传则自动递增
        :return: (background_points, foreground_points)
        """
        
        # 1. 预处理：移除盲区 (0.5m 以内)
        # 只取前3列算距离，但保留后面所有列 (intensity) 用于输出
        t0 = time.time()
        dists = np.linalg.norm(points_np[:, :3], axis=1)
        valid_mask = dists > self.config.get('blind_dis', 0.5)
        points = points_np[valid_mask] # 此时 points 可能是 (M, 4)
        t1 = time.time()
        if len(points) == 0:
            return np.empty((0, points_np.shape[1])), np.empty((0, points_np.shape[1]))
            
        # 2. 投影 (只取 XYZ 进行投影计算)
        u, v, depth = self._project_to_spherical(points[:, :3])
        t2 = time.time()
        # 时间戳处理
        if timestamp is None:
            current_time = self.frame_count * 0.1 # 模拟 10Hz
        else:
            current_time = timestamp
        
        point_out_indices = np.array([], dtype=int)
        
        # 3. 动态检测逻辑
        if self.history:
            tt1 = time.time()
            idx_c1 = self._find_dynamic_points_case1(u, v, depth)
            tt2 = time.time()
            idx_c2 = self._find_dynamic_points_case2(u, v, depth, current_time)
            tt3 = time.time()
            idx_c3 = self._find_dynamic_points_case3(u, v, depth, current_time)
            tt4 = time.time()
            point_out_indices = np.unique(np.concatenate([idx_c1, idx_c2, idx_c3]))
            tt5 = time.time()
            # 4. 聚类去噪 (使用 Open3D 加速)
            # 虽然我们是 Numpy 流程，但聚类这一步 O3D 的 C++ 实现比 Sklearn 快太多
            # 所以这里做一次临时转换是值得的
            
            if len(point_out_indices) > self.config.get('cluster_min_points', 3):
                potential_points = points[point_out_indices]
                
                # 临时转 o3d，仅使用 XYZ
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(potential_points[:, :3])
                
                # C++ 极速聚类
                labels = np.array(pcd_temp.cluster_dbscan(
                    eps=self.config.get('cluster_eps', 0.5), 
                    min_points=self.config.get('cluster_min_points', 3), 
                    print_progress=False
                ))
                
                valid_cluster_mask = (labels != -1)
                final_dynamic_indices = point_out_indices[valid_cluster_mask]
                point_out_indices = final_dynamic_indices
            else:
                point_out_indices = np.array([], dtype=int)
            tt6 = time.time()
            print(f"Case1: {(tt2 - tt1)*1000:.2f} ms, Case2: {(tt3 - tt2)*1000:.2f} ms, Case3: {(tt4 - tt3)*1000:.2f} ms, Clustering: {(tt6 - tt4)*1000:.2f} ms")
        t4 = time.time()
        # 5. 生成结果
        is_dynamic_mask = np.zeros(len(points), dtype=bool)
        if len(point_out_indices) > 0:
            is_dynamic_mask[point_out_indices] = True
            
        foreground_points = points[is_dynamic_mask] # 包含 Intensity
        background_points = points[~is_dynamic_mask] # 包含 Intensity
        t5 = time.time()
        # 6. 更新历史
        curr_frame_map = self._create_history_frame(u, v, depth, current_time)
        self.history.append(curr_frame_map)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        
        t6 = time.time()
        print(f"Timing (ms): Preprocess: {(t1 - t0)*1000:.2f}, Projection: {(t2 - t1)*1000:.2f}, Clustering: {(t4 - t2)*1000:.2f}, Result Generation: {(t5 - t4)*1000:.2f}, History Update: {(t6 - t5)*1000:.2f}")

        self.frame_count += 1
        
        # 返回 (背景, 前景)
        return background_points, foreground_points

    def get_timings(self):
        return self.timings