# 文件名: m_detector_gpu.py
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import cupyx
import numpy as np
import time

# 尝试导入 cuML
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    HAS_CUML = True
except ImportError:
    print("【警告】未检测到 cuML！聚类将无法使用 GPU 加速。")
    HAS_CUML = False

class MDetectorGPU_Ultimate:
    def __init__(self, config=None):
        if config is None: config = {}
        self.config = config
        
        # 参数初始化
        self.hor_res = float(np.deg2rad(config.get('hor_resolution_deg', 0.2)))
        self.ver_res = float(np.deg2rad(config.get('ver_resolution_deg', 2.0)))
        self.fov_up = float(np.deg2rad(config.get('fov_up', 15.0)))
        self.fov_down = float(np.deg2rad(config.get('fov_down', -15.0)))

        self.depth_map_width = int(np.ceil(2 * np.pi / self.hor_res))
        self.depth_map_height = int(np.ceil((self.fov_up - self.fov_down) / self.ver_res))
        
        self.history = [] 
        self.max_history_length = config.get('max_depth_map_num', 5)
        self.frame_count = 0 

        # 阈值参数
        self.k_depth = config.get('k_depth', 0.005) 
        self.enter_min_thr1 = config.get('enter_min_thr1', 1.0)
        self.map_cons_thr1 = config.get('map_cons_depth_thr1', 0.3)
        self.occ_map_thr1 = config.get('occluded_map_thr1', 2)
        self.occ_depth_thr2 = config.get('occ_depth_thr2', 0.15)
        self.map_cons_thr2 = config.get('map_cons_depth_thr2', 0.2)
        self.occ_times_thr2 = config.get('occluded_times_thr2', 2)
        self.v_min_thr2 = config.get('v_min_thr2', 0.5)
        self.occ_depth_thr3 = config.get('occ_depth_thr3', 0.15)
        self.map_cons_thr3 = config.get('map_cons_depth_thr3', 0.2)
        self.occ_times_thr3 = config.get('occluding_times_thr3', 2)
        self.v_min_thr3 = config.get('v_min_thr3', 0.5)

        self.search_radius = 1 
        self.kernel_size = 2 * self.search_radius + 1
        
        # 初始化 cuML DBSCAN
        if HAS_CUML:
            self.dbscan_model = cuDBSCAN(
                eps=config.get('cluster_eps', 0.5), 
                min_samples=config.get('cluster_min_points', 3),
                output_type='cupy'
            )

    def _project_to_spherical_gpu(self, points_gpu):
        depth = cp.linalg.norm(points_gpu, axis=1)
        depth = cp.maximum(depth, 1e-6)
        azimuth = cp.arctan2(points_gpu[:, 1], points_gpu[:, 0])
        elevation = cp.arcsin(cp.clip(points_gpu[:, 2] / depth, -1.0, 1.0))
        u = ((cp.pi - azimuth) / self.hor_res).astype(cp.int32)
        v = ((self.fov_up - elevation) / self.ver_res).astype(cp.int32)
        u = cp.clip(u, 0, self.depth_map_width - 1)
        v = cp.clip(v, 0, self.depth_map_height - 1)
        return u, v, depth

    def _create_history_frame_gpu(self, u, v, depth, timestamp):
        min_depth_map = cp.full((self.depth_map_height, self.depth_map_width), cp.inf, dtype=cp.float32)
        max_depth_map = cp.full((self.depth_map_height, self.depth_map_width), -cp.inf, dtype=cp.float32)
        flat_idx = v * self.depth_map_width + u
        cupyx.scatter_min(min_depth_map.ravel(), flat_idx, depth)
        cupyx.scatter_max(max_depth_map.ravel(), flat_idx, depth)
        max_depth_map[max_depth_map == -cp.inf] = 0.0
        min_filtered = ndimage.minimum_filter(min_depth_map, size=self.kernel_size, mode='nearest')
        max_filtered = ndimage.maximum_filter(max_depth_map, size=self.kernel_size, mode='nearest')
        return {"min_depth": min_depth_map, "max_depth": max_depth_map, "min_filtered": min_filtered, "max_filtered": max_filtered, "timestamp": timestamp}
        
    def _get_adaptive_threshold(self, depth, base_threshold):
        return self.k_depth * depth + base_threshold

    def _find_dynamic_points_case1_gpu(self, u, v, depth):
        occlusion_counts = cp.zeros(len(depth), dtype=cp.uint8)
        enter_thr = self._get_adaptive_threshold(depth, self.enter_min_thr1)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr1)
        for frame in self.history:
            is_closer = depth < (frame["min_depth"][v, u] - enter_thr)
            is_consistent = (depth > (frame["min_filtered"][v, u] - cons_thr)) 
            occlusion_counts += (is_closer & (~is_consistent))
        return cp.where(occlusion_counts >= self.occ_map_thr1)[0]

    def _find_dynamic_points_case2_gpu(self, u, v, depth, current_time):
        if len(self.history) < 2: return cp.array([], dtype=cp.int64)
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr2)
        last_frame = self.history[-1]
        hist_max_last = last_frame["max_depth"][v, u]
        delta_t = current_time - last_frame["timestamp"]
        final_thr = cp.minimum(base_thr, (self.v_min_thr2 * delta_t) + 1e-3)
        is_further = (depth > (hist_max_last + final_thr)) & (hist_max_last > 0.1)
        
        recursive_pass = cp.zeros(len(depth), dtype=cp.uint8)
        for prev_frame in self.history[:-1]:
            dt_hist = last_frame["timestamp"] - prev_frame["timestamp"]
            if dt_hist <= 0: continue
            f_thr_hist = cp.minimum(base_thr, (self.v_min_thr2 * dt_hist) + 1e-3)
            is_prev_moving = (hist_max_last > (prev_frame["max_depth"][v, u] + f_thr_hist)) & (prev_frame["max_depth"][v, u] > 0.1)
            recursive_pass += is_prev_moving

        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr2)
        is_consistent_any = cp.zeros(len(depth), dtype=bool)
        for frame in self.history:
             is_consistent_any |= (depth < (frame["max_filtered"][v, u] + cons_thr))
        final_mask = is_further & (recursive_pass >= 1) & (~is_consistent_any)
        return cp.where(final_mask)[0]

    def _find_dynamic_points_case3_gpu(self, u, v, depth, current_time):
        occlusion_counts = cp.zeros(len(depth), dtype=cp.uint8)
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr3)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr3)
        for frame in self.history:
            final_thr = cp.minimum(base_thr, (self.v_min_thr3 * (current_time - frame["timestamp"])) + 1e-3)
            is_occluding = depth < (frame["min_depth"][v, u] - final_thr)
            is_consistent = depth > (frame["min_filtered"][v, u] - cons_thr)
            occlusion_counts += (is_occluding & (~is_consistent))
        return cp.where(occlusion_counts >= self.occ_times_thr3)[0]

    def process_frame(self, points_np, timestamp=None):
        t0 = time.time()
        # 1. 极速上传 (Host -> Device)
        points_gpu = cp.asarray(points_np[:, :3]) 
        dists = cp.linalg.norm(points_gpu, axis=1)
        valid_mask = dists > self.config.get('blind_dis', 0.5)
        points_gpu = points_gpu[valid_mask]
        
        if len(points_gpu) == 0:
            return np.empty((0, 4)), np.empty((0, 4))

        # 2. 投影
        u, v, depth = self._project_to_spherical_gpu(points_gpu)
        if timestamp is None: timestamp = self.frame_count * 0.1 

        # 3. 检测
        idx_final = cp.array([], dtype=cp.int64)
        if self.history:
            idx_c1 = self._find_dynamic_points_case1_gpu(u, v, depth)
            idx_c2 = self._find_dynamic_points_case2_gpu(u, v, depth, timestamp)
            idx_c3 = self._find_dynamic_points_case3_gpu(u, v, depth, timestamp)
            idx_final = cp.unique(cp.concatenate([idx_c1, idx_c2, idx_c3]))
            
        # 4. 聚类 (cuML)
        final_fg_mask_gpu = cp.zeros(len(points_gpu), dtype=bool)
        if len(idx_final) > self.config.get('cluster_min_points', 3) and HAS_CUML:
            potential_points_gpu = points_gpu[idx_final]
            labels_gpu = self.dbscan_model.fit_predict(potential_points_gpu)
            valid_cluster_mask = (labels_gpu != -1)
            real_dynamic_indices = idx_final[valid_cluster_mask]
            final_fg_mask_gpu[real_dynamic_indices] = True
        elif not HAS_CUML and len(idx_final) > 0:
            final_fg_mask_gpu[idx_final] = True

        # 5. 输出准备 (Device -> Host)
        valid_mask_cpu = valid_mask.get()
        final_fg_mask_cpu = final_fg_mask_gpu.get()
        
        # 使用原始数据切片 (为了保留 Intensity)
        points_valid_np = points_np[valid_mask_cpu]
        fg_points = points_valid_np[final_fg_mask_cpu]
        bg_points = points_valid_np[~final_fg_mask_cpu]

        # 6. 更新历史
        curr_frame_map = self._create_history_frame_gpu(u, v, depth, timestamp)
        self.history.append(curr_frame_map)
        if len(self.history) > self.max_history_length: self.history.pop(0)
        self.frame_count += 1
        
        # print(f"GPU Total Time: {(time.time()-t0)*1000:.2f}ms")
        return bg_points, fg_points