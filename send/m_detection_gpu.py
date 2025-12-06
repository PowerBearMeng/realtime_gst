import cupy as cp
import cupyx.scipy.ndimage as ndimage
import cupyx
import numpy as np
import open3d as o3d
import time

class MDetectorGPU:
    """
    M-detector GPU 加速版 (CuPy 实现)
    """
    def __init__(self, config=None):
        if config is None: config = {}
        self.config = config
        
        # 将配置转为 GPU 友好的 float32
        self.hor_res = float(np.deg2rad(config.get('hor_resolution_deg', 0.2)))
        self.ver_res = float(np.deg2rad(config.get('ver_resolution_deg', 2.0)))
        self.fov_up = float(np.deg2rad(config.get('fov_up', 15.0)))
        self.fov_down = float(np.deg2rad(config.get('fov_down', -15.0)))

        self.depth_map_width = int(np.ceil(2 * np.pi / self.hor_res))
        self.depth_map_height = int(np.ceil((self.fov_up - self.fov_down) / self.ver_res))
        
        # 历史帧存储在 GPU 显存中，不要转回 CPU
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

    def _project_to_spherical_gpu(self, points_gpu):
        """GPU 极速投影"""
        # cp.linalg.norm 极快
        depth = cp.linalg.norm(points_gpu, axis=1)
        depth = cp.maximum(depth, 1e-6)
        
        # 极坐标转换
        azimuth = cp.arctan2(points_gpu[:, 1], points_gpu[:, 0])
        elevation = cp.arcsin(cp.clip(points_gpu[:, 2] / depth, -1.0, 1.0))
        
        u = ((cp.pi - azimuth) / self.hor_res).astype(cp.int32)
        v = ((self.fov_up - elevation) / self.ver_res).astype(cp.int32)
        
        u = cp.clip(u, 0, self.depth_map_width - 1)
        v = cp.clip(v, 0, self.depth_map_height - 1)
        
        return u, v, depth

    def _create_history_frame_gpu(self, u, v, depth, timestamp):
        """GPU 上创建并预滤波历史帧"""
        # 初始化无限远/无限近
        min_depth_map = cp.full((self.depth_map_height, self.depth_map_width), cp.inf, dtype=cp.float32)
        max_depth_map = cp.full((self.depth_map_height, self.depth_map_width), -cp.inf, dtype=cp.float32)
        
        # === 核心优化：使用 scatter 替代 at ===
        # CuPy 的 atomic 操作，比循环快一万倍
        # 需要把二维坐标展平
        flat_idx = v * self.depth_map_width + u
        
        # 更新 min_map
        cupyx.scatter_min(min_depth_map.ravel(), flat_idx, depth)
        # 更新 max_map
        cupyx.scatter_max(max_depth_map.ravel(), flat_idx, depth)
        
        # 处理 max_map 的初始值
        max_depth_map[max_depth_map == -cp.inf] = 0.0

        # === 关键优化：预先计算好 Filter ===
        # 这样在 Case 检查时，不需要实时卷积，直接查表 O(1)
        # GPU 上的 filter 也是极速的
        min_filtered = ndimage.minimum_filter(min_depth_map, size=self.kernel_size, mode='nearest')
        max_filtered = ndimage.maximum_filter(max_depth_map, size=self.kernel_size, mode='nearest')

        return {
            "min_depth": min_depth_map, 
            "max_depth": max_depth_map,
            "min_filtered": min_filtered, # 预计算
            "max_filtered": max_filtered, # 预计算
            "timestamp": timestamp
        }

    def _get_adaptive_threshold(self, depth, base_threshold):
        return self.k_depth * depth + base_threshold

    def _find_dynamic_points_case1_gpu(self, u, v, depth):
        # Case 1: Entering
        occlusion_counts = cp.zeros(len(depth), dtype=cp.uint8)
        enter_thr = self._get_adaptive_threshold(depth, self.enter_min_thr1)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr1)

        for frame in self.history:
            # 1. 直接索引 GPU 数组
            hist_min = frame["min_depth"][v, u]
            
            # 2. Enter Check (GPU 向量化)
            is_closer = depth < (hist_min - enter_thr)
            
            # 3. Consistency Check (查预先 Filter 好的表，代替实时计算)
            # 逻辑：当前深度必须在 (邻域最小 - 阈值) 和 (邻域最大 + 阈值) 之间
            # 这里 Case1 只需要检查 min 一致性
            neigh_min = frame["min_filtered"][v, u]
            is_consistent = (depth > (neigh_min - cons_thr)) 
            
            # 逻辑与
            occlusion_counts += (is_closer & (~is_consistent))
            
        return cp.where(occlusion_counts >= self.occ_map_thr1)[0]

    def _find_dynamic_points_case2_gpu(self, u, v, depth, current_time):
        # Case 2: Moving Away
        if len(self.history) < 2: return cp.array([], dtype=cp.int64)
        
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr2)
        
        last_frame = self.history[-1]
        hist_max_last = last_frame["max_depth"][v, u]
        
        delta_t = current_time - last_frame["timestamp"]
        vel_limit = self.v_min_thr2 * delta_t
        final_thr = cp.minimum(base_thr, vel_limit + 1e-3)
        
        is_further = (depth > (hist_max_last + final_thr)) & (hist_max_last > 0.1)
        
        # 递归检查
        recursive_pass = cp.zeros(len(depth), dtype=cp.uint8)
        # 注意：这里如果 history 很长，循环还在 Python 层，但内部计算在 GPU
        for prev_frame in self.history[:-1]:
            hist_max_prev = prev_frame["max_depth"][v, u]
            dt_hist = last_frame["timestamp"] - prev_frame["timestamp"]
            if dt_hist <= 0: continue
            
            v_lim_hist = self.v_min_thr2 * dt_hist
            f_thr_hist = cp.minimum(base_thr, v_lim_hist + 1e-3)
            
            is_prev_moving = (hist_max_last > (hist_max_prev + f_thr_hist)) & (hist_max_prev > 0.1)
            recursive_pass += is_prev_moving

        # 一致性检查 (查 max_filtered 表)
        is_consistent_any = cp.zeros(len(depth), dtype=bool)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr2)
        
        for frame in self.history:
             neigh_max = frame["max_filtered"][v, u]
             # 简单的单边检查或者双边检查
             # 只要比 max 小一点点，就算 consistent
             is_c = depth < (neigh_max + cons_thr)
             is_consistent_any |= is_c

        final_mask = is_further & (recursive_pass >= 1) & (~is_consistent_any)
        return cp.where(final_mask)[0]

    def _find_dynamic_points_case3_gpu(self, u, v, depth, current_time):
        # Case 3: Occluding
        occlusion_counts = cp.zeros(len(depth), dtype=cp.uint8)
        base_thr = self._get_adaptive_threshold(depth, self.occ_depth_thr3)
        cons_thr = self._get_adaptive_threshold(depth, self.map_cons_thr3)

        for frame in self.history:
            hist_min = frame["min_depth"][v, u]
            delta_t = current_time - frame["timestamp"]
            vel_limit = self.v_min_thr3 * delta_t
            final_thr = cp.minimum(base_thr, vel_limit + 1e-3)
            
            is_occluding = depth < (hist_min - final_thr)
            
            # Consistency
            neigh_min = frame["min_filtered"][v, u]
            is_consistent = depth > (neigh_min - cons_thr)
            
            occlusion_counts += (is_occluding & (~is_consistent))

        return cp.where(occlusion_counts >= self.occ_times_thr3)[0]

    def process_frame(self, points_np, timestamp=None):
        t0 = time.time()
        
        # 1. 数据上传到 GPU (Host -> Device)
        # 只上传 XYZ (N, 3) 用于计算
        points_gpu = cp.asarray(points_np[:, :3]) 
        
        # 预处理：移除盲区 (GPU)
        dists = cp.linalg.norm(points_gpu, axis=1)
        valid_mask = dists > self.config.get('blind_dis', 0.5)
        
        points_gpu = points_gpu[valid_mask]
        # 注意：这里我们需要保留原始的 numpy 索引或者 mask，以便最后输出 intensity
        # 为了简单，我们先把 intensity 也传上去，或者最后用 mask 切片 numpy
        
        if len(points_gpu) == 0:
            return np.empty((0, 4)), np.empty((0, 4))
            
        t1 = time.time() # Preprocess + Upload

        # 2. 投影 (GPU)
        u, v, depth = self._project_to_spherical_gpu(points_gpu)
        t2 = time.time() # Projection

        if timestamp is None:
            current_time = self.frame_count * 0.1 
        else:
            current_time = timestamp
        
        # 3. 动态检测 (GPU)
        # 返回的是 GPU 上的 indices
        idx_final = cp.array([], dtype=cp.int64)
        
        if self.history:
            idx_c1 = self._find_dynamic_points_case1_gpu(u, v, depth)
            idx_c2 = self._find_dynamic_points_case2_gpu(u, v, depth, current_time)
            idx_c3 = self._find_dynamic_points_case3_gpu(u, v, depth, current_time)
            
            idx_final = cp.unique(cp.concatenate([idx_c1, idx_c2, idx_c3]))
            
        t3 = time.time() # Detection
        
        # 4. 聚类去噪 (CPU 混合模式)
        # 策略：只把“潜在动态点”传回 CPU 跑 Open3D 聚类
        # 这样比全量传回要快得多
        
        final_fg_mask_cpu = np.zeros(len(points_gpu), dtype=bool) # 长度对应 valid_mask 后的点
        
        # 如果 GPU 上检测到了动态点
        if len(idx_final) > self.config.get('cluster_min_points', 3):
            # A. 提取潜在点 (Device -> Host)
            # 只传输这几千个点，非常快 (0.x ms)
            potential_indices_cpu = idx_final.get() 
            potential_points_cpu = cp.asnumpy(points_gpu[idx_final])
            
            # B. Open3D 聚类 (CPU)
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(potential_points_cpu)
            
            labels = np.array(pcd_temp.cluster_dbscan(
                eps=self.config.get('cluster_eps', 0.5), 
                min_points=self.config.get('cluster_min_points', 3), 
                print_progress=False
            ))
            
            # C. 过滤噪声
            valid_cluster = (labels != -1)
            real_dynamic_indices = potential_indices_cpu[valid_cluster]
            
            final_fg_mask_cpu[real_dynamic_indices] = True

        t4 = time.time() # Clustering

        # 5. 准备输出 (在 CPU 上切分原始带 intensity 的数据)
        # valid_mask 是 GPU array，转回 CPU
        valid_mask_cpu = valid_mask.get()
        
        # 拿到经过盲区过滤的原始数据 (N_valid, 4)
        points_valid_np = points_np[valid_mask_cpu]
        
        foreground_points = points_valid_np[final_fg_mask_cpu]
        background_points = points_valid_np[~final_fg_mask_cpu]
        
        t5 = time.time() # Result Gen

        # 6. 更新历史 (GPU)
        # 这一步完全在 GPU 上进行
        curr_frame_map = self._create_history_frame_gpu(u, v, depth, current_time)
        self.history.append(curr_frame_map)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
            
        self.frame_count += 1
        t6 = time.time()

        # 打印耗时对比 (只打印关键部分)
        print(f"GPU Timing (ms): Proj: {(t2-t1)*1000:.2f}, Detect: {(t3-t2)*1000:.2f}, Cluster: {(t4-t3)*1000:.2f}, Total: {(t6-t0)*1000:.2f}")

        return background_points, foreground_points