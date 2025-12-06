import open3d as o3d
import numpy as np
import time
import random
import threading
import cupy as cp
from collections import namedtuple, Counter

# --- 配置区域 ---
ADAPTIVE_LAYERS = {
    0: {'distance_range': (0, 5),    'voxel_size': 0.10},
    1: {'distance_range': (5, 10),   'voxel_size': 0.15},
    2: {'distance_range': (10, 20),  'voxel_size': 0.20},
    4: {'distance_range': (20, 25),  'voxel_size': 0.25},
    5: {'distance_range': (25, 30),  'voxel_size': 0.30},
    6: {'distance_range': (30, 50),  'voxel_size': 0.35},
    8: {'distance_range': (50, 60),  'voxel_size': 0.40},
    9: {'distance_range': (60, 75),  'voxel_size': 0.45},
    10: {'distance_range': (75, 150), 'voxel_size': 0.50}
}

def read_points_file(file_path):
    """通用点云读取函数"""
    try:
        if file_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(file_path)
        elif file_path.endswith('.bin'):
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        else:
            return None
        return pcd
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None

def _adaptive_voxelize_gpu(points_np, layer_definitions):
    """GPU 高速体素化，返回 CuPy 数组格式的 Keys"""
    if points_np.shape[0] == 0:
        return cp.array([], dtype=cp.int64)

    points = cp.asarray(points_np[:, :3])
    distances = cp.linalg.norm(points, axis=1)
    
    sorted_layers = sorted(layer_definitions.items())
    bins = cp.array([cfg['distance_range'][1] for _, cfg in sorted_layers])
    point_layer_indices = cp.searchsorted(bins, distances)

    voxel_keys = cp.full(points.shape[0], -1, dtype=cp.int64)

    for idx, (layer_id, cfg) in enumerate(sorted_layers):
        mask = (point_layer_indices == idx)
        if not cp.any(mask): continue
            
        pts_in_layer = points[mask]
        vox = cp.floor(pts_in_layer / cfg['voxel_size']).astype(cp.int32)
        
        key = (cp.int64(layer_id) << 48) | \
              (cp.bitwise_and(vox[:, 0].astype(cp.int64), 0xFFFF) << 32) | \
              (cp.bitwise_and(vox[:, 1].astype(cp.int64), 0xFFFF) << 16) | \
              (cp.bitwise_and(vox[:, 2].astype(cp.int64), 0xFFFF))
        
        voxel_keys[mask] = key

    return voxel_keys

def filter_clusters_by_size_np(points: np.ndarray, max_cluster_size, eps=0.8, min_points=5):
    """Open3D C++ DBSCAN 聚类过滤"""
    if points.shape[0] < min_points: return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.max() < 0: return np.empty((0, 3), dtype=points.dtype)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    valid_labels = unique_labels[counts <= max_cluster_size]
    if len(valid_labels) == 0: return np.empty((0, 3), dtype=points.dtype)
    mask = np.isin(labels, valid_labels)
    return points[mask]

Frame = namedtuple('Frame', ['timestamp', 'points', 'unique_keys'])

class TemporalFrameBuffer:
    """智能帧缓冲区，缓存 Unique Keys 以加速 Counter 更新"""
    def __init__(self, max_duration_seconds: float, max_size: int):
        self.max_duration = max_duration_seconds
        self.max_size = max_size
        self.buffer = []

    def add(self, frame_points: np.ndarray, frame_unique_keys: np.ndarray) -> list:
        current_time = time.monotonic()
        new_frame = Frame(timestamp=current_time, points=frame_points, unique_keys=frame_unique_keys)
        
        removed_data = []
        cutoff_time = current_time - self.max_duration
        
        kept_frames = []
        for frame in self.buffer:
            if frame.timestamp >= cutoff_time:
                kept_frames.append(frame)
            else:
                removed_data.append(frame.unique_keys)
        self.buffer = kept_frames

        if len(self.buffer) < self.max_size:
            self.buffer.append(new_frame)
        else:
            idx = random.randint(0, self.max_size - 1)
            removed_data.append(self.buffer[idx].unique_keys)
            self.buffer[idx] = new_frame
        
        return removed_data
            
    def get_frames(self):
        return [frame.points for frame in self.buffer]

class DynamicSceneProcessor:
    def __init__(self,
                 build_frequency: float = 0.7,
                 min_frames_for_build: int = 10,
                 buffer_duration_sec: float = 5.0,
                 max_frames_in_buffer: int = 50,
                 max_cluster_size: int = 200,
                 maintenance_stride: int = 4,
                 layer_definitions: dict = ADAPTIVE_LAYERS):
        
        self.build_freq = build_frequency
        self.min_frames_for_build = min_frames_for_build
        self.max_cluster_size = max_cluster_size
        self.stride = maintenance_stride
        self.layers = layer_definitions

        self.voxel_counter = Counter()
        self.background_keys_sorted = np.array([], dtype=np.int64)
        
        self.frames_buffer = TemporalFrameBuffer(max_duration_seconds=buffer_duration_sec, max_size=max_frames_in_buffer)
        self._new_frames_count = 0
        self.background_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._update_event = threading.Event()
        
        self.update_thread = threading.Thread(target=self._background_builder_task, daemon=True)
        self.update_thread.start()

    def _background_builder_task(self):
        while not self._stop_event.is_set():
            self._update_event.wait()
            self._update_event.clear()
            if self._stop_event.is_set(): break

            with self.background_lock:
                self._new_frames_count = 0
                # 过滤低频噪声
                current_voxel_counts = {k: v for k, v in self.voxel_counter.items() if v > 1}
            
            num_frames = len(self.frames_buffer.buffer)
            if num_frames == 0: continue

            freq_threshold = num_frames * self.build_freq
            bg_keys = [v for v, c in current_voxel_counts.items() if c >= freq_threshold]

            if bg_keys:
                bg_keys_arr = np.array(bg_keys, dtype=np.int64)
                bg_keys_arr.sort() # 必须排序以供 searchsorted 使用
                with self.background_lock:
                    self.background_keys_sorted = bg_keys_arr
            # print('[DynamicSceneProcessor] Background model updated with '
            #       f'{len(bg_keys)} voxels from {num_frames} frames.')

    def process_frame(self, frame_points: np.ndarray):
        if not isinstance(frame_points, np.ndarray) or frame_points.shape[0] == 0:
            return np.empty((0, 4)), None

        # 1. 全量 GPU 体素化
        keys_gpu_full = _adaptive_voxelize_gpu(frame_points, self.layers)
        
        keys_gpu_sampled = keys_gpu_full

        valid_mask = keys_gpu_sampled >= 0
        unique_keys_cpu = cp.asnumpy(cp.unique(keys_gpu_sampled[valid_mask]))
        
        removed_keys_list = self.frames_buffer.add(frame_points, unique_keys_cpu)

        should_trigger_build = False
        with self.background_lock:
            self.voxel_counter.update(unique_keys_cpu)
            for old_keys in removed_keys_list:
                self.voxel_counter.subtract(old_keys)
            
            if self._new_frames_count % 50 == 0:
                self.voxel_counter = +self.voxel_counter
            
            self._new_frames_count += 1
            if self._new_frames_count >= self.min_frames_for_build:
                should_trigger_build = True

        if should_trigger_build:
            self._update_event.set()

        # 3. 推理阶段 (使用全量数据 + searchsorted)
        with self.background_lock:
            bg_keys_sorted = self.background_keys_sorted
        
        if bg_keys_sorted.size == 0:
             return np.empty((0, 4)), frame_points

        full_keys_cpu = cp.asnumpy(keys_gpu_full)
        
        idx = np.searchsorted(bg_keys_sorted, full_keys_cpu)
        idx[idx == len(bg_keys_sorted)] = 0
        is_background_mask = (bg_keys_sorted[idx] == full_keys_cpu)
        
        foreground_points = frame_points[~is_background_mask]
        background_points = frame_points[is_background_mask] # 可选，仅用于可视化

        # 4. 聚类过滤
        # if foreground_points.shape[0] > 0:
        #     final_objects = filter_clusters_by_size_np(foreground_points, self.max_cluster_size)
        # else:
        #     final_objects = np.empty((0, 4))

        return  background_points, foreground_points

    def stop(self):
        self._stop_event.set()
        self._update_event.set()
        if self.update_thread.is_alive():
            self.update_thread.join()