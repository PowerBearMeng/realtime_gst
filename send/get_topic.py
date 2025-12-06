import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct

# 引入你的算法模块
choice = "MDetector"
choice = "DynamicScene"
from dynamic_scene_processor_optimized import DynamicSceneProcessor
from m_detection import MDetector
from m_detection_gpu import MDetectorGPU

M_DETECTOR_CONFIG = {
    # 基础参数
    'hor_resolution_deg': 0.2,        # 水平分辨率（度）- 比64线粗糙
    'ver_resolution_deg': 2.0,        # 垂直分辨率（度）- 16线间隔大
    'fov_up': 15,                     # 上视场角
    'fov_down': -15,                  # 下视场角
    'blind_dis': 0.2,                 # 盲区距离
    
    # 历史管理
    'max_depth_map_num': 5,           # 减少历史帧数（16线数据稀疏）
    'frame_dur': 0.1,
    
    # Case 1
    'enter_min_thr1': 0.5,
    'map_cons_depth_thr1': 0.3,
    'occluded_map_thr1': 2,
    
    # Case 2
    'occ_depth_thr2': 0.15,
    'map_cons_depth_thr2': 0.2,
    'occluded_times_thr2': 2,
    'v_min_thr2': 0.5, # 最小速度 0.5 m/s
    
    # Case 3
    'occ_depth_thr3': 0.15,
    'map_cons_depth_thr3': 0.2,
    'occluding_times_thr3': 2,
    
    # 通用
    'k_depth': 0.005, # 自适应阈值系数
    'cluster_eps': 3.0, # 聚类半径
    'cluster_min_points': 10,         # 降低最小点数

}

class LidarSenderNode(Node):
    def __init__(self):
        super().__init__('lidar_sender_node')
        
        # 1. 初始化订阅
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.listener_callback,
            10
        )
        
        # --- 新增: 初始化两个发布者 ---
        # 一个发前景 (Foreground)，一个发背景 (Background)
        self.pub_fg = self.create_publisher(PointCloud2, '/lidar/foreground', 10)
        self.pub_bg = self.create_publisher(PointCloud2, '/lidar/background', 10)
        
        # 2. 初始化你的算法类
        if choice == "MDetector":
            self.processor = MDetector(M_DETECTOR_CONFIG)
        elif choice == "MDetectorGPU":
            self.processor = MDetectorGPU(M_DETECTOR_CONFIG)
        elif choice == "DynamicScene":
            self.processor = DynamicSceneProcessor()

        self.get_logger().info('Lidar Sender Node 已启动，等待数据...')

    def listener_callback(self, msg):
        # 1. 极速转 Numpy
        points_np = self.pointcloud2_to_array(msg)
        if points_np is None: return

        t1 = self.get_clock().now()
        
        # 2. 算法处理
        bg_points, fg_points = self.processor.process_frame(points_np)
        
        t2 = self.get_clock().now()
        elapsed = (t2 - t1).nanoseconds / 1e6
        self.get_logger().info(f'算法耗时: {elapsed:.2f} ms | FG: {len(fg_points)}, BG: {len(bg_points)}')

        # --- E. 发布到 ROS 用于可视化 ---
        # 我们需要保留原始 msg 的 header (包含 frame_id 和 时间戳)
        self.publish_point_cloud(self.pub_fg, fg_points, msg.header)
        self.publish_point_cloud(self.pub_bg, bg_points, msg.header)

        # 3. 原有的序列化逻辑 (保持不变，用于后续网络发送)
        # bg_bytes = bg_points.astype(np.float32).tobytes()
        # fg_bytes = fg_points.astype(np.float32).tobytes()
        # self.send_data(...)

    def publish_point_cloud(self, publisher, points, header):
        """
        辅助函数: 将 Numpy (N, 4) 转回 PointCloud2 并发布
        """
        if len(points) == 0:
            return

        # 创建一个新的 PointCloud2 消息
        pc_msg = PointCloud2()
        pc_msg.header = header # 继承原始的时间戳和坐标系
        pc_msg.height = 1
        pc_msg.width = len(points)
        pc_msg.is_dense = True
        pc_msg.is_bigendian = False
        
        # 定义字段: x, y, z, intensity (都是 float32, 4字节)
        # offset 分别是 0, 4, 8, 12
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # 每个点 16 字节
        pc_msg.point_step = 16 
        pc_msg.row_step = pc_msg.point_step * pc_msg.width
        
        # 直接把 numpy 转 bytes 塞进去，极速！
        pc_msg.data = points.astype(np.float32).tobytes()
        
        publisher.publish(pc_msg)

    # def pointcloud2_to_array(self, cloud_msg):
    #     # 极速版: ROS -> Numpy
    #     points = np.frombuffer(cloud_msg.data, dtype=np.float32)
    #     points = points.reshape(-1, 4)
    #     return points
    def pointcloud2_to_array(self, cloud_msg):
        """
        完美适配 26字节 (XYZI + Ring + Time) 的数据
        只提取前 16字节 (XYZI)
        """
        # 1. 先按字节读取 (uint8)，这步永远安全，不会报 buffer size 错
        raw_data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        
        # 2. 获取步长 (你这里应该是 26)
        point_step = cloud_msg.point_step
        
        # 3. 按步长切分：变成 (N, 26) 的矩阵
        try:
            valid_data = raw_data.reshape(-1, point_step)
        except ValueError:
            return None # 防止空数据报错

        # 4. 【关键一步】只取前 16 个字节 (x,y,z,i)，扔掉后面的 ring 和 time
        # .copy() 必不可少，它会把内存复制出来，变成连续的内存块
        xyzi_bytes = valid_data[:, 0:16].copy()
        
        # 5. 现在变成了标准的 (N, 16) 字节，可以直接转 float32 了
        points = xyzi_bytes.view(np.float32).reshape(-1, 4)
        
        return points

def main(args=None):
    rclpy.init(args=args)
    node = LidarSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()