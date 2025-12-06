import rclpy
from rclpy. node import Node
from sensor_msgs.msg import PointCloud2, NavSatFix, Imu 
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import threading
import time
import queue
import concurrent.futures
import struct
# --- GStreamer 相关导入 ---
from gi.repository import GLib
from gstream. gst_sender_core import create_sender
import gstream.config as config
from gstream.stats_logger import create_sender_logger
from gstream.rtt_tracker import RTTTracker

# 引入你的算法模块
choice = "DynamicScene"  # 或 "MDetector"
from dynamic_scene_processor_optimized import DynamicSceneProcessor
from m_detection import MDetector


class LidarSenderNode(Node):
    def __init__(self):
        super().__init__('lidar_sender_node')
        
        # 🔥 1. 创建线程池（用于异步处理点云）
        # ⚠️ 不能叫 self.executor，会和 ROS Node 的 executor 冲突
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 🔥 2.  创建队列（用于传递处理结果到发送线程）
        self. send_queue = queue.Queue(maxsize=10)
        self.latest_gnss = None
        self.latest_imu = None
        self.sensor_lock = threading.Lock()
        # 3. 初始化 ROS 订阅
        self.subscription = self.create_subscription(
            PointCloud2,
            '/infra_points',
            self.listener_callback,
            10
        )
        
        self.gnss_sub = self.create_subscription(
            NavSatFix,
            '/infra_gnss/nav_sat_fix', 
            self.gnss_callback,
            qos_profile_sensor_data 
        )

        # --- 3. 订阅 IMU (同样使用 Sensor Data QoS) ---
        self.imu_sub = self.create_subscription(
            Imu,
            '/infra_imu/imu_raw', 
            self.imu_callback,
            qos_profile_sensor_data
        )
        # 4. 初始化统计记录器
        self.get_logger().info(f'正在初始化 CSV 记录器: {config. SENDER_CSV}')
        self.stats_logger = create_sender_logger(config.SENDER_CSV)
        
        # 5.  初始化暂存字典
        self.sent_info = {}
        
        # 6. 初始化处理器
        if choice == "DynamicScene":
            self.processor = DynamicSceneProcessor()
        elif choice == "MDetector":
            self.processor = MDetector(config.M_DETECTOR_CONFIG)
        
        # 7. 初始化 GStreamer 发送器
        self.get_logger().info(f'正在初始化 GStreamer 发送器 (目标: {config.TARGET_HOST}:{config.TARGET_PORT})...')
        
        self.sender = create_sender(
            target_host=config.TARGET_HOST,
            target_port=config.TARGET_PORT,
            send_rate_hz=0,  # 🔥 改为 0，禁用速率控制
            buffer_size_mb=config.SENDER_BUFFER_SIZE_MB,
            queue_max_buffers=20,  # 🔥 增大队列
            verbose=True
        )

        # 8. 初始化 RTT 追踪器
        self.get_logger().info(f'正在监听 RTT 反馈 (端口: {config. FEEDBACK_PORT})...')
        
        self.rtt_tracker = RTTTracker(
            listen_port=config.FEEDBACK_PORT,
            verbose=False
        )
        self.rtt_tracker.on_feedback = self.on_rtt_feedback
        self.rtt_tracker.start()
        
        # 9. 启动 GStreamer 循环
        self.glib_loop = GLib.MainLoop()
        self.gst_thread = threading.Thread(target=self.glib_loop.run)
        self.gst_thread.daemon = True
        self.gst_thread.start()
        
        if not self.sender.start(self.glib_loop):
            self.get_logger().error('GStreamer 管道启动失败！')
        else:
            self.get_logger().info('GStreamer 管道已准备就绪。')

        # 🔥 10. 启动发送线程
        self.running = True
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self. send_thread.start()
        self.get_logger().info('✅ 发送线程已启动')

        self.frame_count = 0

    def gnss_callback(self, msg):
        with self.sensor_lock:
            self.latest_gnss = {
                'lat': msg.latitude,
                'lon': msg.longitude,
                'alt': msg.altitude,
            }

    def imu_callback(self, msg):
        with self.sensor_lock:
            # 提取四元数 (xyzw) 和 线加速度 (xyz) 和 角速度 (xyz)
            # 根据你的需求，通常 V2I 只需要 姿态(Orientation) 和 加速度
            self.latest_imu = {
                'qx': msg.orientation.x,
                'qy': msg.orientation.y,
                'qz': msg.orientation.z,
                'qw': msg.orientation.w,
                'ax': msg.linear_acceleration.x,
                'ay': msg.linear_acceleration.y,
                'az': msg.linear_acceleration.z,
                'gx': msg.angular_velocity.x,
                'gy': msg.angular_velocity.y,
                'gz': msg.angular_velocity.z
            }
    
    def on_rtt_feedback(self, rtt_ms, seq, received, lost):
        """RTT 反馈回调"""
        loss_rate = (lost / (seq + 1) * 100) if seq >= 0 else 0.0
        
        packet_info = self.sent_info.pop(seq, None)
        
        bytes_sent = 0
        timestamp = 0.0
        eslapsed_ms = 0.0
        
        if packet_info:
            bytes_sent = packet_info['bytes']
            timestamp = packet_info['timestamp']
            eslapsed_ms = packet_info['eslapsed_ms']
        else:
            timestamp = time.time()

        row = {
            'timestamp': f"{timestamp:.3f}",
            'seq': seq,
            'bytes': bytes_sent,
            'rtt_ms': f"{rtt_ms:.2f}",
            'loss_rate': f"{loss_rate:.2f}",
            'eslapsed_ms': f"{eslapsed_ms:.2f}"
        }
        
        self.stats_logger.log_data(row)
        
        self.get_logger().info(
            f"📡 RTT: {rtt_ms:.2f} ms | 丢包: {loss_rate:.1f}% | 处理耗时: {eslapsed_ms:.2f} ms | 已存CSV"
        )
    
    def listener_callback(self, msg):
        """LiDAR 主回调：触发打包发送"""
        points_np = self.pointcloud2_to_array(msg)
        if points_np is None: return
        frame_id = self.frame_count
        self.frame_count += 1
        
        # 🔥 快照：在这一瞬间，把 GNSS 和 IMU 的值复制出来
        current_gnss = None
        current_imu = None
        
        with self.sensor_lock:
            # 复制数据，防止处理时被修改
            if self.latest_gnss: 
                current_gnss = self.latest_gnss.copy()
            if self.latest_imu: current_imu = self.latest_imu.copy()

        # 填充默认值 (防止刚启动时为空)
        if current_gnss is None: 
            current_gnss = {'lat':0.0, 'lon':0.0, 'alt':0.0}
        if current_imu is None:
            current_imu = {'qx':0,'qy':0,'qz':0,'qw':1, 'ax':0,'ay':0,'az':0, 'gx':0,'gy':0,'gz':0}

        # 扔进线程池
        self.thread_pool.submit(self._process_async, points_np, frame_id, current_gnss, current_imu)

    def _process_async(self, points_np, frame_id, gnss, imu):
        t1 = time.perf_counter()
        
        # 1. 点云处理 (直接透传或算法处理)
        processed_data = points_np.astype(np.float32).tobytes()
        
        # 2. 打包协议 (Struct Pack)
        # 格式设计:
        # GNSS (3 doubles) + IMU (10 doubles) + PointCloud
        # 总头部大小 = 13 * 8 = 104 bytes
        
        header = struct.pack(
            'ddddddddddddd', # 13个 d (double)
            gnss['lat'], gnss['lon'], gnss['alt'],          # 3个
            imu['qx'], imu['qy'], imu['qz'], imu['qw'],     # 4个 (姿态)
            imu['ax'], imu['ay'], imu['az'],                # 3个 (加速度)
            imu['gx'], imu['gy'], imu['gz']                 # 3个 (角速度)
        )
        
        final_packet = header + processed_data
        

        t2 = time.perf_counter()
        try:
            self.send_queue.put({
                'frame_id': frame_id,
                'data': final_packet,
                'elapsed_ms': (t2-t1)*1000,
                'timestamp': time.time()
            }, timeout=0.01)
        except queue.Full:
            pass
    
    def _send_loop(self):
        """
        🔥 独立发送线程：从队列取数据并发送
        """
        while self.running:
            try:
                # 从队列取出处理好的数据
                item = self.send_queue.get(timeout=0.1)
                
                frame_id = item['frame_id']
                data = item['data']
                elapsed_ms = item['elapsed_ms']
                timestamp = item['timestamp']
                
                # 构造文件名
                timestamp_ns = int(timestamp * 1e9)
                filename = f"fg_{frame_id}_{timestamp_ns}. bin"
                
                # 记录发送信息
                self.sent_info[frame_id] = {
                    'bytes': len(data),
                    'timestamp': timestamp,
                    'eslapsed_ms': elapsed_ms
                }
                
                # 🔥 发送（这里可能阻塞，但不影响 ROS 回调）
                self.sender.send_packet(filename, data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'发送错误: {e}')

    def pointcloud2_to_array(self, cloud_msg):
        """
        完美适配 26字节 (XYZI + Ring + Time) 的数据
        只提取前 16字节 (XYZI)
        """
        raw_data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        point_step = cloud_msg.point_step
        try:
            valid_data = raw_data. reshape(-1, point_step)
        except ValueError:
            return None
        
        # 提取 XYZI (前16字节)
        xyzi_bytes = valid_data[:, 0:16]. copy()
        points = xyzi_bytes.view(np.float32). reshape(-1, 4)
        return points

    def destroy_node(self):
        """优雅退出"""
        # 🔥 停止发送线程
        self.running = False
        if hasattr(self, 'send_thread') and self.send_thread.is_alive():
            self.send_thread.join(timeout=2.0)
        
        # 停止线程池
        # ⚠️ 改为 self.thread_pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # 停止其他组件
        if hasattr(self, 'sender') and self.sender:
            self.sender.stop()
        if hasattr(self, 'glib_loop') and self.glib_loop:
            self.glib_loop.quit()
        if hasattr(self, 'gst_thread') and self.gst_thread.is_alive():
            self. gst_thread.join(timeout=1.0)
        if hasattr(self, 'rtt_tracker') and self.rtt_tracker:
            self. rtt_tracker.stop()
        if hasattr(self, 'stats_logger'):
            self.get_logger().info('正在保存统计数据到 CSV...')
            self. stats_logger.save_to_csv()
        
        super().destroy_node()


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