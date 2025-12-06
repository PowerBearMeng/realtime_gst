import rclpy
from rclpy. node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import threading
import time
import queue
import concurrent.futures

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
        
        # 3. 初始化 ROS 订阅
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.listener_callback,
            10
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
        """
        🔥 ROS 回调：快速提取数据，立即返回（不阻塞）
        """
        # 1. 快速转换为 Numpy
        points_np = self.pointcloud2_to_array(msg)
        if points_np is None:
            return

        current_frame_id = self.frame_count
        self. frame_count += 1

        # 🔥 2. 提交到线程池处理（立即返回，不阻塞）
        # ⚠️ 改为 self.thread_pool
        self.thread_pool.submit(
            self._process_async,
            points_np,
            current_frame_id
        )
        
        # 回调函数结束 ✅ 不阻塞 ROS
    
    def _process_async(self, points_np, frame_id):
        """
        🔥 后台线程：处理点云（耗时操作）
        """
        t1 = time.perf_counter()
        
        # 算法处理
        # bg_points, fg_points = self.processor.process_frame(points_np)
        # data_to_send = fg_points.astype(np.float32). tobytes()
        
        # 如果要发送原始数据（调试用）：
        data_to_send = points_np.astype(np.float32).tobytes()
        
        t2 = time.perf_counter()
        elapsed_ms = (t2 - t1) * 1000
        
        # 🔥 3. 放入发送队列（不直接调用 send_packet）
        try:
            self.send_queue.put({
                'frame_id': frame_id,
                'data': data_to_send,
                'elapsed_ms': elapsed_ms,
                'timestamp': time.time()
            }, timeout=0.01)  # 设置超时，避免阻塞
        except queue. Full:
            self.get_logger().warn(f'⚠️  发送队列满，丢弃 Frame {frame_id}')
    
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