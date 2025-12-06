import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, NavSatFix, Imu # 🔥 新增消息类型
from std_msgs.msg import Header
import numpy as np
import threading
import time
import struct # 🔥 新增 struct 用于解包
import sys

# --- GStreamer 相关导入 ---
from gi.repository import GLib
from gst_receiver_core import create_receiver
import config as config
from rtt_tracker import FeedbackSender

class LidarReceiverNode(Node):
    def __init__(self):
        super().__init__('lidar_receiver_node')
        
        # --- 1. 创建发布者 (现在有3个了) ---
        # (1) 点云
        self.lidar_pub = self.create_publisher(PointCloud2, '/lidar/received_points', 10)
        # (2) GNSS 🔥
        self.gnss_pub = self.create_publisher(NavSatFix, '/lidar/received_gnss', 10)
        # (3) IMU 🔥
        self.imu_pub = self.create_publisher(Imu, '/lidar/received_imu', 10)

        # --- 2. 初始化 RTT 反馈 ---
        self.get_logger().info(f'正在初始化 RTT 反馈 (目标: {config.FEEDBACK_HOST}:{config.FEEDBACK_PORT})...')
        self.feedback = FeedbackSender(
            target_host=config.FEEDBACK_HOST,
            target_port=config.FEEDBACK_PORT,
            verbose=False
        )
        self.feedback.start()

        # --- 3. 初始化 GStreamer 接收器 ---
        self.get_logger().info(f'正在初始化 GStreamer 接收器 (端口: {config.TARGET_PORT})...')
        self.receiver = create_receiver(
            listen_port=config.TARGET_PORT,
            jitter_buffer_latency=config.RECEIVER_JITTER_BUFFER_LATENCY,
            drop_on_latency=config.RECEIVER_DROP_ON_LATENCY,
            verbose=False
        )
        self.receiver.on_packet_received = self.on_packet_received
        
        # --- 4. 启动后台线程 ---
        self.glib_loop = GLib.MainLoop()
        self.gst_thread = threading.Thread(target=self.glib_loop.run)
        self.gst_thread.daemon = True
        self.gst_thread.start()
        
        if not self.receiver.start(self.glib_loop):
            self.get_logger().error('GStreamer 接收管道启动失败！')
            sys.exit(1)
        else:
            self.get_logger().info('GStreamer 接收管道已就绪，等待数据...')
            
        self.total_received = 0
        self.total_lost = 0
        self.last_seq = -1

    def on_packet_received(self, sequence, send_timestamp, receive_timestamp, 
                           filename, data, packet_size, latency_ms):
        """
        回调函数：处理收到的混合数据包
        """
        # --- RTT 统计 ---
        self.total_received += 1
        if self.last_seq != -1 and sequence > self.last_seq + 1:
            self.total_lost += (sequence - self.last_seq - 1)
        self.last_seq = sequence

        if self.feedback:
            self.feedback.send_feedback_now(
                seq=sequence,
                received=self.total_received,
                lost=self.total_lost,
                send_timestamp=send_timestamp
            )
        
        # ==========================================
        # 🔥🔥🔥 核心解包逻辑开始 🔥🔥🔥
        # ==========================================
        
        # 1. 检查数据长度是否足够包含头部 (13 doubles * 8 bytes = 104 bytes)
        HEADER_SIZE = 104
        if len(data) < HEADER_SIZE:
            self.get_logger().warn(f"数据包太小 ({len(data)} bytes)，无法包含 GPS/IMU 头部，丢弃。")
            return

        # 2. 解包头部 (前 104 字节)
        try:
            # 这里的 'ddddddddddddd' 对应发送端的 pack 格式
            header_values = struct.unpack('ddddddddddddd', data[:HEADER_SIZE])
            
            # 提取 GNSS (前3个)
            lat, lon, alt = header_values[0:3]
            
            # 提取 IMU (中间4个姿态 + 后6个数据)
            qx, qy, qz, qw = header_values[3:7]
            ax, ay, az = header_values[7:10]
            gx, gy, gz = header_values[10:13]
            
            # 3. 提取点云 (104 字节之后的所有数据)
            point_cloud_bytes = data[HEADER_SIZE:]
            points = np.frombuffer(point_cloud_bytes, dtype=np.float32).reshape(-1, 4)
            
        except Exception as e:
            self.get_logger().error(f"解包失败: {e}")
            return

        # ==========================================
        # 🔥🔥🔥 发布消息 🔥🔥🔥
        # ==========================================

        # 构造 ROS 时间戳 (从发送端时间恢复，保证同步)
        ros_time = rclpy.time.Time(seconds=send_timestamp).to_msg()

        # --- 1. 发布 GNSS ---
        gnss_msg = NavSatFix()
        gnss_msg.header.stamp = ros_time
        gnss_msg.header.frame_id = "gnss_link" # ⚠️ 确保这里是你 TF 树里的名字
        gnss_msg.latitude = lat
        gnss_msg.longitude = lon
        gnss_msg.altitude = alt
        gnss_msg.status.status = 0 # 假设 Fix
        self.gnss_pub.publish(gnss_msg)

        # --- 2. 发布 IMU ---
        imu_msg = Imu()
        imu_msg.header.stamp = ros_time
        imu_msg.header.frame_id = "imu_link"   # ⚠️ 确保这里是你 TF 树里的名字
        # 填充姿态
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        # 填充线加速度
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        # 填充角速度
        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        self.imu_pub.publish(imu_msg)

        # --- 3. 发布 PointCloud2 ---
        pc_msg = self.create_pointcloud2(points, ros_time)
        self.lidar_pub.publish(pc_msg)
        print(f"已发布帧 {sequence}: GPS({lat:.6f}, {lon:.6f}, {alt:.2f}) | Points: {len(points)}")
        # 打印日志 (可选)
        # self.get_logger().info(f"发布帧 {sequence}: GPS({lat:.2f}, {lon:.2f}) | Points: {len(points)}")

    def create_pointcloud2(self, points, ros_time):
        """
        构建点云消息 (参数改为直接传 ros_time 对象)
        """
        msg = PointCloud2()
        msg.header = Header()
        msg.header.frame_id = "rslidar" # ⚠️ 确保这里是你 TF 树里的名字
        msg.header.stamp = ros_time     # 使用统一的时间戳
        
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()
        
        return msg

    def destroy_node(self):
        if hasattr(self, 'receiver') and self.receiver:
            self.receiver.stop()
        if hasattr(self, 'glib_loop') and self.glib_loop:
            self.glib_loop.quit()
        if hasattr(self, 'gst_thread') and self.gst_thread:
            self.gst_thread.join(timeout=1)
        if hasattr(self, 'feedback') and self.feedback:
            self.feedback.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LidarReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()