# 文件名: ros_bridge.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import zmq
import struct

class LidarBridgeNode(Node):
    def __init__(self):
        super().__init__('lidar_bridge_node')
        
        # 1. 连接 GPU 服务
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        ipc_addr = "ipc:///tmp/lidar_gpu_service.ipc"
        
        self.get_logger().info(f"正在连接 GPU 服务: {ipc_addr} ...")
        self.socket.connect(ipc_addr)
        
        # 2. ROS 设置
        self.sub = self.create_subscription(
            PointCloud2, '/rslidar_points', self.listener_callback, 10
        )
        self.pub_bg = self.create_publisher(PointCloud2, '/lidar/background', 10)
        self.pub_fg = self.create_publisher(PointCloud2, '/lidar/foreground', 10)
        
        self.get_logger().info("ROS 桥接节点就绪！")

    def listener_callback(self, msg):
        # 1. 极速解析 ROS -> Numpy
        points_np = self.pointcloud2_to_array(msg)
        if points_np is None or len(points_np) == 0: return
            
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        try:
            # 2. 发送请求给 GPU
            ts_bytes = struct.pack('d', timestamp)
            rows_bytes = struct.pack('I', points_np.shape[0])
            # points_np.tobytes() 是很快的内存拷贝
            self.socket.send_multipart([ts_bytes, rows_bytes, points_np.tobytes()])
            
            # 3. 等待结果 (阻塞)
            parts = self.socket.recv_multipart()
            
            # 4. 解析并发布
            bg_rows = struct.unpack('I', parts[0])[0]
            bg_bytes = parts[1]
            fg_rows = struct.unpack('I', parts[2])[0]
            fg_bytes = parts[3]
            
            self.publish_bytes(self.pub_bg, bg_bytes, bg_rows, msg.header)
            self.publish_bytes(self.pub_fg, fg_bytes, fg_rows, msg.header)
            
            # self.get_logger().info(f"FPS: FG={fg_rows} BG={bg_rows}")

        except zmq.ZMQError as e:
            self.get_logger().error(f"ZMQ Error: {e}")

    def publish_bytes(self, publisher, data_bytes, num_points, header):
        if num_points == 0: return
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = num_points
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = 16 * num_points
        msg.data = data_bytes
        publisher.publish(msg)

    def pointcloud2_to_array(self, cloud_msg):
        # 极速解析 PointCloud2 -> Numpy (假设 XYZI 16字节)
        try:
            return np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, 4)
        except Exception:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = LidarBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()