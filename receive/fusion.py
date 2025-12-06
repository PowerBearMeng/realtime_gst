import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, NavSatFix, Imu
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
import numpy as np
# import message_filters  # 用于时间同步(可选，这里先用最新帧策略)
from scipy.spatial.transform import Rotation as R
import math
import struct
import time

class V2IFusionNode(Node):
    def __init__(self):
        super().__init__('v2i_fusion_node')

        # ==========================================
        # 1. 缓存区 (存储最新收到的数据)
        # ==========================================
        self.sender_gnss = None
        self.sender_imu_rot = None # 存储 Scipy Rotation 对象
        
        self.ego_gnss = None
        self.ego_imu_rot = None    # 存储 Scipy Rotation 对象

        # ==========================================
        # 2. 订阅发送端 (Infrastructure) 的数据
        #    来自你的 receiver_node.py
        # ==========================================
        self.create_subscription(NavSatFix, '/lidar/received_gnss', self.sender_gnss_cb, 10)
        self.create_subscription(Imu, '/lidar/received_imu', self.sender_imu_cb, 10)
        # 点云是主触发信号
        self.create_subscription(PointCloud2, '/lidar/received_points', self.pointcloud_cb, 10)

        # ==========================================
        # 3. 订阅本车 (Ego Vehicle) 的数据
        #    ⚠️ 请修改为你车上真实的 Topic 名字
        # ==========================================
        self.create_subscription(NavSatFix, '/your_gnss/nav_sat_fix', self.ego_gnss_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/your_imu/imu_raw', self.ego_imu_cb, qos_profile_sensor_data)

        # ==========================================
        # 4. 发布融合后的结果
        # ==========================================
        # 这个点云是在你本车坐标系 (base_link) 下的
        self.fused_pub = self.create_publisher(PointCloud2, '/v2i/aligned_points', 10)

        self.get_logger().info("🚀 V2I 融合算法节点已启动...")

    # --- 回调函数：只负责更新数据缓存 ---

    def sender_gnss_cb(self, msg):
        self.sender_gnss = {'lat': msg.latitude, 'lon': msg.longitude, 'alt': msg.altitude}

    def sender_imu_cb(self, msg):
        # 将四元数转为旋转矩阵对象
        self.sender_imu_rot = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def ego_gnss_cb(self, msg):
        self.ego_gnss = {'lat': msg.latitude, 'lon': msg.longitude, 'alt': msg.altitude}

    def ego_imu_cb(self, msg):
        self.ego_imu_rot = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    # --- 主处理逻辑：收到点云时触发融合 ---
    
    def pointcloud_cb(self, pc_msg):
        t1 = time.perf_counter()
        
        # 1. 检查数据是否就绪
        if (self.sender_gnss is None or 
            self.sender_imu_rot is None or 
            self.ego_gnss is None or 
            self.ego_imu_rot is None):  
            self.get_logger().warn("等待 GPS/IMU 数据初始化...", throttle_duration_sec=2.0)
            return

        # 2. 解析点云
        points_sender_frame = self.pointcloud2_to_array(pc_msg)
        if points_sender_frame is None: return
        
        xyz_sender = points_sender_frame[:, :3]
        intensity = points_sender_frame[:, 3:4]

        # ==========================================
        # 🔥 计算相对位姿 (为了打印给你看)
        # ==========================================
        
        # A. 计算世界坐标系下的相对位移 (Translation in World)
        d_x, d_y = self.calc_relative_position(
            self.ego_gnss['lat'], self.ego_gnss['lon'],
            self.sender_gnss['lat'], self.sender_gnss['lon']
        )
        d_z = self.sender_gnss['alt'] - self.ego_gnss['alt']
        T_world = np.array([d_x, d_y, d_z]) 

        # B. 计算本车 IMU 的逆 (用于把世界坐标转回本车坐标)
        R_ego_inv = self.ego_imu_rot.inv()

        # C. 🔥 核心：计算最终作用在点云上的总旋转矩阵
        # 数学意义：R_total = R_ego_inv * R_sender
        # 这代表了从 Sender 坐标系 到 Ego 坐标系 的直接旋转关系
        R_total = R_ego_inv * self.sender_imu_rot
        
        # D. 🔥 核心：计算最终作用在点云上的总平移向量 (在本车坐标系下)
        # 数学意义：T_final = R_ego_inv * T_world
        T_final = R_ego_inv.apply(T_world)


        np.set_printoptions(precision=2, suppress=True)
        
        # 获取欧拉角 (Yaw, Pitch, Roll) 方便直观理解
        euler_sender = self.sender_imu_rot.as_euler('zyx', degrees=True)
        euler_ego    = self.ego_imu_rot.as_euler('zyx', degrees=True)
        euler_relative = R_total.as_euler('zyx', degrees=True)

        print("\n" + "="*40)
        print(f"Frame: {pc_msg.header.frame_id}")
        print(f"📍 [位置差(World)] 北: {d_y:.1f}m, 东: {d_x:.1f}m, 高: {d_z:.1f}m")
        print(f"🚜 [本车姿态(YPR)] {euler_ego}")
        print(f"📡 [路侧姿态(YPR)] {euler_sender}")
        print("-" * 20)
        print(f"🔄 [最终相对旋转(YPR)] {euler_relative}")
        print(f"   (如果这个接近[0,0,0], 说明根本没转)")
        print(f"🔢 [最终相对旋转矩阵 R (3x3)]:\n{R_total.as_matrix()}")
        print(f"➡️ [最终相对平移 T (3x1)]:\n{T_final}")
        print("="*40 + "\n")

        
        # 1. 旋转
        points_rotated = R_total.apply(xyz_sender)
        # 2. 平移
        points_final = points_rotated + T_final
        
        # 3. 拼回强度并发布
        final_points = np.hstack((points_final, intensity))
        
        t2 = time.perf_counter()
        # print(f"融合耗时: {(t2 - t1)*1000:.1f}ms") # 这行可以注释掉，上面已经打印够多了
        
        out_msg = self.array_to_pointcloud2(final_points, pc_msg.header.stamp, "rslidar")
        self.fused_pub.publish(out_msg)



    def calc_relative_position(self, lat_ref, lon_ref, lat_target, lon_target):
        """
        计算 target 相对于 ref 的 x(东), y(北) 距离
        使用简单的等距圆柱投影近似 (V2I 短距离足够精确)
        """
        R_earth = 6378137.0
        d_lat = math.radians(lat_target - lat_ref)
        d_lon = math.radians(lon_target - lon_ref)
        
        # Y轴 = 北向
        y = d_lat * R_earth
        # X轴 = 东向 (需修正纬度收缩)
        x = d_lon * R_earth * math.cos(math.radians((lat_ref + lat_target) / 2))
        
        return x, y

    def pointcloud2_to_array(self, cloud_msg):
        raw_data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        try:
            return raw_data.view(np.float32).reshape(-1, 4)
        except ValueError:
            self.get_logger().error("点云解析失败，请检查格式")
            return None

    def array_to_pointcloud2(self, points, stamp, frame_id):
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * msg.width
        msg.fields = [
            PointField(name='x', offset=0, datatype=7, count=1),
            PointField(name='y', offset=4, datatype=7, count=1),
            PointField(name='z', offset=8, datatype=7, count=1),
            PointField(name='intensity', offset=12, datatype=7, count=1),
        ]
        msg.data = points.astype(np.float32).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = V2IFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()