import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu


STATIC_LAT = 39.961151123046875
STATIC_LON = 116.35480499267578
STATIC_ALT = 41.53900146484375

STATIC_IMU_Q = [-0.008162485316089308, 0.0008378955216517463, 0.01481927375229791, 0.9998565201517735]
STATIC_IMU_ACC = [0.007568359375, 0.04296875, 0.999267578125]
STATIC_IMU_GYRO = [0.0213623046875, -0.03662109375, 0.0518798828125]


class FakeSensorNode(Node):
    def __init__(self):
        super().__init__('fake_sensor_node')

        # 1. 话题名称
        self.gnss_pub = self.create_publisher(NavSatFix, '/infra_gnss/nav_sat_fix', 10)
        self.imu_pub = self.create_publisher(Imu, '/infra_imu/imu_raw', 10)

        # 2. 定时器: 100Hz (0.01秒)
        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.get_logger().info("🚀 虚拟传感器已启动 (100Hz)...")

    def timer_callback(self):
        # 获取当前系统时间 (解决时间同步)
        now = self.get_clock().now().to_msg()

        # --- 发布 GNSS ---
        gnss_msg = NavSatFix()
        gnss_msg.header.stamp = now
        gnss_msg.header.frame_id = "gnss_link"
        
        # 状态设为 0 (STATUS_FIX)
        gnss_msg.status.status = 0 
        gnss_msg.status.service = 1
        
        gnss_msg.latitude = float(STATIC_LAT)
        gnss_msg.longitude = float(STATIC_LON)
        gnss_msg.altitude = float(STATIC_ALT)
        
        # 协方差不填充，保持默认的 0，并将类型设为 UNKNOWN (0)
        gnss_msg.position_covariance_type = 0
        
        self.gnss_pub.publish(gnss_msg)

        # --- 发布 IMU ---
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "imu_link"
        
        # 姿态 (用于旋转)
        imu_msg.orientation.x = float(STATIC_IMU_Q[0])
        imu_msg.orientation.y = float(STATIC_IMU_Q[1])
        imu_msg.orientation.z = float(STATIC_IMU_Q[2])
        imu_msg.orientation.w = float(STATIC_IMU_Q[3])
        
        # 加速度
        imu_msg.linear_acceleration.x = float(STATIC_IMU_ACC[0])
        imu_msg.linear_acceleration.y = float(STATIC_IMU_ACC[1])
        imu_msg.linear_acceleration.z = float(STATIC_IMU_ACC[2])
        
        # 角速度
        imu_msg.angular_velocity.x = float(STATIC_IMU_GYRO[0])
        imu_msg.angular_velocity.y = float(STATIC_IMU_GYRO[1])
        imu_msg.angular_velocity.z = float(STATIC_IMU_GYRO[2])
        
        # 协方差保持默认 (全部为 0.0)
        
        self.imu_pub.publish(imu_msg)

def main():
    rclpy.init()
    node = FakeSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()