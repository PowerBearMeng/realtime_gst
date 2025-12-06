# 文件名: config.py
"""
配置文件 - 所有可调参数集中在这里
"""
import time
# ============= 网络配置 =============
# TARGET_HOST = '10.29.172.205'# '192.168.1.100' # 
# FEEDBACK_HOST = '10.29.242.65' # '192.168.1.101'

TARGET_HOST = '192.168.2.100' 
FEEDBACK_HOST = '192.168.2.101'
TARGET_PORT = 5004
FEEDBACK_PORT = 5005
# ============= 发送端配置 =============

# 数据源
# SOURCE_FOLDER = '/home/mfh/driving/my_mot/pcd/0825/test_i_02_renamed'
SOURCE_FOLDER = '/home/mfh/driving/my_mot/fused_output/0825_test_02_double'
# SOURCE_FOLDER = '/home/mfh/driving/code/RCPCC/output/compressed'
# SOURCE_FOLDER = '/home/mfh/driving/my_mot/pcd/0825/i'
FILE_EXTENSION = '.pcd'  # 支持 .pcd, .bin, .json 等任意格式
LOOP_FILES = False        # 是否循环发送

# 发送参数
SEND_RATE_HZ = 10        # 发送频率（Hz）

# GStreamer 缓冲区设置
SENDER_BUFFER_SIZE_MB = 400                # UDP 发送缓冲区（MB）
SENDER_QUEUE_MAX_BUFFERS = 200            # 队列最大缓冲包数
SENDER_QUEUE_LEAKY = True                # 队列满时是否丢弃旧包

# RTT 测量配置
RTT_MEASUREMENT_ENABLED = True           # ← 新增：是否启用 RTT 测量
RTT_FEEDBACK_INTERVAL = 1.0              # ← 新增：反馈间隔（秒）


# 统计输出
SENDER_CSV = '../stats/sender_stats/1106/' + time.strftime("%Y%m%d_%H%M%S") + '.csv'
SENDER_STATS_PRINT_INTERVAL = 5.0        # 统计打印间隔（秒）

# ============= 接收端配置 =============
# 输出设置
OUTPUT_FOLDER = '../received_pcd'
RECEIVER_CSV = '../stats/receiver_stats/' + time.strftime("_%Y%m%d_%H%M%S") + '.csv'

# GStreamer 缓冲区设置
RECEIVER_JITTER_BUFFER_LATENCY = 50     # 抖动缓冲延迟（毫秒）
RECEIVER_DROP_ON_LATENCY = True          # 超时是否丢包
RECEIVER_APPSINK_MAX_BUFFERS = 200        # appsink 最大缓冲

# 统计输出
RECEIVER_STATS_PRINT_INTERVAL = 5.0      # 统计打印间隔（秒）

# ============= 调试选项 =============
VERBOSE = True           # 是否打印详细日志
SAVE_FILES = False  # 接收端是否保存文件

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