# 文件名: gpu_server.py
import time
import zmq
import numpy as np
import struct
from m_detector_gpu import MDetectorGPU_Ultimate

def main():
    print(">>> [GPU进程] 初始化 cuML 算法引擎...")
    # 配置你的雷达参数
    config = {
        'hor_resolution_deg': 0.2, 
        'ver_resolution_deg': 2.0,
        'fov_up': 15.0,
        'fov_down': -15.0,
        'cluster_eps': 0.6,
        'cluster_min_points': 5
    }
    detector = MDetectorGPU_Ultimate(config)
    
    # 建立 ZMQ 服务 (IPC模式)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    ipc_addr = "ipc:///tmp/lidar_gpu_service.ipc"
    socket.bind(ipc_addr)
    
    print(f">>> [GPU进程] 服务启动成功！监听地址: {ipc_addr}")
    print(">>> [GPU进程] 等待 ROS 数据流...")

    while True:
        try:
            # 1. 接收: [时间戳, 行数, 数据体]
            parts = socket.recv_multipart()
            
            # 解析头部
            timestamp = struct.unpack('d', parts[0])[0]
            rows = struct.unpack('I', parts[1])[0]
            
            # 零拷贝转 Numpy (从 bytes 直接映射，不消耗 CPU)
            points_np = np.frombuffer(parts[2], dtype=np.float32).reshape(rows, -1)
            
            # 2. GPU 极速推理
            bg_points, fg_points = detector.process_frame(points_np, timestamp)
            
            # 3. 发送结果
            bg_data = bg_points.astype(np.float32)
            fg_data = fg_points.astype(np.float32)
            
            bg_rows_bytes = struct.pack('I', bg_data.shape[0])
            fg_rows_bytes = struct.pack('I', fg_data.shape[0])
            
            socket.send_multipart([
                bg_rows_bytes, bg_data.tobytes(),
                fg_rows_bytes, fg_data.tobytes()
            ])
            
        except KeyboardInterrupt:
            print("停止服务...")
            break
        except Exception as e:
            print(f"Server Error: {e}")
            socket.send_multipart([b'', b'', b'', b''])

if __name__ == "__main__":
    main()