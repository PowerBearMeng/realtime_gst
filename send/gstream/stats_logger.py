# 文件名: stats_logger.py
"""
简化的统计记录器
在内存中缓冲数据，在最后统一写入 CSV
"""

import csv
import time
import os


class StatsLogger:
    """
    统计记录器：在内存中缓冲数据
    """
    
    def __init__(self, csv_path, mode='sender'):
        """
        Args:
            csv_path: CSV 文件路径
            mode: 'sender' 或 'receiver'
        """
        self.csv_path = csv_path
        self.mode = mode
        self.start_time = time.time()
        
        # --- 核心修改 1: 重命名此属性 ---
        self.log_buffer = [] # <-- 从 self.log_data 改为 self.log_buffer
        
        if self.mode == 'sender':
            # 发送端 CSV：时间戳、序列号、字节数、RTT、丢帧率
            self.fieldnames = ['timestamp', 'seq', 'bytes', 'rtt_ms', 'loss_rate', 'eslapsed_ms']
        else:
            # 接收端 CSV：时间戳、序列号、字节数、累计丢帧数
            self.fieldnames = ['timestamp', 'seq', 'bytes', 'lost_frames']
        
        print(f"📝 日志记录器已初始化 (模式: {self.mode})")

    def log_data(self, row_dict):
        """
        --- 核心修改 2: 现在此方法名没有冲突 ---
        (记录一行字典数据到内存)
        """
        self.log_buffer.append(row_dict) # <-- 对应修改
    
    def save_to_csv(self):
        """
        --- 核心修改 3: 在程序结束时调用，将所有数据写入文件 ---
        """
        if not self.log_buffer: # <-- 对应修改
            print("ℹ️  没有日志数据，不写入 CSV。")
            return
            
        print(f"\n⏳ 正在保存 CSV 到: {self.csv_path} ...")
        
        try:
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            
            with open(self.csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(self.log_buffer) # <-- 对应修改
            
            print(f"✓ CSV 已保存: {self.csv_path} (共 {len(self.log_buffer)} 行)") # <-- 对应修改
        
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")

    def close(self):
        """(旧方法，现在由 save_to_csv 替代)"""
        self.save_to_csv()


def create_sender_logger(csv_path):
    """创建发送端记录器"""
    return StatsLogger(csv_path, mode='sender')


def create_receiver_logger(csv_path):
    """创建接收端记录器"""
    return StatsLogger(csv_path, mode='receiver')