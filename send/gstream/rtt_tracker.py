# 文件名: rtt_tracker.py
"""
简化的 RTT 追踪器
只负责发送/接收反馈包，不做统计计算
"""

import socket
import struct
import time
import threading


# 反馈包格式：[magic(4)][original_send_timestamp(8)][seq(8)][received(8)][lost(8)]
MAGIC = 0x52544650  # "RTFP"
FEEDBACK_FORMAT = '!IdQQQ'
FEEDBACK_SIZE = struct.calcsize(FEEDBACK_FORMAT)


class RTTTracker:
    """RTT 追踪器（发送端使用）"""
    
    def __init__(self, listen_port, verbose=False):
        """
        Args:
            listen_port: 监听反馈的端口
            verbose: 是否打印详细日志
        """
        self.listen_port = listen_port
        self.verbose = verbose
        self.sock = None
        self.running = False
        self.thread = None
        
        # RTT 数据
        self.latest_rtt = None
        
        # 接收端反馈的统计
        self.receiver_last_seq = 0
        self.receiver_received = 0
        self.receiver_lost = 0
        
        # 回调函数
        self.on_feedback = None
    
    def start(self):
        """启动 RTT 测量"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(('', self.listen_port))
            self.sock.settimeout(0.1)
            
            self.running = True
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()
            
            if self.verbose:
                print(f"✓ RTT 追踪器已启动，监听端口: {self.listen_port}")
        
        except Exception as e:
            print(f"❌ RTT 追踪器启动失败: {e}")
    
    def stop(self):
        """停止 RTT 测量"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.sock:
            self.sock.close()
        if self.verbose:
            print("✓ RTT 追踪器已停止")
    
    def _listen(self):
        """监听反馈包"""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                if len(data) >= FEEDBACK_SIZE:
                    self._process_feedback(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running and self.verbose:
                    print(f"⚠️  接收反馈包错误: {e}")
    
    def _process_feedback(self, data):
        """处理反馈包"""
        try:
            magic, original_send_time, seq, received, lost = struct.unpack(
                FEEDBACK_FORMAT, data[:FEEDBACK_SIZE]
            )
            
            if magic != MAGIC:
                return
            
            rtt_ms = (time.time() - original_send_time) * 1000
            self.latest_rtt = rtt_ms
            
            self.receiver_last_seq = seq
            self.receiver_received = received
            self.receiver_lost = lost
            
            if self.on_feedback:
                self.on_feedback(rtt_ms, seq, received, lost)
        
        except Exception as e:
            if self.verbose:
                print(f"❌ 解析反馈包失败: {e}")
    
    def get_rtt(self):
        """获取最新 RTT（毫秒）"""
        return self.latest_rtt
    
    def get_loss_rate(self):
        """获取接收端丢帧率（%）"""
        if self.receiver_last_seq == 0:
            return 0.0
        expected = self.receiver_last_seq + 1
        return (self.receiver_lost / expected) * 100


# --- 核心修改从这里开始 ---

class FeedbackSender:
    """反馈发送器（接收端使用）"""
    
    def __init__(self, target_host, target_port, verbose=False):
        """
        Args:
            target_host: 发送端 IP
            target_port: 发送端反馈监听端口
            verbose: 是否打印详细日志
        """
        self.target_host = target_host
        self.target_port = target_port
        self.verbose = verbose
        self.sock = None
        # 移除了 interval, running, thread 和所有状态变量
    
    def start(self):
        """启动反馈发送（只创建 Socket）"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            if self.verbose:
                print(f"✓ 反馈发送器已启动 (Socket ready)，目标: {self.target_host}:{self.target_port}")
        
        except Exception as e:
            print(f"❌ 反馈发送器启动失败: {e}")
    
    def stop(self):
        """停止反馈发送（只关闭 Socket）"""
        if self.sock:
            self.sock.close()
        if self.verbose:
            print("✓ 反馈发送器已停止")
    
    
    def send_feedback_now(self, seq, received, lost, send_timestamp):
        """
        立即发送一个反馈包
        (在 on_packet_received 中被直接调用)
        """
        # 如果 socket 没创建成功，则不发送
        if not self.sock:
            return
            
        try:
            data = struct.pack(
                FEEDBACK_FORMAT,
                MAGIC,
                send_timestamp, # 原始发送时间
                seq,
                received,
                lost
            )
            self.sock.sendto(data, (self.target_host, self.target_port))
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  发送反馈包错误: {e}")