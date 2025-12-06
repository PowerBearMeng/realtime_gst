# 文件名: main_sender.py
"""
发送端主程序
"""

import sys
import os
import signal
import time # 导入 time
from gi.repository import GLib

from config import *
from io_feeder import raw_file_feeder
from gst_sender_core import create_sender
from stats_logger import create_sender_logger
from rtt_tracker import RTTTracker


# 全局变量
logger = None
rtt_tracker = None
sender = None
main_loop = None

# 统计
total_sent = 0
total_bytes = 0
total_lost_frames = 0
latest_loss_rate = 0.0

# --- 核心修改 1: 暂存已发送包的信息 ---
# (用于在收到 RTT 反馈时，关联 'bytes' 和 'timestamp')
sent_packet_info = {}


def on_packet_sent(sequence, filename, data_size, packet_size):
    """
    发送数据包回调
    --- 核心修改 2: 只暂存信息，不记录日志 ---
    """
    global total_sent, total_bytes
    
    total_sent += 1
    total_bytes += packet_size
    
    # 计算相对时间戳
    timestamp = time.time() - logger.start_time
    
    # 暂存信息，等待 on_feedback 时再记录
    sent_packet_info[sequence] = {
        'timestamp': timestamp,
        'bytes': packet_size
    }
    
    if VERBOSE:
        print(f"[{sequence:4d}] {filename:30s} {data_size:8d} B")


def on_feedback(rtt_ms, seq, received, lost):
    """
    收到 RTT 反馈回调
    --- 核心修改 3: 在这里记录日志，解决 RTT 错位问题 ---
    """
    global total_lost_frames, latest_loss_rate
    
    total_lost_frames = lost
    latest_loss_rate = (lost / (seq + 1)) * 100 if seq >= 0 else 0
    
    # 从暂存区取出对应包的信息
    packet_info = sent_packet_info.pop(seq, {}) # .pop() 确保只记录一次，并防止内存泄漏
    
    # 记录到 CSV (现在 RTT 和 seq 是完全对应的)
    row = {
        'timestamp': f"{packet_info.get('timestamp', 0.0):.3f}",
        'seq': seq,
        'bytes': packet_info.get('bytes', 0),
        'rtt_ms': f'{rtt_ms:.2f}',
        'loss_rate': f'{latest_loss_rate:.2f}'
    }
    logger.log_data(row)
    
    if VERBOSE:
        print(f"📡 RTT: {rtt_ms:6.2f} ms | 接收: {received} 帧 | 丢失: {lost} 帧 | 丢帧率: {latest_loss_rate:.1f}%")





def main():
    global logger, rtt_tracker, sender, main_loop
    
    if not os.path.isdir(SOURCE_FOLDER):
        print(f"❌ 文件夹未找到: {SOURCE_FOLDER}")
        print(f"   请在 config.py 中修改 SOURCE_FOLDER")
        sys.exit(1)
    
    print("=" * 80)
    print("📤 发送端启动")
    print("=" * 80)
    print(f"目标地址:   {TARGET_HOST}:{TARGET_PORT}")
    print(f"发送频率:   {SEND_RATE_HZ} Hz")
    print(f"源文件夹:   {SOURCE_FOLDER}")
    print(f"文件类型:   {FILE_EXTENSION}")
    print(f"循环发送:   {'是' if LOOP_FILES else '否'}")
    print(f"反馈端口:   {FEEDBACK_PORT}")
    print(f"CSV 输出:   {SENDER_CSV}")
    print("=" * 80)
    print()
    
    logger = create_sender_logger(SENDER_CSV)
    
    rtt_tracker = RTTTracker(listen_port=FEEDBACK_PORT, verbose=VERBOSE)
    rtt_tracker.on_feedback = on_feedback
    rtt_tracker.start()
    
    sender = create_sender(
        target_host=TARGET_HOST,
        target_port=TARGET_PORT,
        send_rate_hz=SEND_RATE_HZ,
        buffer_size_mb=SENDER_BUFFER_SIZE_MB,
        queue_max_buffers=SENDER_QUEUE_MAX_BUFFERS,
        queue_leaky=SENDER_QUEUE_LEAKY,
        verbose=VERBOSE
    )
    sender.on_packet_sent = on_packet_sent
    
    feeder = raw_file_feeder(SOURCE_FOLDER, FILE_EXTENSION, loop=LOOP_FILES)
    
    def send_next():
        try:
            filename, data = next(feeder)
            sender.send_packet(filename, data)
            return True
        except StopIteration:
            print("\n✓ 发送完成")
            main_loop.quit()
            return False
        except Exception as e:
            print(f"❌ 错误: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            return True
    
    main_loop = GLib.MainLoop()
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    if not sender.start(main_loop):
        sys.exit(1)
    
    GLib.timeout_add(sender.get_send_interval_ms(), send_next)
    
    print("按 Ctrl+C 停止\n")
    print("-" * 80)
    
    try:
        main_loop.run()
    except KeyboardInterrupt:
        print("\n\n⏹  停止中...")
    finally:
        sender.stop()
        rtt_tracker.stop()
        logger.save_to_csv() 
        
        print("\n" + "=" * 80)
        print("📊 最终统计")
        print("=" * 80)
        print(f"总发送:     {total_sent} 帧 ({total_bytes/1e6:.2f} MB)")
        print(f"丢帧率:     {latest_loss_rate:.2f}%")
        rtt = rtt_tracker.get_rtt()
        if rtt:
            print(f"最新 RTT:   {rtt:.2f} ms")
        print("=" * 80)


if __name__ == '__main__':
    main()