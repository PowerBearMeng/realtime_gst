# 文件名: main_receiver.py
"""
接收端主程序
"""

import sys
import os
import signal
import time # 导入 time
from gi.repository import GLib

from config import *
from gst_receiver_core import create_receiver
from stats_logger import create_receiver_logger
from rtt_tracker import FeedbackSender


# 全局变量
logger = None
feedback = None
receiver = None
main_loop = None

# 统计
total_received = 0
total_bytes = 0
last_seq = -1
total_lost_frames = 0


def on_packet_received(sequence, send_timestamp, receive_timestamp,
                       filename, data, packet_size, latency_ms):
    """接收数据包回调"""
    global total_received, total_bytes, last_seq, total_lost_frames
    
    if sequence <= last_seq:
        if VERBOSE:
            print(f"⚠️  丢弃乱序/重复包: 序列号 {sequence} (已收到 {last_seq})")
        return
        
    total_received += 1
    total_bytes += packet_size
    
    if sequence > last_seq + 1:
        lost_frames = sequence - last_seq - 1
        total_lost_frames += lost_frames
        if VERBOSE:
            print(f"⚠️  检测到丢帧: 序列号 {last_seq + 1} 到 {sequence - 1} (丢失 {lost_frames} 帧)")
    
    last_seq = sequence
    
    # --- 核心修改 1: 适配新的 logger 方法 ---
    # 计算相对时间戳
    timestamp = time.time() - logger.start_time
    
    # 记录到 CSV
    row = {
        'timestamp': f'{timestamp:.3f}',
        'seq': sequence,
        'bytes': packet_size,
        'lost_frames': total_lost_frames
    }
    logger.log_data(row)
    
    # 更新反馈
    if feedback:
        feedback.send_feedback_now(
            seq=sequence,
            received=total_received,
            lost=total_lost_frames,
            send_timestamp=send_timestamp 
        )
    
    if SAVE_FILES:
        output_path = os.path.join(OUTPUT_FOLDER, f"rx_{total_received:04d}_{filename}")
        try:
            with open(output_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    if VERBOSE:
        print(f"[{sequence:4d}] {filename:30s} {len(data):8d} B | 延迟: {latency_ms:6.2f} ms")


def print_stats():
    """定时打印统计"""
    if total_received > 0:
        loss_rate = (total_lost_frames / (last_seq + 1)) * 100 if last_seq >= 0 else 0
        throughput_mbps = (total_bytes * 8) / (total_received / SEND_RATE_HZ) / 1e6 if total_received > 0 else 0
        
        print(f"📊 已收: {total_received:4d} 帧 | {total_bytes/1e6:.1f} MB | "
              f"吞吐量: {throughput_mbps:.2f} Mbps | "
              f"丢帧率: {loss_rate:.1f}%")
    return True


def main():
    global logger, feedback, receiver, main_loop
    
    if SAVE_FILES:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 80)
    print("📥 接收端启动")
    print("=" * 80)
    print(f"监听端口:   {TARGET_PORT}")
    print(f"反馈目标:   {FEEDBACK_HOST}:{FEEDBACK_PORT}")
    print(f"输出文件夹: {OUTPUT_FOLDER if SAVE_FILES else '不保存文件'}")
    print(f"CSV 输出:   {RECEIVER_CSV}")
    print("=" * 80)
    print()
    
    logger = create_receiver_logger(RECEIVER_CSV)
    
    feedback = FeedbackSender(
        target_host=FEEDBACK_HOST,
        target_port=FEEDBACK_PORT,
        verbose=VERBOSE
    )
    feedback.start()
    
    receiver = create_receiver(
        listen_port=TARGET_PORT,
        jitter_buffer_latency=RECEIVER_JITTER_BUFFER_LATENCY,
        drop_on_latency=RECEIVER_DROP_ON_LATENCY,
        appsink_max_buffers=RECEIVER_APPSINK_MAX_BUFFERS,
        verbose=VERBOSE
    )
    receiver.on_packet_received = on_packet_received
    
    main_loop = GLib.MainLoop()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    if not receiver.start(main_loop):
        sys.exit(1)
    
    GLib.timeout_add(1000, print_stats)
    
    print("按 Ctrl+C 停止\n")
    print("-" * 80)
    
    try:
        main_loop.run()
    except KeyboardInterrupt:
        print("\n\n⏹  停止中...")
    finally:
        receiver.stop()
        feedback.stop()
        
        # --- 核心修改 2: 调用新的 save 方法 ---
        logger.save_to_csv()
        
        print("\n" + "=" * 80)
        print("📊 最终统计")
        print("=" * 80)
        print(f"总接收:     {total_received} 帧 ({total_bytes/1e6:.2f} MB)")
        loss_rate = (total_lost_frames / (last_seq + 1)) * 100 if last_seq >= 0 else 0
        print(f"丢帧率:     {loss_rate:.2f}%")
        print("=" * 80)


if __name__ == '__main__':
    main()