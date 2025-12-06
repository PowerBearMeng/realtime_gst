# 文件名: gst_sender_core.py
"""
GStreamer 发送核心模块
封装所有 GStreamer 相关的发送逻辑
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

import time
import struct
from typing import Optional, Callable


class GstSenderCore:
    """GStreamer 发送核心类"""
    
    def __init__(
        self,
        target_host: str,
        target_port: int,
        send_rate_hz: float,  # ← 保留参数，但实际不使用
        buffer_size_mb: int = 5,
        queue_max_buffers: int = 10,
        queue_leaky: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            target_host: 目标主机 IP
            target_port: 目标端口
            send_rate_hz: 发送频率（Hz）- 设为 0 表示无限制
            buffer_size_mb: UDP 缓冲区大小（MB）
            queue_max_buffers: 队列最大缓冲包数
            queue_leaky: 队列满时是否丢弃旧包
            verbose: 是否打印详细日志
        """
        self. target_host = target_host
        self.target_port = target_port
        self.send_rate_hz = send_rate_hz
        self.verbose = verbose
        
        self.pipeline = None
        self.appsrc = None
        self.main_loop = None
        self. frame_count = 0
        
        # 回调函数
        self.on_packet_sent: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # 初始化 GStreamer
        Gst.init(None)
        
        # 创建管道
        self._create_pipeline(buffer_size_mb, queue_max_buffers, queue_leaky)
    
    def _create_pipeline(self, buffer_size_mb: int, queue_max_buffers: int, queue_leaky: bool):
        """创建 GStreamer 管道"""
        buffer_size_bytes = buffer_size_mb * 1024 * 1024
        leaky_mode = 2 if queue_leaky else 0

        # 🔥 关键修改：去除时间控制，随到随发
        pipeline_str = (
            f"appsrc name=my_source "
            f"is-live=true "
            f"format=time "
            f"!   "

            f"application/x-gst,capsversion=(int)0 !   "

            f"queue "
            f"max-size-buffers={queue_max_buffers} "
            f"max-size-bytes=0 "
            f"max-size-time=0 "  # 🔥 改为 0，不限制时间
            f"leaky={leaky_mode} !   "

            f"rtpgstpay "
            f"config-interval=1 "
            f"pt=96 !  "

            f"application/x-rtp,"
            f"media=(string)application,"
            f"clock-rate=(int)90000,"
            f"encoding-name=(string)X-GST,"
            f"payload=(int)96 ! "

            f"udpsink "
            f"host={self.target_host} "
            f"port={self. target_port} "
            f"sync=false "  # 🔥 保持 false，立即发送
            f"async=false "
            f"buffer-size={buffer_size_bytes}"
        )

        if self.verbose:
            print(f"📡 GStreamer 发送管道:")
            print(f"   {pipeline_str}")
            print(f"   UDP 缓冲: {buffer_size_mb} MB")
            print(f"   队列缓冲: {queue_max_buffers} 包")
            print(f"   丢弃模式: {'启用 (leaky)' if queue_leaky else '禁用 (阻塞)'}")
            print(f"   速率控制: 禁用 (随到随发)\n")

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name('my_source')

        caps = Gst.Caps.from_string("application/x-gst,capsversion=(int)0")
        self.appsrc.set_property('caps', caps)
        self.appsrc.set_property('format', Gst.Format.TIME)
        self.appsrc.set_property('is-live', True)
        self.appsrc.set_property('block', False)  # 🔥 不阻塞

        # 监听管道消息
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
    
    def _on_bus_message(self, bus, message):
        """处理 GStreamer 总线消息"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ GStreamer 错误: {err}")
            if self.verbose:
                print(f"   调试信息: {debug}")
            if self.on_error:
                self.on_error(err, debug)
            if self.main_loop:
                self.main_loop.quit()
        elif t == Gst.MessageType.WARNING:
            if self. verbose:
                warn, debug = message.parse_warning()
                print(f"⚠️  GStreamer 警告: {warn}")
        return True
    
    def pack_metadata(self, filename: str, data: bytes) -> bytes:
        """
        打包元数据和数据
        格式：[seq(8)][timestamp(8)][filename_len(4)][filename][data]
        """
        sequence_number = self.frame_count
        timestamp = time.time()
        filename_bytes = filename.encode('utf-8')
        filename_length = len(filename_bytes)
        
        packed = (
            struct.pack('Q', sequence_number) +
            struct.pack('d', timestamp) +
            struct.pack('I', filename_length) +
            filename_bytes +
            data
        )
        
        return packed
    
    def send_packet(self, filename: str, data: bytes) -> bool:
        """发送一个数据包（一帧 PCD 文件）"""
        packed_data = self.pack_metadata(filename, data)

        buf = Gst.Buffer.new_allocate(None, len(packed_data), None)
        buf. fill(0, packed_data)

        # 🔥 不设置任何时间戳，GStreamer 会立即发送
        # 不需要 pts, dts, duration

        ret = self.appsrc.emit('push-buffer', buf)

        if ret == Gst. FlowReturn.OK:
            self.frame_count += 1
            if self.on_packet_sent:
                self.on_packet_sent(
                    sequence=self.frame_count - 1,
                    filename=filename,
                    data_size=len(data),
                    packet_size=len(packed_data)
                )
            return True
        else:
            if self.verbose:
                print(f"⚠️  push-buffer 失败: {ret}")
            return False
    
    def start(self, main_loop: GLib.MainLoop):
        """启动管道"""
        self.main_loop = main_loop
        ret = self.pipeline. set_state(Gst. State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ 无法启动 GStreamer 管道")
            return False
        if self.verbose:
            print("✓ GStreamer 管道已启动\n")
        return True
    
    def stop(self):
        """停止管道"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.verbose:
            print("✓ GStreamer 管道已停止")
    
    def get_send_interval_ms(self) -> int:
        """获取发送间隔（毫秒）"""
        if self.send_rate_hz > 0:
            return int(1000 / self.send_rate_hz)
        return 0


def create_sender(target_host: str, target_port: int, send_rate_hz: float, **kwargs):
    """便捷函数：创建发送器"""
    return GstSenderCore(
        target_host=target_host,
        target_port=target_port,
        send_rate_hz=send_rate_hz,
        **kwargs
    )