# 文件名: io_feeder.py
import os
import numpy as np
# import open3d as o3d  <--- 删除
from pypcd import pypcd # <--- 导入 pypcd
import time

def folder_feeder(folder_path: str, extension: str):
    """
    智能 "喂食器", 使用 pypcd 解析文件。
    会同时读取 XYZ 和 Intensity。
    """
    print(f"正在从文件夹 '{folder_path}' 查找 {extension} 文件...")
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(extension)])
        if not files:
            print(f"错误: 在 '{folder_path}' 中没有找到 {extension} 文件。")
            return
    except FileNotFoundError:
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return

    print(f"开始预解析所有 {len(files)} 个 {extension} 文件到内存...")
    preloaded_data = []
    start_preload_time = time.monotonic()
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            if extension == ".json":
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                preloaded_data.append(("json_data", raw_data, filename))
                
            elif extension == ".pcd":
                # --- 核心修复: 使用 pypcd ---
                pc = pypcd.PointCloud.from_path(file_path)
                
                # pypcd 会将所有字段加载到 pc.pc_data (一个 structured NumPy array)
                # 我们可以通过字段名访问它们
                
                # 1. 提取 XYZ
                # (使用 np.vstack 和 .T 比 stack 更快)
                np_points = np.vstack([pc.pc_data['x'], 
                                       pc.pc_data['y'], 
                                       pc.pc_data['z']]).T.astype(np.float32)
                
                np_intensities = None
                # 2. 提取 Intensity (如果存在)
                if 'intensity' in pc.fields:
                     np_intensities = pc.pc_data['intensity'].astype(np.float32)
                else:
                    print(f"警告: {filename} 中未找到 'intensity' 字段 (pypcd)。")

                data_object = {
                    "points": np_points, # (N, 3)
                    "intensities": np_intensities # (N,) or None
                }
                preloaded_data.append(("pcd_data", data_object, filename))
            
        except Exception as e:
            print(f"警告: pypcd 解析文件 {filename} 失败: {e}")
            
    end_preload_time = time.monotonic()
    print(f"✔️ 所有 {len(preloaded_data)} 个文件已预解析完毕，耗时 {(end_preload_time - start_preload_time):.2f} 秒。")

    for data_key, data_object, filename in preloaded_data:
        yield data_key, data_object, filename


def raw_file_feeder(folder_path: str, extension: str, loop: bool = False):
    """
    一个简单的"喂食器"，用于读取文件的原始二进制内容。

    它会遍历文件夹，找到匹配扩展名的文件，
    然后 yield (文件名, 文件的原始字节)。

    参数:
    folder_path (str): 要扫描的文件夹路径。
    extension (str): 要匹配的文件扩展名 (例如 '.pcd', '.bin', '.json')。
    loop (bool): 是否在发送完所有文件后循环播放。
    """
    print(f"Feeder: 正在从文件夹 '{folder_path}' 查找 {extension} 文件...")
    
    file_list = []
    try:
        # 排序以保证发送顺序
        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(extension)])
        if not file_list:
            print(f"Feeder 错误: 在 '{folder_path}' 中没有找到 {extension} 文件。")
            return
    except FileNotFoundError:
        print(f"Feeder 错误: 文件夹 '{folder_path}' 不存在。")
        return

    print(f"Feeder: 找到了 {len(file_list)} 个文件。准备开始发送...")

    while True:
        for filename in file_list:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "rb") as f:
                    # 读取文件的全部原始字节
                    raw_data = f.read()
                
                # yield 文件名和数据
                yield filename, raw_data
                
            except Exception as e:
                print(f"Feeder 警告: 读取文件 {filename} 失败: {e}")
        
        if not loop:
            break # 默认只发送一次
        else:
            print("Feeder: 已完成一轮发送，正在循环...")
            time.sleep(1) # 循环时稍作停顿
    
    print("Feeder: 所有文件发送完毕。")