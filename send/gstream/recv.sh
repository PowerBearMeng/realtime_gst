#!/bin/bash

# CoDel 配置 - 为激光雷达优化
# target=100ms: 目标排队延迟 100ms（约 1 帧的时间）
# interval=1000ms: 1秒评估窗口（10帧的时间）

# cd  /home/mfh/driving/my_trans/gstream

mm-link ~/driving/data/trace_data/1106/trace/1_1.trace \
        ~/driving/data/trace_data/1106/trace/1_1.trace \
        --downlink-queue=codel \
        --downlink-queue-args="target=100,interval=100,packets=10000" \
        -- bash -c "cd ~/driving/my_trans/gstream && /bin/python3 main_receive.py"

# mm-link ~/driving/data/trace_data/1106/trace/static.trace \
#         ~/driving/data/trace_data/1106/trace/static.trace \
#         -- bash -c "cd ~/driving/my_trans/gstream && /bin/python3 main_receive.py"

