import cv2
import numpy as np
import time
import random
import os

# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型和配置文件的路径
prototxt_path = os.path.join(script_dir, 'deploy.prototxt')
model_path = os.path.join(script_dir, 'mobilenet_iter_73000.caffemodel')

# 加载模型
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 类别列表
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 为每个类别分配随机颜色
colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in classes}

# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化计时器
start_time = time.time()
print_interval = 1  # 设置打印间隔时间（秒）

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理帧
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # 设置输入
    net.setInput(blob)
    detections = net.forward()

    # 处理检测结果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            color = colors[classes[idx]]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 仅在达到打印间隔时间时打印检测到的对象信息
            if time.time() - start_time >= print_interval:
                print(f"Detected {classes[idx]} with confidence {confidence:.2f}")
                start_time = time.time()

    # 显示结果帧
    cv2.imshow("Frame", frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
