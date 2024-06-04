import cv2
import dlib
import numpy as np
import os

# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型和配置文件的路径
predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
face_rec_model_path = os.path.join(script_dir, "dlib_face_recognition_resnet_model_v1.dat")

# 初始化dlib的面部检测器和面部识别模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

# 创建目录存储人脸数据
face_data_dir = os.path.join(script_dir, 'face_data')
if not os.path.exists(face_data_dir):
    os.makedirs(face_data_dir)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化已知人脸数据
known_faces = []
known_names = []

def register_face(name, face_descriptor):
    known_faces.append(face_descriptor)
    known_names.append(name)
    with open(os.path.join(face_data_dir, f'{name}.npy'), 'wb') as f:
        np.save(f, face_descriptor)

def load_registered_faces():
    for file in os.listdir(face_data_dir):
        if file.endswith('.npy'):
            name = os.path.splitext(file)[0]
            face_descriptor = np.load(os.path.join(face_data_dir, file))
            known_faces.append(face_descriptor)
            known_names.append(name)

# 加载已注册的人脸数据
load_registered_faces()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 获取面部特征点
        shape = predictor(gray, face)
        face_descriptor = face_rec.compute_face_descriptor(frame, shape)

        # 转换为numpy数组
        face_descriptor = np.array(face_descriptor)

        # 识别或注册新面孔
        matches = []
        for known_face in known_faces:
            distance = np.linalg.norm(known_face - face_descriptor)
            matches.append(distance)

        if matches and min(matches) < 0.6:
            name = known_names[np.argmin(matches)]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            name = f"Person_{len(known_faces) + 1}"
            register_face(name, face_descriptor)
            cv2.putText(frame, "New Face Registered", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
