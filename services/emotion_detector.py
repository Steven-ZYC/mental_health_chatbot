import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import json


class EmotionDetector:
    def __init__(self, model_path='models/best_model.pth',
                 labels_path='models/class_labels.json',
                 cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        # 加载类别标签
        with open(labels_path, 'r') as f:
            self.class_labels = list(json.load(f).keys())

        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.225])
        ])

        # 可视化参数
        self.COLOR_MAP = {
            'happy': (0, 255, 0),      # 绿色
            'neutral': (200, 200, 200),# 灰色
            'sad': (0, 0, 255),        # 红色
            'surprise': (255, 255, 0), # 青色
            'angry': (0, 0, 255),      # 红色
            'disgust': (0, 128, 0),    # 深绿色
            'fear': (255, 0, 255)      # 紫色
        }
        self.TEXT_SCALE = 0.8
        self.TEXT_THICKNESS = 1

        # 状态信息
        self.cap = None
        self.latest_emotions = []

    class EmotionResNet(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.base_model = models.resnet18(weights=None)
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

        def forward(self, x):
            return self.base_model(x)

    def _load_model(self, model_path):
        model = self.EmotionResNet(num_classes=len(self.class_labels)).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def predict_emotion(self, face_img):
        """预测单张人脸图像的表情"""
        image = Image.fromarray(face_img)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        return self.class_labels[preds[0]], probs[0].cpu().numpy()

    def process_frame(self, frame):
        """处理视频帧并返回带有标注信息的帧"""
        self.latest_emotions = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 处理每个检测到的人脸
        for i, (x, y, w, h) in enumerate(faces):
            face_img = gray[y:y+h, x:x+w]
            emotion_label, probs = self.predict_emotion(face_img)
            self.latest_emotions.append({
                'label': emotion_label,
                'probabilities': dict(zip(self.class_labels, probs)),
                'bbox': (x, y, w, h)
            })

            # 绘制人脸框和标签
            color = self.COLOR_MAP.get(emotion_label, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion_label}: {probs[self.class_labels.index(emotion_label)] * 100:.1f}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, self.TEXT_SCALE, color, self.TEXT_THICKNESS)

            # 为第一个检测到的人脸绘制概率条
            if i == 0:
                for idx, (cls, prob) in enumerate(zip(self.class_labels, probs)):
                    bar_width = int(prob * 100)
                    cv2.rectangle(frame, (10, 30 * idx + 10),
                                (10 + bar_width, 30 * idx + 25),
                                self.COLOR_MAP[cls], -1)
                    cv2.putText(frame, f"{cls}: {prob * 100:.1f}%",
                                (15, 30 * idx + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def start_camera(self, camera_index=0):
        """启动摄像头采集"""
        self.cap = cv2.VideoCapture(camera_index)
        return self.cap.isOpened()

    def get_frame(self):
        """获取一帧画面"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def run_as_library(self):
        """以独立程序模式运行"""
        if not self.start_camera():
            print("Error: Camera not accessible")
            return

        while True:
            ret, frame = self.get_frame()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Emotion Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()


if __name__ == '__main__':
    detector = EmotionDetector()
    detector.run_as_library()