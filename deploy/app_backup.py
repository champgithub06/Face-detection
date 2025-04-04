from flask import Flask, render_template, request, Response, url_for
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from PIL import Image

# ---------------------------
# Flask App Initialization
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load the Image Upload Model (ResNet-based)
# ---------------------------
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from torch.nn import Dropout

def create_improved_model(model_name='resnet50', num_classes=3, pretrained=False):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError("Only 'resnet50' is demonstrated here. Update as needed.")
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# Load the ResNet-based model from state_dict
model_path = "saved_models/model_resnet50_state_dict.pth"  # Adjust if needed
upload_model = create_improved_model(model_name='resnet50', num_classes=3, pretrained=False)
with torch.serialization.safe_globals([
    ResNet, Bottleneck, Conv2d, BatchNorm2d, BatchNorm1d, ReLU,
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Linear, Dropout
]):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
upload_model.load_state_dict(state_dict, strict=False)
upload_model.eval()

# Transformation used for image upload detection
test_transform = transforms.Compose([
    transforms.Resize((226, 226)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# ---------------------------
# Load DNN Face Detector (for image upload detection)
# ---------------------------
face_proto = "deploy.prototxt"
face_model = "res10_300x300_ssd_iter_140000.caffemodel"
if not os.path.isfile(face_proto) or not os.path.isfile(face_model):
    print("Error: DNN face detector files not found!")
    exit(1)
dnn_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

# ---------------------------
# Import or Define the FaceMaskDetector Class for Real-Time Detection
# ---------------------------
# (Assuming the FaceMaskDetector class code you provided is available here.)
# For this example, we assume it is defined in a module named 'detector_module'
# from detector_module import FaceMaskDetector
#
# If not, paste the FaceMaskDetector class code here.
#
# For demonstration, we'll assume the class is available as below:

class FaceMaskDetector:
    def __init__(self, model_dir="saved_models", yolo_model_path="yolo_v8.pt", model_choice="ensemble"):
        self.model_dir = model_dir
        self.yolo_model_path = yolo_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_choice = model_choice.lower().strip()
        # For simplicity in this example, we'll load only the resnet50 model.
        self.resnet50_model = self._load_model(f"{model_dir}/model_resnet50_state_dict.pth", model_type='resnet50')
        # Download DNN face detector if needed
        self.face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        print("FaceMaskDetector initialized for real-time detection.")

    def _create_model(self, model_name='resnet50', num_classes=3, pretrained=True):
        if model_name == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        return model

    def _load_model(self, model_path, model_type='resnet50'):
        model = self._create_model(model_name=model_type, pretrained=False, num_classes=3)
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"{model_type} loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        model.eval()
        model.to(self.device)
        return model

    def predict_mask(self, face_image):
        transform = transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.resnet50_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        prediction_idx = predicted.item()
        probability = probabilities[0][prediction_idx].item()
        predicted_class = class_names[prediction_idx]
        return predicted_class, probability

    def _process_frame(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue
                try:
                    face_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    continue
                predicted_class, prob = self.predict_mask(face_image)
                if predicted_class == 'with_mask':
                    color = (0, 255, 0)
                elif predicted_class == 'without_mask':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                label = f"{predicted_class}: {prob:.2f}"
                y_text = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

# Initialize the detector instance for realtime prediction
detector = FaceMaskDetector(model_dir="saved_models", yolo_model_path="yolo_v8.pt", model_choice="ensemble")

# ---------------------------
# Routes for Image Upload Detection
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return render_template('index.html', prediction="No file uploaded")
        file = request.files['imagefile']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")
        image_dir = "images"
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, file.filename)
        file.save(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return render_template('index.html', prediction="Error loading image")
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        face_count = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                face_roi = img[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                face_tensor = test_transform(face_pil).unsqueeze(0)
                with torch.no_grad():
                    outputs = upload_model(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                pred_idx = predicted.item()
                predicted_class = class_names[pred_idx]
                probability = probabilities[0][pred_idx].item()
                if predicted_class == 'with_mask':
                    color = (0, 255, 0)
                elif predicted_class == 'without_mask':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                label = f"{predicted_class}: {probability:.2f}"
                y_text = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img, label, (startX, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if face_count == 0:
            cv2.putText(img, "No faces detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)
        annotated_path = os.path.join(static_dir, "annotated.png")
        cv2.imwrite(annotated_path, img)
        prediction_text = f"Detected {face_count} face(s)" if face_count > 0 else "No faces detected"
        return render_template('index.html', prediction=prediction_text, annotated_image=url_for('static', filename='annotated.png'))
    return render_template('index.html')

# ---------------------------
# Routes for Real-Time Webcam Detection
# ---------------------------
@app.route('/camera')
def camera():
    return render_template('camera.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame using FaceMaskDetector's method
        processed_frame = detector._process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# Main Entry
# ---------------------------
if __name__ == '__main__':
    app.run(port=3000, debug=True)
