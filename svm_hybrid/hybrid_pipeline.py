import os
import cv2
import torch
from ultralytics import YOLO
from svm_classifier import SVMClassifier

class HybridDetector:
    def __init__(self, yolo_weights_path, svm_model_path="svm_model.pkl"):
        """Initialize YOLO for localization and SVM for classification."""
        print(f"Loading YOLO model from {yolo_weights_path}...")
        self.yolo_model = YOLO(yolo_weights_path)
        
        self.svm = SVMClassifier(svm_model_path)
        
        # Check if SVM is trained
        if not self.svm.load_model():
            print("WARNING: SVM model not found! You must train the SVM first.")
            self.svm_ready = False
        else:
            self.svm_ready = True

    def detect_and_classify(self, image_path, conf_threshold=0.25):
        """
        1. Run YOLO to find bounding boxes
        2. Crop the image at those boxes
        3. Run SVM to classify the crop
        """
        if not self.svm_ready:
            raise RuntimeError("SVM model is not trained. Please train it before running inference.")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # 1. YOLO for Localization
        results = self.yolo_model(img, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    yolo_conf = float(box.conf[0].cpu().numpy())
                    
                    # Prevent out-of-bounds crops
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # 2. Crop the region of interest
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    # 3. SVM for Classification
                    predicted_class, svm_conf = self.svm.predict(crop)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': predicted_class,
                        'svm_confidence': svm_conf,
                        'yolo_confidence': yolo_conf
                    })
                    
        return img, detections

    def draw_results(self, img, detections):
        """Draw bounding boxes and labels on the image."""
        result_img = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} ({det['svm_confidence']:.2f})"
            
            # Draw rectangle
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return result_img
