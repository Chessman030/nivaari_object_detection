#!/usr/bin/env python3
#model stored at C:\Users\Raghav\runs\detect\train
"""
Model Loader for Civic Issues Detection
Handles model loading, downloading, and inference operations
"""

import os
import torch
from ultralytics import YOLO
import yaml


# Base directory of the project (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CivicIssuesModelLoader:
    """Handles loading and managing the trained YOLO model"""
    
    def __init__(self):
        """Initialize the model loader"""
        self.model = None
        self.model_path = None
        self.class_names = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from data.yaml"""
        try:
            with open('data.yaml', 'r') as file:
                data = yaml.safe_load(file)
                self.class_names = data.get('names', [])
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
            self.class_names = [
                'Damaged_Concrete_Structures', 'Damaged_Electric_Poles', 
                'Damaged_Road_Signs', 'Dead_Animal_Pollution', 'Fallen_Trees', 
                'Garbage', 'Graffiti', 'pothole', 'road_crack'
            ]
    
    def find_latest_model(self):
        """Find the trained model at the specified location"""
        # Your specific model path
        model_path = os.path.join(BASE_DIR,"weights", "best.pt")
        
        if os.path.exists(model_path):
            return model_path
        
        # Fallback to local runs directory if specific path doesn't exist
        runs_path = "runs/train"
        if os.path.exists(runs_path):
            for item in os.listdir(runs_path):
                item_path = os.path.join(runs_path, item)
                if os.path.isdir(item_path):
                    fallback_path = os.path.join(item_path, "weights", "best.pt")
                    if os.path.exists(fallback_path):
                        return fallback_path
        
        return None
    
    def list_available_models(self):
        """List all available trained models"""
        runs_path = "runs/train"
        models = []
        
        if not os.path.exists(runs_path):
            return models
            
        for item in os.listdir(runs_path):
            item_path = os.path.join(runs_path, item)
            if os.path.isdir(item_path):
                model_path = os.path.join(item_path, "weights", "best.pt")
                if os.path.exists(model_path):
                    models.append({
                        'name': item,
                        'path': model_path,
                        'modified': os.path.getmtime(model_path)
                    })
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)
        return models
    
    def load_model(self, model_path=None):
        """Load the YOLO model"""
        try:
            if model_path is None:
                model_path = self.find_latest_model()
                
            if model_path is None:
                raise ValueError("No trained model found. Please train a model first.")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                print("Using CPU for inference")
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, image_path, conf_threshold=0.25, save_results=False):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to the image
            conf_threshold (float): Confidence threshold
            save_results (bool): Whether to save annotated results
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Run prediction
            results = self.model(
                image_path,
                conf=conf_threshold,
                save=save_results,
                device=self.device
            )
            
            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                        
                        detections.append({
                            'class_name': class_name,
                            'class_id': class_id,
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        })
            
            return {
                'image_path': image_path,
                'detections': detections,
                'total_issues': len(detections),
                'model_used': self.model_path,
                'confidence_threshold': conf_threshold
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return None
            
        return {
            'model_path': self.model_path,
            'device': self.device,
            'classes': self.class_names,
            'num_classes': len(self.class_names)
        }