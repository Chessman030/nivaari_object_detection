import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import joblib
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, model_path="svm_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.classes = [
            'Damaged_Concrete_Structures', 'Damaged_Electric_Poles', 
            'Damaged_Road_Signs', 'Dead_Animal_Pollution', 'Fallen_Trees', 
            'Garbage', 'Graffiti', 'pothole', 'road_crack'
        ]
        
        # ----- Setup CNN (ResNet) Feature Extractor -----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet18
        print("Loading ResNet for feature extraction...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer to get the raw features (512 dimensions)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval() # Set to evaluation mode
        
        # Standard preprocessing for ResNet models
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image):
        """Extract CNN (ResNet) features from a given OpenCV image crop."""
        # Convert OpenCV BGR to PIL RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Apply transforms and add batch dimension
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features without computing gradients
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            
        # Flatten the features to a 1D array (from shape [1, 512, 1, 1] to [512])
        features_np = features.cpu().numpy().flatten()
        return features_np

    def train(self, X_train, y_train):
        """Train the SVM model on the CNN features."""
        print("Training SVM model... This might take a while.")
        self.model = SVC(kernel='linear', probability=True, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained SVM model."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Loaded SVM model from {self.model_path}")
            return True
        return False

    def predict(self, image_crop):
        """Predict the class of an image crop using CNN features + SVM."""
        if self.model is None:
            raise ValueError("SVM model is not loaded or trained yet.")
        
        # Extract CNN features
        features = self.extract_features(image_crop)
        features = features.reshape(1, -1)
        
        # Pass features to SVM
        pred_idx = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0]
        confidence = max(prob)
        
        return self.classes[pred_idx], confidence
