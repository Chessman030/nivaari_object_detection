import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from svm_classifier import SVMClassifier

def map_folder_to_class(folder_name):
    folder_name = folder_name.lower()
    if 'pothole' in folder_name:
        return 'pothole'
    elif 'damaged road' in folder_name:
        return 'road_crack'
    elif 'road sign' in folder_name:
        return 'Damaged_Road_Signs'
    elif 'garbage' in folder_name or 'littering' in folder_name:
        return 'Garbage'
    elif 'vandalism' in folder_name or 'graffiti' in folder_name:
        return 'Graffiti'
    elif 'concrete' in folder_name:
        return 'Damaged_Concrete_Structures'
    elif 'electric' in folder_name:
        return 'Damaged_Electric_Poles'
    elif 'animal' in folder_name:
        return 'Dead_Animal_Pollution'
    elif 'tree' in folder_name:
        return 'Fallen_Trees'
    return None

def collect_crops(data_dir, svm_clf, yolo_model_path):
    print("Loading YOLO model to detect bounding boxes in the training images...")
    yolo = YOLO(yolo_model_path)
    
    crops = []
    labels = []
    
    # Traverse the directory structure
    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root)
        class_name = map_folder_to_class(folder_name)
        
        if not class_name:
            continue
            
        try:
            class_idx = svm_clf.classes.index(class_name)
        except ValueError:
            continue
            
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
            
        print(f"Processing '{folder_name}' mapped to '{class_name}' ({len(image_files)} images)")
        
        # We process a subset of images to speed up training, or all of them.
        for img_name in tqdm(image_files, desc=f"Extracting {class_name}"):
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Use YOLO to find the object since we don't have bounding box labels
            results = yolo(img, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Only take boxes with a decent confidence to avoid false crops
                        if float(box.conf[0]) < 0.25:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Fix bounds
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                            
                        # Extract Deep Features using ResNet
                        features = svm_clf.extract_features(crop)
                        
                        crops.append(features)
                        labels.append(class_idx)
                        
                        # Just take the first valid bounding box per image to avoid bias
                        break
                        
    return crops, labels

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    yolo_weights = os.path.join(base_dir, "weights", "best.pt")
    
    svm = SVMClassifier("svm_model.pkl")
    
    print("--- Nivaari SVM Training Setup (ResNet Features) ---")
    print(f"Looking for data in: {data_dir}")
    X_train, y_train = collect_crops(data_dir, svm, yolo_weights)
    
    if len(X_train) > 0:
        print(f"Extracted {len(X_train)} object crops.")
        print("2. Training SVM Model on Deep Features...")
        svm.train(X_train, y_train)
        print("🎉 Training Complete! You can now run `streamlit run app.py`.")
    else:
        print(f"Error: No training data found or no images could be processed in {data_dir}.")
