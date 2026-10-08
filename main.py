#!/usr/bin/env python3
"""
Simplified Civic Issues Detection System
3 Options: Accuracy/Loss Testing, User Image Prediction, Exit
"""

import sys
import os


def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🏙️  CIVIC ISSUES DETECTION SYSTEM")
    print("="*60)
    print("AI-powered detection of civic problems in images")
    print("Classes: Potholes, Road Cracks, Graffiti, Garbage, and more!")
    print("="*60)


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import cv2
        import ultralytics
        from PIL import Image
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n🔧 Please install required packages:")
        print("   pip install -r requirements.txt")
        print("\n📋 Required packages:")
        print("   - torch")
        print("   - opencv-python")
        print("   - ultralytics")
        print("   - Pillow")
        return False


def show_menu():
    """Show main menu"""
    print("\n🚀 CIVIC ISSUES DETECTION SYSTEM")
    print("-" * 40)
    print("1. 📊 Test Accuracy & Loss (Test Folder)")
    print("2. 🖼️  Predict on User Image")
    print("3. 🚪 Exit")
    print("-" * 40)


def test_accuracy_loss():
    """Test accuracy and loss using images from test folder"""
    try:
        from test_validation import ModelTester
        tester = ModelTester()
        
        print("\n📊 Testing Accuracy & Loss on Test Dataset...")
        
        # First run validation to get accuracy metrics
        print("🔄 Running model validation for accuracy metrics...")
        validation_result = tester.validate_model()
        
        if validation_result and 'metrics' in validation_result:
            print("\n📈 ACCURACY & LOSS RESULTS")
            print("=" * 40)
            metrics = validation_result['metrics']
            
            if 'metrics/precision(B)' in metrics:
                print(f"Precision: {metrics['metrics/precision(B)']:.3f}")
            if 'metrics/recall(B)' in metrics:
                print(f"Recall: {metrics['metrics/recall(B)']:.3f}")
            if 'metrics/mAP50(B)' in metrics:
                print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.3f}")
            
            print("=" * 40)
        else:
            print("❌ Validation failed")
        
        # Also test on test folder images for detailed results
        test_folder = "./test/images"
        if os.path.exists(test_folder):
            print(f"\n🔄 Testing on test folder images...")
            result = tester.test_directory(test_folder, max_images=20)
            
            if result:
                print(f"\n✅ Test folder results:")
                print(f"   Images tested: {result['num_images']}")
                print(f"   Total detections: {result['total_detections']}")
                print(f"   Average inference time: {result['avg_inference_time']:.2f}s")
                print(f"   Average detections per image: {result['avg_detections']:.1f}")
            else:
                print("❌ Test folder analysis failed")
        else:
            print(f"❌ Test folder not found: {test_folder}")
            
    except Exception as e:
        print(f"❌ Testing error: {str(e)}")


def predict_user_image():
    """Predict on user input image with confidence like temp_model.py"""
    try:
        from model_loader import CivicIssuesModelLoader
        import cv2
        
        print("\n🖼️  User Image Prediction")
        
        # Get image path
        image_path = input("📁 Enter image path: ").strip().strip('"\'')
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Get confidence threshold
        try:
            conf_input = input("🎯 Confidence threshold (0.1-0.9) [0.25]: ").strip()
            conf_threshold = float(conf_input) if conf_input else 0.25
            conf_threshold = max(0.1, min(0.9, conf_threshold))
        except ValueError:
            conf_threshold = 0.25
        
        print(f"🔍 Analyzing image with confidence threshold: {conf_threshold}...")
        
        # Load model and predict
        model_loader = CivicIssuesModelLoader()
        if not model_loader.load_model():
            print("❌ Failed to load model")
            return
        
        results = model_loader.predict(image_path, conf_threshold)
        
        if results:
            print(f"\n✅ Detection completed!")
            print(f"🎯 Issues found: {results['total_issues']}")
            
            if results['detections']:
                print("\n📝 Detected Issues:")
                print("-" * 50)
                
                # Read the image for visualization (like temp_model.py)
                img = cv2.imread(image_path)
                
                for i, detection in enumerate(results['detections'], 1):
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    bbox = detection['bbox']
                    
                    print(f"{i:2d}. {class_name}")
                    print(f"     Confidence: {confidence:.2%}")
                    print(f"     Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                    
                    # Draw bounding box (like temp_model.py)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{class_name} {confidence:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    print()
                
                # Show image with detections (like temp_model.py)
                print("🖼️ Opening image window... (Press any key to close)")
                cv2.imshow("Civic Issues Detection Results", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print("\n✅ No issues detected in this image")
                # Show original image
                img = cv2.imread(image_path)
                cv2.putText(img, "No Issues Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("No Issues Detected", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            print("❌ Detection failed")
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        cv2.destroyAllWindows()  # Ensure windows are closed


def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    while True:
        try:
            show_menu()
            
            choice = input("\n👆 Select option (1-3): ").strip()
            
            if choice == '1':
                test_accuracy_loss()
            
            elif choice == '2':
                predict_user_image()
            
            elif choice == '3':
                print("\n👋 Thank you for using Civic Issues Detection System!")
                print("🏙️ Making cities better, one detection at a time!")
                break
            
            else:
                print("❌ Invalid option. Please choose 1-3.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🔄 Returning to main menu...")


if __name__ == "__main__":
    main()