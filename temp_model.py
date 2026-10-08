import os
import cv2
from inference_sdk import InferenceHTTPClient

# Initialize the official Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="LPnxdUh8RdGPUkbQBSop"  # Your API Key
)

def main():
    print("🚀 Civic Issues Detection System (Visual Edition)")
    print("==================================================")
    
    while True:
        print("\nOptions:")
        print("1. Upload custom image")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1 or 3): ").strip()
        
        if choice == '1':
            image_path = input("📁 Enter path to your image: ").strip().replace('"', '')
            
            if not os.path.exists(image_path):
                print("❌ Error: Image not found!")
                continue
                
            print(f"🔍 Analyzing image with SDK: {image_path}...")
            
            try:
                # The Magic SDK Command
                result = CLIENT.infer(image_path, model_id="civic-issues-lroee/3")
                predictions = result.get('predictions', [])
                print(f"\n✅ Success! Found {len(predictions)} issues.")
                
                # --- VISUALIZATION MAGIC STARTS HERE ---
                # Read the image using OpenCV
                img = cv2.imread(image_path)
                
                for i, detection in enumerate(predictions, 1):
                    class_name = detection['class']
                    confidence = detection['confidence']
                    print(f"  {i}. {class_name} (Confidence: {confidence:.2f})")
                    
                    # Roboflow gives us the center coordinates, width, and height
                    x_center = detection['x']
                    y_center = detection['y']
                    width = detection['width']
                    height = detection['height']
                    
                    # Calculate the top-left and bottom-right corners for OpenCV
                    x1 = int(x_center - (width / 2))
                    y1 = int(y_center - (height / 2))
                    x2 = int(x_center + (width / 2))
                    y2 = int(y_center + (height / 2))
                    
                    # Draw the bounding box (Green color)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw a solid background for the text so it's readable
                    label = f"{class_name} {confidence:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
                    
                    # Put the text on top
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                # Pop open a window to show the final image!
                print("🖼️ Opening image window... (Press '0' on your keyboard to close the pop-up)")
                cv2.imshow("Nivaari Detection Results", img)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()
                # ---------------------------------------
                    
            except Exception as e:
                print(f"❌ API Error: {e}")
                
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()