#!/usr/bin/env python3
"""
Image Popup Display for Civic Issues Detection
Handles displaying images with bounding boxes and predictions
"""

import cv2
import numpy as np
import os
from datetime import datetime


class DetectionPopup:
    """Handles image display with detection results"""
    
    def __init__(self):
        """Initialize the popup display"""
        self.colors = self._generate_colors()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.text_thickness = 1
    
    def _generate_colors(self):
        """Generate distinct colors for different classes"""
        colors = [
            (0, 255, 0),      # Green - Garbage
            (255, 0, 0),      # Blue - Pothole
            (0, 0, 255),      # Red - Road crack
            (255, 255, 0),    # Cyan - Graffiti
            (255, 0, 255),    # Magenta - Fallen Trees
            (0, 255, 255),    # Yellow - Dead Animal
            (128, 255, 0),    # Lime - Damaged Concrete
            (255, 128, 0),    # Orange - Damaged Electric Poles
            (128, 0, 255),    # Purple - Damaged Road Signs
        ]
        return colors
    
    def _get_color_for_class(self, class_id):
        """Get color for a specific class"""
        return self.colors[class_id % len(self.colors)]
    
    def _draw_text_with_background(self, img, text, position, color, background_color=(0, 0, 0)):
        """Draw text with background for better visibility"""
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.text_thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (x, y - text_height - baseline),
            (x + text_width, y + baseline),
            background_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img, text, (x, y), self.font, self.font_scale, color, self.text_thickness
        )
    
    def display_detections(self, image_path, detections, window_title="Civic Issues Detection Results"):
        """
        Display image with detection results in a popup window
        
        Args:
            image_path (str): Path to the image file
            detections (list): List of detection results
            window_title (str): Title for the popup window
            
        Returns:
            bool: True if successful, False if error
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Error: Could not load image from {image_path}")
                return False
            
            # Make a copy for drawing
            display_img = img.copy()
            
            # Draw detections
            for detection in detections:
                class_name = detection['class_name']
                class_id = detection['class_id']
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                x1, y1, x2, y2 = bbox
                color = self._get_color_for_class(class_id)
                
                # Draw bounding box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, self.thickness)
                
                # Prepare label text
                label = f"{class_name}: {confidence:.2%}"
                
                # Draw label with background
                self._draw_text_with_background(
                    display_img, label, (x1, y1 - 5), color, (0, 0, 0)
                )
            
            # Add summary information
            summary_text = f"Issues Found: {len(detections)}"
            img_height, img_width = display_img.shape[:2]
            
            # Draw summary at the top
            self._draw_text_with_background(
                display_img, summary_text, (10, 30), (255, 255, 255), (0, 0, 0)
            )
            
            # Calculate display size (max 1200x800 while maintaining aspect ratio)
            max_width, max_height = 1200, 800
            scale = min(max_width / img_width, max_height / img_height, 1.0)
            
            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))
            
            # Display the image
            cv2.imshow(window_title, display_img)
            
            print("🖼️  Image displayed in popup window")
            print("📋 Detection Summary:")
            print(f"   Total Issues: {len(detections)}")
            
            if detections:
                print("   Detected Issues:")
                for i, detection in enumerate(detections, 1):
                    print(f"      {i}. {detection['class_name']} ({detection['confidence']:.1%})")
            
            print("🔄 Controls:")
            print("   - Press any key to close the popup")
            print("   - Press 's' to save the annotated image")
            print("   - Press ESC to close")
            
            # Wait for key press
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Save annotated image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_result_{timestamp}.jpg"
                    cv2.imwrite(filename, display_img)
                    print(f"💾 Annotated image saved as: {filename}")
                    
                elif key == 27 or key != 255:  # ESC or any other key
                    break
            
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            cv2.destroyAllWindows()
            return False
    
    def display_no_detections(self, image_path, window_title="No Issues Detected"):
        """Display image when no detections are found"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Error: Could not load image from {image_path}")
                return False
            
            # Calculate display size
            img_height, img_width = img.shape[:2]
            max_width, max_height = 1200, 800
            scale = min(max_width / img_width, max_height / img_height, 1.0)
            
            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Add "No Issues Found" text
            self._draw_text_with_background(
                img, "✅ No Issues Detected", (10, 30), (0, 255, 0), (0, 0, 0)
            )
            
            cv2.imshow(window_title, img)
            print("🖼️  Image displayed - No issues detected")
            print("🔄 Press any key to close the popup")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            cv2.destroyAllWindows()
            return False
    
    def close_all_windows(self):
        """Close all OpenCV windows"""
        cv2.destroyAllWindows()


# Function for easy access from other modules
def show_detection_popup(image_path, detections, title="Detection Results"):
    """
    Convenient function to show detection popup
    
    Args:
        image_path (str): Path to image
        detections (list): Detection results
        title (str): Window title
    
    Returns:
        bool: Success status
    """
    popup = DetectionPopup()
    
    if detections and len(detections) > 0:
        return popup.display_detections(image_path, detections, title)
    else:
        return popup.display_no_detections(image_path, title)