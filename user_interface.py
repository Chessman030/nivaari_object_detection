#!/usr/bin/env python3
"""
User Interface for Civic Issues Detection
Handles user interaction for image selection and display of results
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from model_loader import CivicIssuesModelLoader
from popup_display import show_detection_popup
from test_validation import ModelTester


class CivicIssuesGUI:
    """Main GUI application for civic issues detection"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = tk.Tk()
        self.root.title("🏙️ Civic Issues Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.model_loader = CivicIssuesModelLoader()
        self.tester = ModelTester()
        self.selected_image = None
        self.current_results = None
        
        # Setup GUI
        self.setup_gui()
        self.check_models()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🏙️ Civic Issues Detection System",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        # Model selection frame
        model_frame = tk.LabelFrame(
            self.root,
            text="📊 Model Selection",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        model_frame.pack(fill='x', padx=10, pady=5)
        
        # Model dropdown
        tk.Label(model_frame, text="Select Model:", bg='#f0f0f0').pack(anchor='w')
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state='readonly',
            width=50
        )
        self.model_combo.pack(fill='x', pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        # Refresh models button
        refresh_btn = tk.Button(
            model_frame,
            text="🔄 Refresh Models",
            command=self.refresh_models,
            bg='#3498db',
            fg='white',
            font=('Arial', 10)
        )
        refresh_btn.pack(side='right', padx=5)
        
        # Image selection frame
        image_frame = tk.LabelFrame(
            self.root,
            text="🖼️ Image Selection",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        image_frame.pack(fill='x', padx=10, pady=5)
        
        # Buttons for image selection
        btn_frame = tk.Frame(image_frame, bg='#f0f0f0')
        btn_frame.pack(fill='x', pady=5)
        
        select_btn = tk.Button(
            btn_frame,
            text="📁 Select Image",
            command=self.select_image,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            height=2,
            width=15
        )
        select_btn.pack(side='left', padx=5)
        
        detect_btn = tk.Button(
            btn_frame,
            text="🔍 Detect Issues",
            command=self.detect_issues,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 11, 'bold'),
            height=2,
            width=15
        )
        detect_btn.pack(side='left', padx=5)
        
        # Settings frame
        settings_frame = tk.LabelFrame(
            self.root,
            text="⚙️ Settings",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        conf_frame.pack(fill='x', pady=5)
        
        tk.Label(conf_frame, text="Confidence Threshold:", bg='#f0f0f0').pack(side='left')
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_scale = tk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient='horizontal',
            variable=self.conf_var,
            bg='#f0f0f0'
        )
        self.conf_scale.pack(side='right', fill='x', expand=True, padx=10)
        
        # Status frame
        status_frame = tk.LabelFrame(
            self.root,
            text="📊 Status",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Status text
        self.status_text = tk.Text(
            status_frame,
            height=15,
            bg='white',
            font=('Consolas', 10)
        )
        
        scrollbar = tk.Scrollbar(status_frame)
        scrollbar.pack(side='right', fill='y')
        self.status_text.pack(fill='both', expand=True)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # Bottom buttons frame
        bottom_frame = tk.Frame(self.root, bg='#f0f0f0')
        bottom_frame.pack(fill='x', padx=10, pady=5)
        
        # Test and validation buttons
        test_btn = tk.Button(
            bottom_frame,
            text="🧪 Test Model",
            command=self.open_test_window,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10)
        )
        test_btn.pack(side='left', padx=5)
        
        validate_btn = tk.Button(
            bottom_frame,
            text="📊 Validate Model",
            command=self.validate_model,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10)
        )
        validate_btn.pack(side='left', padx=5)
        
        about_btn = tk.Button(
            bottom_frame,
            text="❓ About",
            command=self.show_about,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10)
        )
        about_btn.pack(side='right', padx=5)
        
        clear_btn = tk.Button(
            bottom_frame,
            text="🗑️ Clear",
            command=self.clear_status,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10)
        )
        clear_btn.pack(side='right', padx=5)
        
        self.log_message("🚀 Civic Issues Detection System initialized")
        
    def check_models(self):
        """Check for available trained models"""
        self.refresh_models()
        
    def refresh_models(self):
        """Refresh the list of available models"""
        models = self.model_loader.list_available_models()
        
        self.model_combo['values'] = []
        
        if models:
            model_names = [f"{model['name']} (Latest)" if i == 0 else model['name'] 
                          for i, model in enumerate(models)]
            self.model_combo['values'] = model_names
            self.model_combo.set(model_names[0])  # Select the latest model
            self.log_message(f"📊 Found {len(models)} trained models")
        else:
            self.model_combo['values'] = ["No trained models found"]
            self.model_combo.set("No trained models found")
            self.log_message("⚠️ No trained models found. Please train a model first.")
    
    def on_model_selected(self, event=None):
        """Handle model selection"""
        selected = self.model_var.get()
        if selected and "No trained models" not in selected:
            self.log_message(f"🎯 Selected model: {selected}")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image for civic issues detection",
            filetypes=file_types
        )
        
        if file_path:
            self.selected_image = file_path
            self.log_message(f"📁 Selected image: {os.path.basename(file_path)}")
            self.log_message(f"   Path: {file_path}")
        
    def detect_issues(self):
        """Run detection on the selected image"""
        if not self.selected_image:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if "No trained models" in self.model_var.get():
            messagebox.showerror("No Model", "No trained models available. Please train a model first!")
            return
        
        try:
            self.log_message("🔄 Loading model...")
            self.root.config(cursor="wait")
            self.root.update()
            
            # Load model
            if not self.model_loader.load_model():
                self.log_message("❌ Failed to load model")
                return
            
            self.log_message("🔍 Running detection...")
            
            # Get confidence threshold
            conf_threshold = self.conf_var.get()
            
            # Run prediction
            results = self.model_loader.predict(self.selected_image, conf_threshold)
            
            if results:
                self.current_results = results
                
                # Log results
                self.log_message(f"✅ Detection completed!")
                self.log_message(f"   Issues found: {results['total_issues']}")
                self.log_message(f"   Confidence threshold: {conf_threshold}")
                
                if results['detections']:
                    self.log_message("   Detected issues:")
                    for i, detection in enumerate(results['detections'], 1):
                        conf_pct = detection['confidence'] * 100
                        self.log_message(f"      {i}. {detection['class_name']} ({conf_pct:.1f}%)")
                
                # Show popup with results
                self.log_message("🖼️ Opening detection results popup...")
                show_detection_popup(
                    self.selected_image, 
                    results['detections'], 
                    "🏙️ Civic Issues Detection Results"
                )
                
            else:
                self.log_message("❌ Detection failed")
                
        except Exception as e:
            self.log_message(f"❌ Error during detection: {str(e)}")
            messagebox.showerror("Detection Error", f"An error occurred: {str(e)}")
        
        finally:
            self.root.config(cursor="")
    
    def open_test_window(self):
        """Open test options window"""
        test_window = tk.Toplevel(self.root)
        test_window.title("🧪 Model Testing")
        test_window.geometry("400x300")
        test_window.configure(bg='#f0f0f0')
        
        # Title
        tk.Label(
            test_window,
            text="🧪 Model Testing Options",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=20)
        
        # Test single image
        tk.Button(
            test_window,
            text="📸 Test Single Image",
            command=lambda: self.test_single_image(test_window),
            bg='#3498db',
            fg='white',
            font=('Arial', 11),
            width=20,
            height=2
        ).pack(pady=5)
        
        # Test directory
        tk.Button(
            test_window,
            text="📁 Test Directory",
            command=lambda: self.test_directory(test_window),
            bg='#e74c3c',
            fg='white',
            font=('Arial', 11),
            width=20,
            height=2
        ).pack(pady=5)
        
        # Performance benchmark
        tk.Button(
            test_window,
            text="⚡ Performance Benchmark",
            command=lambda: self.benchmark_performance(test_window),
            bg='#f39c12',
            fg='white',
            font=('Arial', 11),
            width=20,
            height=2
        ).pack(pady=5)
        
        # Close button
        tk.Button(
            test_window,
            text="❌ Close",
            command=test_window.destroy,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 11),
            width=20
        ).pack(pady=20)
    
    def test_single_image(self, parent):
        """Test model on a single image"""
        file_path = filedialog.askopenfilename(
            parent=parent,
            title="Select test image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        
        if file_path:
            parent.destroy()
            self.log_message(f"🧪 Testing single image: {os.path.basename(file_path)}")
            
            try:
                conf_threshold = self.conf_var.get()
                result = self.tester.test_single_image(
                    file_path, 
                    conf_threshold=conf_threshold,
                    show_popup=True
                )
                
                if result:
                    self.log_message(f"✅ Test completed in {result['inference_time']:.2f}s")
                else:
                    self.log_message("❌ Test failed")
                    
            except Exception as e:
                self.log_message(f"❌ Test error: {str(e)}")
    
    def test_directory(self, parent):
        """Test model on a directory of images"""
        dir_path = filedialog.askdirectory(
            parent=parent,
            title="Select directory with test images"
        )
        
        if dir_path:
            parent.destroy()
            self.log_message(f"📁 Testing directory: {dir_path}")
            
            try:
                conf_threshold = self.conf_var.get()
                result = self.tester.test_directory(
                    dir_path,
                    conf_threshold=conf_threshold,
                    max_images=20
                )
                
                if result:
                    self.log_message(f"✅ Batch test completed!")
                    self.log_message(f"   Images tested: {result['num_images']}")
                    self.log_message(f"   Average time: {result['avg_inference_time']:.2f}s")
                    self.log_message(f"   Total detections: {result['total_detections']}")
                else:
                    self.log_message("❌ Batch test failed")
                    
            except Exception as e:
                self.log_message(f"❌ Batch test error: {str(e)}")
    
    def benchmark_performance(self, parent):
        """Run performance benchmark"""
        parent.destroy()
        self.log_message("⚡ Running performance benchmark...")
        
        try:
            result = self.tester.test_performance_benchmark()
            
            if result:
                self.log_message(f"🚀 Benchmark completed!")
                self.log_message(f"   Average time: {result['avg_time']:.3f}s")
                self.log_message(f"   FPS: {result['fps']:.1f}")
                self.log_message(f"   Device: {result['device'].upper()}")
            else:
                self.log_message("❌ Benchmark failed")
                
        except Exception as e:
            self.log_message(f"❌ Benchmark error: {str(e)}")
    
    def validate_model(self):
        """Run model validation"""
        if "No trained models" in self.model_var.get():
            messagebox.showerror("No Model", "No trained models available. Please train a model first!")
            return
        
        self.log_message("📊 Running model validation...")
        
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            result = self.tester.validate_model()
            
            if result and 'metrics' in result:
                self.log_message("✅ Validation completed!")
                metrics = result['metrics']
                
                if 'metrics/precision(B)' in metrics:
                    self.log_message(f"   Precision: {metrics['metrics/precision(B)']:.3f}")
                if 'metrics/recall(B)' in metrics:
                    self.log_message(f"   Recall: {metrics['metrics/recall(B)']:.3f}")
                if 'metrics/mAP50(B)' in metrics:
                    self.log_message(f"   mAP@0.5: {metrics['metrics/mAP50(B)']:.3f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    self.log_message(f"   mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.3f}")
            else:
                self.log_message("❌ Validation failed")
                
        except Exception as e:
            self.log_message(f"❌ Validation error: {str(e)}")
        finally:
            self.root.config(cursor="")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
🏙️ Civic Issues Detection System

A YOLOv11-based system for detecting civic problems in images:
• Potholes
• Road cracks  
• Graffiti
• Garbage
• Damaged infrastructure
• And more!

Features:
✓ Easy image selection
✓ Real-time detection
✓ Visual results with bounding boxes
✓ Model testing and validation
✓ Performance benchmarking

Built with PyTorch and Ultralytics YOLO.
        """
        messagebox.showinfo("About", about_text)
    
    def clear_status(self):
        """Clear the status text"""
        self.status_text.delete(1.0, tk.END)
        self.log_message("🗑️ Status cleared")
    
    def log_message(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the application"""
    app = CivicIssuesGUI()
    app.run()


if __name__ == "__main__":
    main()