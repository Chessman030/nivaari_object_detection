#!/usr/bin/env python3
"""
Test and Validation for Civic Issues Detection Model
Handles model testing, validation, and performance evaluation
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from model_loader import CivicIssuesModelLoader
from popup_display import show_detection_popup


class ModelTester:
    """Handles testing and validation of trained models"""
    
    def __init__(self):
        """Initialize the model tester"""
        self.model_loader = CivicIssuesModelLoader()
        
    def test_single_image(self, image_path, model_path=None, conf_threshold=0.25, show_popup=True):
        """
        Test the model on a single image
        
        Args:
            image_path (str): Path to test image
            model_path (str): Optional specific model path
            conf_threshold (float): Confidence threshold
            show_popup (bool): Whether to show popup with results
            
        Returns:
            dict: Test results
        """
        print(f"\n🔍 Testing single image: {os.path.basename(image_path)}")
        
        # Load model
        if not self.model_loader.load_model(model_path):
            return None
        
        # Run prediction
        start_time = time.time()
        results = self.model_loader.predict(image_path, conf_threshold)
        inference_time = time.time() - start_time
        
        if results is None:
            print("❌ Prediction failed")
            return None
        
        # Display results
        print(f"⚡ Inference time: {inference_time:.2f} seconds")
        print(f"🎯 Issues found: {results['total_issues']}")
        
        if results['detections']:
            print("📝 Detected issues:")
            for i, detection in enumerate(results['detections'], 1):
                print(f"   {i}. {detection['class_name']} - {detection['confidence']:.1%}")
        else:
            print("✅ No issues detected in this image")
        
        # Show popup if requested
        if show_popup:
            show_detection_popup(image_path, results['detections'], "Test Results")
        
        return {
            'image_path': image_path,
            'inference_time': inference_time,
            'results': results,
            'success': True
        }
    
    def test_directory(self, test_dir, model_path=None, conf_threshold=0.25, max_images=10):
        """
        Test the model on a directory of images
        
        Args:
            test_dir (str): Directory containing test images
            model_path (str): Optional specific model path
            conf_threshold (float): Confidence threshold
            max_images (int): Maximum number of images to test
            
        Returns:
            dict: Batch test results
        """
        print(f"\n📁 Testing directory: {test_dir}")
        
        if not os.path.exists(test_dir):
            print(f"❌ Directory not found: {test_dir}")
            return None
        
        # Get test images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for file in os.listdir(test_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(test_dir, file))
        
        if not image_files:
            print(f"❌ No image files found in {test_dir}")
            return None
        
        # Limit number of images
        image_files = image_files[:max_images]
        print(f"🖼️  Found {len(image_files)} images to test")
        
        # Load model
        if not self.model_loader.load_model(model_path):
            return None
        
        # Test each image
        test_results = []
        total_inference_time = 0
        total_detections = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Testing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            start_time = time.time()
            results = self.model_loader.predict(image_path, conf_threshold)
            inference_time = time.time() - start_time
            
            if results:
                total_inference_time += inference_time
                total_detections += results['total_issues']
                
                test_results.append({
                    'image_path': image_path,
                    'inference_time': inference_time,
                    'detections': results['total_issues'],
                    'details': results['detections']
                })
                
                print(f"   ⚡ Time: {inference_time:.2f}s | Issues: {results['total_issues']}")
            else:
                print(f"   ❌ Failed to process image")
        
        # Summary
        if test_results:
            avg_inference_time = total_inference_time / len(test_results)
            avg_detections = total_detections / len(test_results)
            
            print(f"\n📊 BATCH TEST SUMMARY")
            print(f"=" * 40)
            print(f"Images tested: {len(test_results)}")
            print(f"Total inference time: {total_inference_time:.2f}s")
            print(f"Average inference time: {avg_inference_time:.2f}s")
            print(f"Total detections: {total_detections}")
            print(f"Average detections per image: {avg_detections:.1f}")
            print(f"=" * 40)
            
            return {
                'num_images': len(test_results),
                'total_inference_time': total_inference_time,
                'avg_inference_time': avg_inference_time,
                'total_detections': total_detections,
                'avg_detections': avg_detections,
                'results': test_results,
                'success': True
            }
        
        return None
    
    def validate_model(self, model_path=None):
        """
        Run validation on the validation set using YOLO's built-in validation
        
        Args:
            model_path (str): Optional specific model path
            
        Returns:
            dict: Validation results
        """
        print("\n📊 Running model validation...")
        
        # Load model
        if not self.model_loader.load_model(model_path):
            return None
        
        try:
            # Run validation using YOLO's built-in validation
            print("🔄 Running validation on validation set...")
            validation_results = self.model_loader.model.val(data='data.yaml')
            
            # Extract key metrics
            metrics = validation_results.results_dict
            
            print(f"\n📈 VALIDATION RESULTS")
            print(f"=" * 40)
            
            if 'metrics/precision(B)' in metrics:
                print(f"Precision: {metrics['metrics/precision(B)']:.3f}")
            if 'metrics/recall(B)' in metrics:
                print(f"Recall: {metrics['metrics/recall(B)']:.3f}")
            if 'metrics/mAP50(B)' in metrics:
                print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.3f}")
            
            print(f"=" * 40)
            
            return {
                'metrics': metrics,
                'model_path': self.model_loader.model_path,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None
    
    def test_performance_benchmark(self, model_path=None, num_iterations=10):
        """
        Run performance benchmark using a test image
        
        Args:
            model_path (str): Optional specific model path
            num_iterations (int): Number of iterations for timing
            
        Returns:
            dict: Performance results
        """
        print(f"\n⚡ Running performance benchmark ({num_iterations} iterations)...")
        
        # Find a test image
        test_image = None
        test_dirs = ['test/images', 'valid/images', 'train/images']
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image = os.path.join(test_dir, file)
                        break
                if test_image:
                    break
        
        if not test_image:
            print("❌ No test image found for benchmarking")
            return None
        
        # Load model
        if not self.model_loader.load_model(model_path):
            return None
        
        print(f"🖼️  Using test image: {os.path.basename(test_image)}")
        
        # Warm up
        print("🔄 Warming up...")
        for _ in range(3):
            self.model_loader.predict(test_image, 0.25)
        
        # Run benchmark
        print(f"⏱️  Running {num_iterations} iterations...")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            results = self.model_loader.predict(test_image, 0.25)
            end_time = time.time()
            
            if results:
                times.append(end_time - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"   Completed {i + 1}/{num_iterations} iterations")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            fps = 1.0 / avg_time
            
            print(f"\n🚀 PERFORMANCE RESULTS")
            print(f"=" * 40)
            print(f"Average inference time: {avg_time:.3f}s")
            print(f"Minimum inference time: {min_time:.3f}s")
            print(f"Maximum inference time: {max_time:.3f}s")
            print(f"Frames per second: {fps:.1f} FPS")
            print(f"Device: {self.model_loader.device.upper()}")
            print(f"=" * 40)
            
            return {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'fps': fps,
                'device': self.model_loader.device,
                'iterations': len(times),
                'success': True
            }
        
        return None


# Convenience functions for easy access
def test_image(image_path, conf_threshold=0.25, show_popup=True):
    """Quick function to test a single image"""
    tester = ModelTester()
    return tester.test_single_image(image_path, conf_threshold=conf_threshold, show_popup=show_popup)


def validate_model():
    """Quick function to validate the model"""
    tester = ModelTester()
    return tester.validate_model()


def benchmark_performance():
    """Quick function to benchmark model performance"""
    tester = ModelTester()
    return tester.test_performance_benchmark()


if __name__ == "__main__":
    """Command line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and validate civic issues detection model")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--dir", type=str, help="Directory of test images")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-popup", action="store_true", help="Don't show popup results")
    
    args = parser.parse_args()
    
    tester = ModelTester()
    
    if args.image:
        tester.test_single_image(args.image, conf_threshold=args.conf, show_popup=not args.no_popup)
    elif args.dir:
        tester.test_directory(args.dir, conf_threshold=args.conf)
    elif args.validate:
        tester.validate_model()
    elif args.benchmark:
        tester.test_performance_benchmark()
    else:
        print("Please specify --image, --dir, --validate, or --benchmark")
        parser.print_help()