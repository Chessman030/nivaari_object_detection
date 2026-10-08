#!/usr/bin/env python3
"""
Command Line Interface for Civic Issues Detection
Simple script for quick image detection via command line
"""

import sys
import os
import argparse
from model_loader import CivicIssuesModelLoader
from popup_display import show_detection_popup
from test_validation import ModelTester


def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🏙️  CIVIC ISSUES DETECTION SYSTEM")
    print("="*60)
    print("AI-powered detection of civic problems in images")
    print("Classes: Potholes, Road Cracks, Graffiti, Garbage, and more!")
    print("="*60)


def detect_image_cli(image_path, conf_threshold=0.25, model_path=None, show_popup=True):
    """Detect issues in an image via CLI"""
    print(f"\n🔍 Detecting issues in: {os.path.basename(image_path)}")
    
    # Initialize model loader
    model_loader = CivicIssuesModelLoader()
    
    # Load model
    print("🔄 Loading model...")
    if not model_loader.load_model(model_path):
        print("❌ Failed to load model")
        return False
    
    # Run detection
    results = model_loader.predict(image_path, conf_threshold)
    
    if results is None:
        print("❌ Detection failed")
        return False
    
    # Display results
    print(f"\n✅ Detection completed!")
    print(f"🎯 Issues found: {results['total_issues']}")
    print(f"📊 Confidence threshold: {conf_threshold}")
    
    if results['detections']:
        print("\n📝 Detected Issues:")
        print("-" * 50)
        for i, detection in enumerate(results['detections'], 1):
            conf_pct = detection['confidence'] * 100
            bbox = detection['bbox']
            print(f"{i:2d}. {detection['class_name']}")
            print(f"     Confidence: {conf_pct:.1f}%")
            print(f"     Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
            print()
    else:
        print("\n✅ No issues detected in this image")
    
    # Show popup if requested
    if show_popup:
        print("🖼️  Opening visual results popup...")
        show_detection_popup(image_path, results['detections'], "CLI Detection Results")
    
    return True


def interactive_mode():
    """Run interactive mode for multiple image detection"""
    print_banner()
    print("\n🚀 Interactive Mode")
    print("Enter image paths to detect issues (type 'quit' to exit)")
    
    model_loader = CivicIssuesModelLoader()
    
    while True:
        try:
            image_path = input("\n📁 Enter image path (or 'quit'): ").strip().strip('"\'')
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not image_path:
                continue
                
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            # Get confidence threshold
            try:
                conf_input = input("🎯 Confidence threshold (0.1-0.9) [0.25]: ").strip()
                conf_threshold = float(conf_input) if conf_input else 0.25
                conf_threshold = max(0.1, min(0.9, conf_threshold))
            except ValueError:
                conf_threshold = 0.25
            
            # Run detection
            detect_image_cli(image_path, conf_threshold, show_popup=True)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Civic Issues Detection - Command Line Interface")
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    parser.add_argument("--conf", "-c", type=float, default=0.25, 
                       help="Confidence threshold (0.1-0.9, default: 0.25)")
    parser.add_argument("--model", "-m", type=str, help="Path to specific model weights")
    parser.add_argument("--no-popup", action="store_true", help="Don't show popup results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test", type=str, help="Test model on image")
    parser.add_argument("--validate", action="store_true", help="Run model validation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        print_banner()
        model_loader = CivicIssuesModelLoader()
        models = model_loader.list_available_models()
        
        if models:
            print(f"\n📊 Available Models ({len(models)} found):")
            print("-" * 50)
            for i, model in enumerate(models, 1):
                status = " (Latest)" if i == 1 else ""
                print(f"{i}. {model['name']}{status}")
                print(f"   Path: {model['path']}")
                print(f"   Modified: {model['modified']}")
                print()
        else:
            print("\n⚠️  No trained models found")
            print("Please train a model first using temp_model.py")
        
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Test operations
    if args.test or args.validate or args.benchmark:
        print_banner()
        tester = ModelTester()
        
        if args.test:
            print(f"\n🧪 Testing model on: {args.test}")
            tester.test_single_image(args.test, conf_threshold=args.conf, show_popup=not args.no_popup)
        
        elif args.validate:
            print("\n📊 Running model validation...")
            tester.validate_model()
        
        elif args.benchmark:
            print("\n⚡ Running performance benchmark...")
            tester.test_performance_benchmark()
        
        return
    
    # Single image detection
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image file not found: {args.image}")
            sys.exit(1)
        
        print_banner()
        success = detect_image_cli(
            args.image, 
            conf_threshold=args.conf,
            model_path=args.model,
            show_popup=not args.no_popup
        )
        
        if not success:
            sys.exit(1)
    
    else:
        # Default to interactive mode if no arguments
        interactive_mode()


if __name__ == "__main__":
    main()