# 🏙️ Civic Issues Detection System

An advanced AI-powered system for detecting and classifying civic issues in images using YOLOv11. Features an intuitive GUI interface, command-line tools, and comprehensive testing capabilities.

## 🎯 Detected Issues

The model identifies 9 types of civic problems:

1. **Damaged_Concrete_Structures** - Cracks, holes in concrete infrastructure
2. **Damaged_Electric_Poles** - Broken or damaged electrical infrastructure  
3. **Damaged_Road_Signs** - Broken, missing, or vandalized road signs
4. **Dead_Animal_Pollution** - Road kill and animal carcasses
5. **Fallen_Trees** - Trees blocking roads or pathways
6. **Garbage** - Litter, trash accumulation
7. **Graffiti** - Unauthorized markings on public property
8. **Pothole** - Road surface holes and depressions
9. **Road_crack** - Cracks in road surfaces

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Your Model (if not already done)
Use your existing `temp_model.py` script in the conda environment:
```bash
# Activate your conda environment first
conda activate your_environment
python temp_model.py
```

### 3. Launch the System
```bash
python main.py
```

## 📁 Project Structure

```
📦 nivaari_object_detection/
├── 🚀 main.py                 # Main launcher (START HERE)
├── 🖥️ user_interface.py       # GUI application  
├── 💻 cli_interface.py        # Command line interface
├── 🔧 model_loader.py         # Model loading and inference
├── 🖼️ popup_display.py        # Visual results display
├── 🧪 test_validation.py      # Testing and validation tools
├── 📊 temp_model.py           # Your training script (DON'T MODIFY)
├── ⚙️ data.yaml              # Dataset configuration
├── 📋 requirements.txt        # Python dependencies
└── 📖 README.md              # This file
```

## 🎮 Usage Options

### 🖥️ GUI Interface (Recommended)
- **Intuitive visual interface**
- **Drag-and-drop image selection**
- **Real-time detection with popup results**
- **Built-in model testing and validation**
- **Confidence threshold adjustment**

Launch with:
```bash
python main.py  # Select option 1
```

### 💻 Command Line Interface
- **Perfect for automation and scripting**
- **Batch processing capabilities**
- **Performance benchmarking**

Examples:
```bash
# Detect issues in single image
python cli_interface.py --image "path/to/image.jpg"

# Interactive mode
python cli_interface.py --interactive

# Batch testing on directory
python cli_interface.py --test "path/to/test/dir"

# Run model validation
python cli_interface.py --validate

# Performance benchmark
python cli_interface.py --benchmark
```

### 🧪 Testing & Validation
```bash
# Test single image with popup results
python test_validation.py --image "test_image.jpg" --conf 0.3

# Validate model performance
python test_validation.py --validate

# Run performance benchmark
python test_validation.py --benchmark
```

## ✨ Key Features

### 🎯 **Smart Detection**
- YOLOv11-powered object detection
- 9 different civic issue classes
- Adjustable confidence thresholds
- GPU acceleration support

### 🖼️ **Visual Results**
- Interactive popup displays
- Bounding boxes with labels
- Confidence scores
- Save annotated images

### 📊 **Comprehensive Testing**
- Single image testing
- Directory batch processing  
- Model validation metrics
- Performance benchmarking
- GPU/CPU compatibility

### 🔧 **Easy Integration**
- Modular design
- Clean API structure
- Separate components for different functions
- Cross-platform compatibility

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: YOLOv11 (Ultralytics)
- **Classes**: 9 civic issue types
- **Input**: RGB images (various sizes)
- **Output**: Bounding boxes + class predictions + confidence scores

### System Requirements
- **Python**: 3.8+
- **GPU**: CUDA-compatible (optional, falls back to CPU)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for model weights

### Dependencies
- `ultralytics` - YOLOv11 implementation
- `torch` - PyTorch deep learning framework
- `opencv-python` - Computer vision operations
- `Pillow` - Image processing
- `tkinter` - GUI framework
- `PyYAML` - Configuration files

## 📈 Performance

### Inference Speed
- **GPU (RTX 4050)**: ~50-100 FPS
- **CPU**: ~5-15 FPS
- **Image Size**: Optimized for various resolutions

### Detection Accuracy
- Trained on diverse civic issues dataset
- mAP@0.5 metrics available via validation
- Optimized confidence thresholds per class

## 🔄 Workflow

1. **Image Input** → Select image via GUI or CLI
2. **Model Loading** → Automatic latest model detection
3. **Inference** → GPU-accelerated or CPU detection  
4. **Results Display** → Visual popup with bounding boxes
5. **Analysis** → Confidence scores and class labels

## 🆘 Troubleshooting

### Model Not Found
```bash
# List available models
python main.py  # Select option 4
```
If no models found, train using `temp_model.py` first.

### GPU Issues
System automatically falls back to CPU if GPU unavailable.

### Import Errors
```bash
pip install -r requirements.txt
```

### Performance Issues
- Lower confidence threshold
- Use smaller image sizes
- Check GPU memory usage

## 📊 Model Training

Your trained model from `temp_model.py` should be located in:
```
runs/train/[training_session]/weights/best.pt
```

The system automatically detects and loads the latest trained model.

## 💡 Tips

1. **Optimal Confidence**: Start with 0.25, adjust based on results
2. **Image Quality**: Higher resolution = better detection
3. **Testing**: Use validation tools before production
4. **Performance**: GPU significantly improves inference speed
5. **Batch Processing**: Use CLI for multiple images

## 🤝 Contributing

This system is designed for civic issue detection. Feel free to:
- Add new civic issue classes
- Improve detection accuracy
- Enhance user interface
- Add new features

## 📝 License

Built for civic improvement and urban planning applications.

---

**🏙️ Making cities better, one detection at a time!**

### 1. Setup Environment

```bash
# Check system requirements and install dependencies
python setup.py
```

### 2. Train Model

```bash
# Start training with optimized parameters for RTX 4050
python train_yolov11.py
```

### 3. Test Trained Model

```bash
# Interactive prediction mode
python predict_civic_issues.py

# Single image prediction
python predict_civic_issues.py --image "path/to/your/image.jpg"

# Batch prediction on folder
python predict_civic_issues.py --folder "path/to/images/folder"
```

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)

### Optimized For
- **RTX 4050 6GB VRAM** (training parameters specifically tuned)
- **Windows 10/11**
- **CUDA 11.8+**

### Without GPU
- CPU training is supported but will be significantly slower
- Reduced batch sizes and image resolution recommended

## 📁 Project Structure

```
nivaari_object_detection/
├── data.yaml                 # Dataset configuration
├── train/                    # Training data
│   ├── images/              # Training images
│   └── labels/              # Training labels (YOLO format)
├── valid/                   # Validation data
│   ├── images/              # Validation images
│   └── labels/              # Validation labels
├── test/                    # Test data
│   ├── images/              # Test images
│   └── labels/              # Test labels
├── train_yolov11.py         # Main training script
├── predict_civic_issues.py  # Inference script
├── setup.py                 # System setup and dependency installation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Training Parameters (RTX 4050 Optimized)

```python
# Safe parameters for RTX 4050 6GB VRAM
epochs = 25          # Moderate epoch count
batch_size = 8       # Safe for 6GB VRAM  
img_size = 640       # Standard resolution
model_size = "n"     # Nano model (lightweight)
patience = 5         # Early stopping
```

### Custom Configuration

Edit parameters in `train_yolov11.py`:

```python
# For more VRAM (RTX 3070/4060+)
batch_size = 16
model_size = "s"     # Small model

# For less VRAM (GTX 1060/RTX 3050)
batch_size = 4
img_size = 416
model_size = "n"
```

## 📊 Expected Results

### Training Metrics
- **mAP@0.5**: 0.75+ (target)
- **mAP@0.5:0.95**: 0.45+ (target)
- **Training time**: 2-4 hours (RTX 4050)
- **Model size**: ~6-14MB (nano model)

### Performance by Class
Results will vary based on dataset quality and training duration.

## 🖼️ Usage Examples

### Interactive Mode
```bash
python predict_civic_issues.py
# Follow prompts to analyze images
```

### Batch Processing
```bash
python predict_civic_issues.py --folder "test_images/" --conf 0.3 --output "results/"
```

### Programmatic Use
```python
from predict_civic_issues import CivicIssuesInference

# Initialize inference
detector = CivicIssuesInference("model/best.pt")

# Predict on single image
results, annotated = detector.predict_image("image.jpg")

# Batch predict
detector.batch_predict("images_folder/", "output_folder/")
```

## 📈 Training Process

### 1. Automatic Setup
The training script automatically:
- Downloads YOLOv11 weights
- Configures GPU/CPU settings
- Sets safe memory parameters
- Enables early stopping

### 2. Training Monitoring
Monitor training progress using:
- Real-time console output
- TensorBoard logs: `tensorboard --logdir runs/detect`
- Training curve plots (auto-generated)

### 3. Model Selection
Best model is saved based on validation mAP:
- `runs/detect/civic_issues_yolov11/weights/best.pt`
- `runs/detect/civic_issues_yolov11/weights/last.pt`

## ⚠️ Safety Features

### GPU Protection
- Automatic VRAM monitoring
- Safe batch size calculation
- Out-of-memory error handling
- Temperature warnings (if available)

### Early Stopping
- Prevents overfitting
- Saves training time
- Automatic best model selection

### Error Recovery
- Graceful failure handling
- Automatic cache clearing
- Checkpoint restoration

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
batch_size = 4  # or even 2

# Reduce image size  
img_size = 416  # instead of 640

# Use smaller model
model_size = "n"  # nano model
```

### Low Accuracy
- Increase training epochs: `epochs = 50`
- Improve dataset quality (more annotations)
- Use larger model if VRAM allows: `model_size = "s"`
- Adjust data augmentation parameters

### Slow Training
- Ensure CUDA is properly installed
- Close other GPU-intensive applications
- Use smaller dataset for testing
- Consider cloud GPU training

### Installation Issues
```bash
# Update pip
python -m pip install --upgrade pip

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install ultralytics separately
pip install ultralytics
```

## 📝 Dataset Format

The dataset uses YOLO format:
- Images in `.jpg`, `.png` format
- Labels in `.txt` files with same name as images
- Each line: `class_id center_x center_y width height` (normalized 0-1)

Example label file `image.txt`:
```
0 0.5 0.5 0.2 0.3    # Damaged_Concrete_Structures at center
8 0.8 0.9 0.1 0.1    # Road_crack in bottom-right
```

## 🔍 Model Details

### Architecture
- **Base**: YOLOv11n (Nano) for efficiency
- **Input**: 640x640 pixels (configurable)
- **Output**: Bounding boxes + class probabilities
- **Anchor-free**: Modern YOLO architecture

### Training Features
- Mixed Precision Training (AMP)
- Data augmentation (HSV, geometric)
- Multi-scale training
- Label smoothing
- Mosaic augmentation

## 📚 Additional Resources

### Documentation
- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Datasets
- Expand dataset with [Roboflow](https://roboflow.com/)
- Civic issues datasets on [OpenImages](https://storage.googleapis.com/openimages/web/index.html)

### Hardware Optimization
- NVIDIA GPU optimization guides
- CUDA best practices
- Memory profiling tools

## 🤝 Contributing

1. Fork the repository
2. Add your improvements
3. Test with different hardware configurations
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11 implementation
- **Roboflow** for dataset management tools  
- **PyTorch** team for the deep learning framework
- **CUDA** team for GPU acceleration

---

**⭐ If this project helps you, please star the repository!**

For questions or issues, please open a GitHub issue or contact the development team.
