# Smart Manufacturing AI - PCB Defect Detection

An AI-powered solution for automated PCB (Printed Circuit Board) defect detection using deep learning and computer vision techniques. This project demonstrates practical smart manufacturing applications by implementing a complete pipeline from data preprocessing to model deployment for industrial quality control.

## Overview

This repository provides a complete pipeline for building an intelligent PCB defect detection system using YOLOv5. The project leverages the [PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects/data) from Kaggle to train a robust object detection model capable of identifying various types of manufacturing defects in printed circuit boards. The implementation focuses on real-world industrial applications with support for edge deployment formats.

## Repository Structure

```
Smart-Manufacturing-AI/
│
├── Smart Manufacturing AI.ipynb    # Main notebook with complete pipeline
├── LICENSE                         # Apache-2.0 license file
└── README.md                      # Project documentation
```

## Dataset

The project uses the **PCB Defects Dataset** from Kaggle, which contains:
- High-resolution PCB images with various defect types
- XML annotation files with bounding box coordinates
- Multiple defect categories for comprehensive training

**Defect Classes:**
- `mouse_bite` - Small indentations on PCB edges
- `spur` - Unwanted copper extensions
- `open_circuit` - Broken electrical connections
- `short` - Unintended electrical connections
- `missing_hole` - Absent drilling points
- `spurious_copper` - Extra copper deposits

## Features

### Data Preparation & Processing
- Automated dataset splitting into train/validation/test sets with customizable ratios
- XML annotation parsing and conversion to YOLO format
- Bounding box coordinate normalization
- Defect class mapping and label generation
- Interactive data exploration and visualization

### Model Training & Configuration
- YOLOv5-based object detection optimized for PCB defects
- Custom training configuration with adjustable hyperparameters
- Batch size, epoch count, and data augmentation customization
- Real-time training metrics and validation tracking

### Model Export & Deployment
- Export to multiple formats for production deployment:
  - ONNX format for cross-platform inference
  - TensorFlow SavedModel for TensorFlow Serving
  - TensorFlow Lite for edge and mobile deployment
- Optimized models for low-power industrial devices

### Visualization & Analysis
- Data distribution analysis and visualization
- Bounding box annotation display
- Training/validation split breakdowns
- Model performance metrics and evaluation charts

## Technical Implementation

### Core Technologies
- **YOLOv5**: State-of-the-art object detection framework
- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization

### Key Components
- **Dataset Management**: Automated splitting and organization
- **Annotation Processing**: XML to YOLO format conversion
- **Model Training**: Custom YOLOv5 configuration for PCB defects
- **Format Conversion**: Multi-format model export pipeline
- **Deployment Ready**: Edge-optimized model variants

## How To Run

### Prerequisites
```bash
pip install torch torchvision pandas numpy matplotlib seaborn opencv-python
pip install kagglehub pyyaml joblib
```

### Step-by-Step Execution
1. **Clone the repository**
   ```bash
   git clone https://github.com/PranavSingla122/Smart-Manufacturing-AI.git
   cd Smart-Manufacturing-AI
   ```

2. **Setup Environment**
   - Open Google Colab or local Jupyter Lab
   - Install required dependencies

3. **Dataset Preparation**
   - Download the PCB defects dataset using kagglehub
   - The notebook automatically handles dataset organization

4. **Run the Pipeline**
   - Open `Smart Manufacturing AI.ipynb`
   - Execute cells sequentially following the notebook sections:
     - Data setup and exploration
     - Annotation processing and conversion
     - Model training configuration
     - YOLOv5 training execution
     - Model evaluation and metrics
     - Export to production formats

5. **Model Deployment**
   - Use exported ONNX models for cross-platform deployment
   - Deploy TensorFlow Lite models on edge devices
   - Integrate models into manufacturing quality control systems

## Results & Performance

The trained model achieves:
- High accuracy in detecting multiple defect types simultaneously
- Real-time inference capability suitable for production lines
- Robust performance across various PCB layouts and designs
- Optimized model sizes for edge deployment scenarios

## Applications in Smart Manufacturing

- **Quality Control**: Automated inspection of PCB manufacturing lines
- **Process Optimization**: Real-time defect feedback for process improvement
- **Cost Reduction**: Reduced manual inspection requirements
- **Consistency**: Standardized defect detection across production facilities
- **Traceability**: Detailed defect logging and analytics

## Model Export Formats

- **ONNX**: Cross-platform deployment with optimized inference
- **TensorFlow SavedModel**: Integration with TensorFlow ecosystem
- **TensorFlow Lite**: Mobile and edge device deployment
- **PyTorch**: Native format for further research and development

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas for improvement include:
- Additional defect class implementations
- Alternative model architectures (YOLOv8, RCNN variants)
- Enhanced data augmentation techniques
- Production deployment scripts and examples
- Performance optimization for specific hardware

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Create a Pull Request


## Acknowledgments

- [Kaggle PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects/data) by akhatova
- YOLOv5 team for the excellent object detection framework
- Open-source community for tools and libraries used in this project
