# Malaria Diagnosis using CNN Transfer Learning

A comprehensive deep learning project for automated malaria diagnosis using MobileNetV2 transfer learning on the NIH Malaria Cell Images dataset.

## 🎯 Project Overview

This project implements an automated malaria diagnosis system using transfer learning with MobileNetV2. The notebook runs 7 systematic experiments testing various hyperparameters to find optimal configurations for malaria parasite detection in blood cell images.

## ✨ Features

- **Fully Automated**: Just click "Run All" and the notebook handles everything
- **Bulletproof Checkpoint Recovery**: Auto-saves after each experiment, resumes if interrupted
- **7 Systematic Experiments**: Tests different hyperparameters including:
  - Model architectures (alpha variations)
  - Data augmentation techniques
  - Regularization (dropout)
  - Optimizer configurations
  - Learning rate schedules
- **Comprehensive Analysis**: Generates detailed visualizations and comparison metrics
- **Optimized Performance**: Uses 160x160 images for 30% faster training
- **Memory Efficient**: Includes memory cleanup to prevent crashes
- **GPU Accelerated**: Optimized for GPU training (~2-3 hours runtime)

## 📊 Dataset

Uses the **NIH Malaria Cell Images** dataset from Kaggle:
- **27,558** labeled cell images
- **2 classes**: 
  - Parasitized (infected with malaria parasites)
  - Uninfected (healthy blood cells)
- Automatically downloaded via `kagglehub`

## 🧪 Experimental Configurations

The notebook runs 7 different experiments:

1. **Baseline**: Standard MobileNetV2 (alpha=1.0) with data augmentation
2. **Lightweight Model**: Reduced width (alpha=0.75) for faster inference
3. **Minimal Augmentation**: Tests impact of reduced data augmentation
4. **Regularization**: Adds dropout (0.5) to prevent overfitting
5. **Alternative Optimizer**: Uses RMSprop instead of Adam
6. **Learning Rate Decay**: Implements exponential decay schedule
7. **Combined Best Practices**: Combines successful techniques from previous experiments

## 🔧 Requirements

### Python Libraries
```
numpy
pandas
matplotlib
seaborn
tensorflow >= 2.x
keras
scikit-learn
opencv-python
pillow
kagglehub
```

### Hardware
- **GPU recommended** for training (2-3 hours)
- CPU training possible but slower
- Minimum 8GB RAM

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/Elvis-Kayonga/Malaria-Diagnosis-CNN-Transfer-Learning-ML-model.git
cd Malaria-Diagnosis-CNN-Transfer-Learning-ML-model
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn opencv-python pillow kagglehub
```

3. Set up Kaggle API credentials (for dataset download):
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🚀 Usage

### Option 1: Jupyter Notebook
```bash
jupyter notebook elvis-malaria-model.ipynb
```
Then click "Run All" to execute all experiments.

### Option 2: Kaggle Notebook
1. Upload the notebook to Kaggle
2. Enable GPU accelerator
3. Click "Run All"

### Resuming After Interruption
If the notebook is interrupted (power outage, timeout, etc.):
1. Simply click "Run All" again
2. The checkpoint system automatically resumes from the last completed experiment
3. No progress will be lost!

## 📈 Results

The notebook generates comprehensive results including:

- **Training/Validation Curves**: Loss and accuracy over epochs
- **Confusion Matrices**: Detailed classification performance
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Comparison Tables**: Side-by-side metrics for all experiments
- **Hyperparameter Analysis**: Impact of different configurations
- **Final Summary Report**: Best performing model and recommendations

### Key Metrics Tracked
- Accuracy (training & validation)
- Loss (training & validation)
- Precision, Recall, F1-Score
- ROC-AUC Score
- Training time per experiment

## 📁 Project Structure

```
Malaria-Diagnosis-CNN-Transfer-Learning-ML-model/
│
├── README.md                      # Project documentation
├── elvis-malaria-model.ipynb      # Main Jupyter notebook with all experiments
└── experiment_results/            # Generated during execution
    ├── checkpoints/               # Model checkpoints
    ├── visualizations/            # Generated plots and figures
    └── metrics/                   # Performance metrics and logs
```

## 🔍 Workflow Steps

1. **Import Libraries**: Load all required dependencies
2. **Download Dataset**: Automatic download of NIH Malaria dataset
3. **Visualize Samples**: Display example images from both classes
4. **Set Random Seeds**: Ensure reproducibility
5. **Define Experiments**: Configure all 7 experimental setups
6. **Prepare Data Split**: Train/validation/test split with stratification
7. **Define Helper Functions**: Utilities for training and evaluation
8. **Run Experiments**: Execute all 7 experiments with checkpointing
9. **Load Results**: Retrieve all experiment results
10. **Create Visualizations**: Generate comparison plots
11. **Compare Learning Curves**: Analyze training dynamics
12. **Analyze Hyperparameters**: Study impact of different configurations
13. **Generate Report**: Produce final summary with recommendations

## 🛡️ Safety Features

- ✅ Auto-saves after each experiment
- ✅ Checkpoint recovery system
- ✅ Works across Kaggle session timeouts
- ✅ Memory cleanup prevents crashes
- ✅ Error handling ensures no data loss
- ✅ GPU memory growth configuration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: NIH Malaria Cell Images dataset
- **Model**: MobileNetV2 by Google
- **Framework**: TensorFlow/Keras
- **Platform**: Kaggle Notebooks

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project was developed as part of a machine learning assignment focusing on transfer learning and systematic hyperparameter experimentation.
