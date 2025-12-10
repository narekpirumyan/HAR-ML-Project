# Human Activity Recognition (HAR) Using Smartphones

A comprehensive machine learning project that classifies human physical activities from smartphone sensor data using both traditional machine learning and deep learning approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [License & Citation](#license--citation)

## 🎯 Overview

This project implements and compares multiple machine learning approaches to classify 6 human activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) from smartphone sensor data. The project demonstrates:

- **Comprehensive EDA**: Exploratory data analysis with feature correlation, variance analysis, and visualization
- **Traditional ML Models**: Random Forest, XGBoost, and SVM classifiers
- **Deep Learning**: Hybrid CNN-LSTM architecture for automatic feature learning
- **Feature Selection**: Three methods (univariate, model-based, RFE) for dimensionality reduction
- **Model Comparison**: Performance analysis across different algorithmic approaches

### Activities Classified

1. **WALKING**
2. **WALKING_UPSTAIRS**
3. **WALKING_DOWNSTAIRS**
4. **SITTING**
5. **STANDING**
6. **LAYING**

## 📊 Dataset

### UCI HAR Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Subjects**: 30 volunteers (ages 19-48)
- **Sampling Rate**: 50 Hz
- **Window Size**: 2.56 seconds (128 readings per window)
- **Overlap**: 50% between windows

### Data Structure

- **Training Set**: 7,352 samples (70% of volunteers)
- **Test Set**: 2,947 samples (30% of volunteers)
- **Engineered Features**: 561 features (normalized to [-1, 1])
- **Raw Signals**: 9 sensor channels × 128 timesteps (for deep learning)

### Sensor Data

- **Accelerometer**: 3-axis (X, Y, Z)
  - Total acceleration (gravity + body motion)
  - Body acceleration (gravity removed using Butterworth filter)
- **Gyroscope**: 3-axis (X, Y, Z) - angular velocity (rad/s)

### Feature Engineering

Features extracted from time and frequency domains:
- **Time Domain**: Mean, Std, MAD, Max, Min, Energy, Entropy, Correlation, SMA
- **Frequency Domain**: FFT coefficients, meanFreq, skewness, kurtosis
- **Derived Signals**: Jerk signals, magnitude signals, angles between vectors

**Note**: The 561 features were already normalized to [-1, 1] by the dataset creators.

## 📁 Project Structure

```
HAR/
│
├── HAR.ipynb                          # Main Jupyter notebook with all analysis
├── README.md                          # This file
│
├── HAR with Smartphone/
│   └── UCI HAR Dataset/
│       └── UCI HAR Dataset/
│           ├── activity_labels.txt   # Activity label mappings
│           ├── features.txt           # Feature names
│           ├── features_info.txt     # Feature descriptions
│           ├── README.txt            # Dataset README
│           │
│           ├── train/                # Training data
│           │   ├── X_train.txt       # Training features (561 features)
│           │   ├── y_train.txt       # Training labels
│           │   ├── subject_train.txt # Subject IDs
│           │   └── Inertial Signals/ # Raw sensor signals
│           │
│           └── test/                  # Test data
│               ├── X_test.txt        # Test features (561 features)
│               ├── y_test.txt        # Test labels
│               ├── subject_test.txt  # Subject IDs
│               └── Inertial Signals/ # Raw sensor signals
│
└── model_checkpoints/                 # Saved deep learning models
    ├── best_cnn_lstm_model.h5
    └── best_cnn_lstm_model.keras
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn xgboost tensorflow keras matplotlib seaborn
```

Or install from requirements file (if provided):

```bash
pip install -r requirements.txt
```

### Required Packages

- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Traditional machine learning models
- **xgboost**: Gradient boosting framework
- **tensorflow/keras**: Deep learning framework
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualizations

## 💻 Usage

### Running the Notebook

1. **Clone or download** this repository
2. **Navigate** to the project directory
3. **Open** `HAR.ipynb` in Jupyter Notebook
4. **Run all cells** sequentially, or run specific sections:
   - **Section 1**: Data Loading & EDA
   - **Section 2**: Feature Selection
   - **Section 3**: Traditional ML Models
   - **Section 4**: Deep Learning Model
   - **Section 5**: Model Comparison & Evaluation

### Expected Runtime

- **EDA & Feature Selection**: ~5-10 minutes
- **Traditional ML Models**: ~1-2 minutes
- **Deep Learning Training**: ~84 minutes (on CPU)

## 🔬 Methodology

### Phase 1: Exploratory Data Analysis (EDA)

- Data quality assessment (missing values, normalization status)
- Feature variance and correlation analysis
- Activity distribution analysis
- Subject-activity relationships
- Comprehensive visualizations

### Phase 2: Feature Selection

Three approaches implemented:

1. **Univariate Feature Selection**
   - Statistical tests: f_classif, mutual_info_classif, chi2
   - Optimal: 150 features (73% reduction)

2. **Model-Based Feature Importance**
   - Random Forest and XGBoost importance scores
   - Cross-model importance (surprising effectiveness)
   - Optimal: 200-250 features

3. **Recursive Feature Elimination (RFE)**
   - Iterative feature removal
   - Optimal: 200-250 features

### Phase 3: Traditional Machine Learning

**Models Implemented:**

1. **Random Forest**
   - n_estimators: 100
   - Test Accuracy: 92.53%
   - Training Time: ~4.22s

2. **XGBoost**
   - n_estimators: 200, max_depth: 8
   - Test Accuracy: 93.82%
   - Training Time: ~37.44s

3. **Support Vector Machine (SVM)**
   - Kernel: RBF, C: 10.0
   - Test Accuracy: **96.20%** ⭐ (Best)
   - Training Time: ~10.54s

### Phase 4: Deep Learning

**Hybrid CNN-LSTM Architecture:**

```
Input (128, 9)
    ↓
[3 CNN Blocks]
  Conv1D(64) → Conv1D(128) → Conv1D(256)
  BatchNorm + MaxPool + Dropout(0.3)
    ↓
[2 LSTM Layers]
  LSTM(128) → LSTM(64)
  Dropout(0.3)
    ↓
[2 Dense Layers]
  Dense(64) → Dense(32)
  BatchNorm + Dropout
    ↓
Output: Dense(6, softmax)
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Batch Size: 64
- Epochs: 100 (with early stopping)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

**Results:**
- Test Accuracy: 92.47%
- Training Time: ~84 minutes
- Parameters: 380,070

## 📈 Results

### Model Performance Comparison

| Model | Type | Test Accuracy | Training Time | Parameters |
|-------|------|---------------|---------------|------------|
| **SVM** | Traditional ML | **96.20%** | 10.54s | N/A |
| XGBoost | Traditional ML | 93.82% | 37.44s | N/A |
| Random Forest | Traditional ML | 92.53% | 4.22s | N/A |
| Hybrid CNN-LSTM | Deep Learning | 92.47% | 5,020s | 380,070 |

### Key Findings

1. **SVM achieved the best accuracy** (96.20%) among all models
2. **Traditional ML outperformed deep learning** due to well-engineered features
3. **Feature selection** can reduce features by 73% with minimal accuracy loss
4. **Cross-model feature importance** works surprisingly well
5. **All models** show excellent generalization (small train-test gaps)

### Feature Selection Results

- **Optimal feature count**: 150-250 features (vs. 561 original)
- **Best method**: Model-based importance with cross-model selection
- **Accuracy retention**: 99%+ with 200-250 features
- **Training time improvement**: 78-86% reduction

## 🛠️ Technologies Used

- **Python 3.x**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Traditional ML models and utilities
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 🔍 Key Findings

### Why Traditional ML Performed Better?

1. **Well-Engineered Features**: 561 expert-designed features capture domain knowledge
2. **Normalized Data**: Features pre-normalized to [-1, 1] optimal for SVM
3. **Dataset Size**: Traditional ML effective with moderate dataset sizes
4. **Feature Quality**: Expert features outperform learned features for this problem

### When to Use Each Approach?

**Use Traditional ML (SVM/XGBoost) When:**
- ✅ Pre-engineered features are available
- ✅ Fast training/inference required
- ✅ Interpretability is important
- ✅ Limited computational resources
- ✅ Moderate dataset size

**Use Deep Learning (CNN-LSTM) When:**
- ✅ Raw sensor data available (no feature engineering)
- ✅ Large datasets for better generalization
- ✅ Complex temporal patterns need learning
- ✅ Computational resources available
- ✅ Transfer learning needed

## 🚧 Future Improvements

1. **Hyperparameter Optimization**
   - Grid search or Bayesian optimization
   - Learning rate scheduling experiments
   - Architecture tuning for deep learning

2. **Model Enhancements**
   - Ensemble methods combining multiple models
   - Attention mechanisms for LSTM
   - Data augmentation for deep learning

3. **Feature Engineering**
   - Apply optimal feature selection in production
   - Experiment with additional feature combinations
   - Domain-specific feature engineering

4. **Deployment**
   - Model serialization for production
   - Real-time inference optimization
   - Mobile deployment considerations

## 📄 License & Citation

### Dataset Citation

If you use this dataset, please cite:

```
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. 
Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 
International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012
```

### Dataset License

The UCI HAR Dataset is in the public domain and can be used freely.

### Project License

This project is provided for educational and research purposes.

## 📚 Additional Resources

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [Experiment Video](http://www.youtube.com/watch?v=XOEN9W05_4A)

## 👤 Author

Created as part of AI/ML coursework - 2025

---

**Note**: This project focuses on comparing different algorithmic approaches rather than optimizing each model individually. The objective is to demonstrate the effectiveness of various machine learning techniques for human activity recognition.

