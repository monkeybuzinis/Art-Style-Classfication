# Art Style Classification with Transfer Learning

A deep learning project that classifies artwork by artistic style using transfer learning with EfficientNet-B0. This project demonstrates how pre-trained convolutional neural networks can learn abstract artistic concepts from visual characteristics alone, achieving human non-expert level performance.

## Overview

This project tackles the challenging problem of art style classification, which is difficult even for human experts. By focusing exclusively on visual characteristics (line, color, composition, technique, brushwork, texture), the model learns to distinguish between different artistic movements using transfer learning from ImageNet-pretrained EfficientNet-B0.

## Dataset

- **Source**: [WikiArt Dataset from Kaggle](https://www.kaggle.com/datasets/steubk/wikiart)
- **Total Images**: ~81,444 images (training set: ~43,000)
- **Art Styles**: 16+ different artistic movements including:
  - Impressionism (13,060 images - 16.04%)
  - Realism (10,733 images - 13.18%)
  - Romanticism (7,019 images - 8.62%)
  - Expressionism (6,736 images - 8.27%)
  - Post-Impressionism (6,450 images - 7.92%)
  - Symbolism, Art Nouveau, Baroque, Abstract Expressionism, and more

### Dataset Characteristics

- **Class Imbalance**: Class sizes range from 336 to 2,100 training samples
- **High Intra-class Variation**: Styles like Impressionism include diverse works (Monet's soft landscapes vs. Degas' dynamic figures)
- **Low Inter-class Separation**: Similar styles like Realism and Romanticism both use naturalistic rendering

## Model Architecture

### EfficientNet-B0

EfficientNet-B0 is the baseline model in the EfficientNet family, developed by Google AI to balance accuracy and computational efficiency through systematic scaling of depth, width, and image resolution.

**Architecture Components:**

1. **Stem**
   - Initial convolution (32 filters, 3x3 kernel, stride 2)
   - Batch normalization and ReLU6 activation

2. **Body**
   - Series of MBConv blocks with depthwise separable convolutions
   - Squeeze-and-excitation layers for channel attention
   - Configurable expansion ratios, kernel sizes, and strides

3. **Head**
   - Final convolutional block
   - Global average pooling
   - Fully connected layer with softmax activation for classification

## Transfer Learning Approach

### Why Transfer Learning?

**Feature Reusability**: Low-level features (edges, textures, colors) learned on ImageNet's 1.2M images transfer effectively to art analysis, providing a strong foundation without art-specific pre-training.

### Two-Stage Training Strategy

#### Stage 1: Feature Extraction
- **Frozen Base Model**: EfficientNet-B0 backbone remains frozen
- **Trainable Layers**: Only the classification head is trained
- **Purpose**: Learn task-specific features while preserving ImageNet knowledge
- **Benefits**: Faster training, reduced overfitting risk

#### Stage 2: Fine-Tuning
- **Unfrozen Layers**: Last 50 layers of EfficientNet-B0 are unfrozen
- **Data Augmentation**: Dynamic augmentation (random flips, rotations, zooms) to prevent memorization
- **Purpose**: Learn style-invariant patterns and refine feature representations
- **Benefits**: Improved accuracy, better generalization

### Benefits of Two-Stage Transfer Learning

-  **Improved Performance**: Better accuracy than training from scratch, especially with limited data
-  **Faster Training**: Initial stage trains only final layers; fine-tuning requires less computation
-  **Reduced Overfitting**: Freezing backbone initially prevents early overfitting to smaller dataset

## Requirements

### Python Packages

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn kagglehub tqdm
```

### Key Libraries
- **Deep Learning**: `tensorflow`, `keras`
- **Data Processing**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Metrics**: `scikit-learn`
- **Dataset**: `kagglehub`
- **Utilities**: `tqdm`, `pathlib`

### Hardware Requirements

- **GPU**: Highly recommended (CUDA-compatible GPU with at least 8GB VRAM)
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: At least 10GB free space for dataset and model weights

## Project Structure

```
art-style-classification/
├── Art-Style-Classification-with-Transfer-Learning.ipynb
└── README.md
```

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Art-Style-Classification-with-Transfer-Learning.ipynb
   ```

2. **Download the Dataset**:
   The notebook automatically downloads the WikiArt dataset from Kaggle using `kagglehub`

3. **Run the Notebook Cells Sequentially**:
   - Dataset exploration and analysis
   - Data preprocessing and splitting
   - Stage 1: Feature extraction training
   - Stage 2: Fine-tuning
   - Model evaluation and visualization

## Training Process

### Data Preprocessing
- Image resizing to EfficientNet-B0 input size (typically 224x224)
- Data augmentation (flips, rotations, zooms)
- Train/validation/test split
- Handling class imbalance

### Training Configuration
- **Optimizer**: Adam or similar adaptive optimizer
- **Callbacks**:
  - `EarlyStopping`: Stop training when validation performance plateaus
  - `ReduceLROnPlateau`: Reduce learning rate when validation stalls
  - `ModelCheckpoint`: Save best model weights during training
- **Loss Function**: Categorical cross-entropy
- **Metrics**: Accuracy, precision, recall, F1-score

## Evaluation Metrics

The project includes comprehensive evaluation:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score per class
- **Confusion Matrix**: Visual representation of classification errors
- **ROC-AUC Curves**: For multi-class classification evaluation
- **Classification Report**: Detailed per-class performance metrics

## Key Challenges Addressed

1. **High Intra-class Variation**: Same style can look very different (e.g., Monet vs. Degas in Impressionism)
2. **Low Inter-class Separation**: Similar styles share visual characteristics
3. **Class Imbalance**: Uneven distribution of samples across art styles
4. **Abstract Concepts**: Learning artistic style from visual data alone

## Results

The model achieves human non-expert level performance while requiring only hours of training, demonstrating that deep learning can effectively learn abstract artistic concepts from visual data.

## Visualizations

The notebook includes:
- Dataset distribution charts
- Training/validation accuracy and loss curves
- Confusion matrices
- ROC curves
- Sample predictions with confidence scores

## References

- [EfficientNet Architecture](https://www.geeksforgeeks.org/computer-vision/efficientnet-architecture/)
- [WikiArt Dataset on Kaggle](https://www.kaggle.com/datasets/steubk/wikiart)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## Author

**Khanh Le**



## License

This project uses the WikiArt dataset from Kaggle. Please refer to the dataset's license for usage terms.

## Notes

- The model focuses on visual characteristics only, without using historical/cultural context
- Training time varies based on hardware (GPU highly recommended)
- Model performance depends on class balance and data quality
- Fine-tuning stage requires careful learning rate scheduling to avoid catastrophic forgetting

# Art-Style-Classfication
