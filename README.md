
# Brain Tumor Classification Using MRI Scans

## Overview
This project is a deep learning-based approach to accurately detect and classify brain tumor types from MRI images. Using convolutional neural networks (CNNs), the model is trained on MRI scan data to identify the presence and type of tumor based on labeled image samples. This solution aims to assist medical professionals by providing a reliable, automated classification system that can work with MRI data.

## Project Description
The goal of this project is to build a machine learning model that classifies brain tumors from MRI images into different categories. The model is trained on an extensive dataset of labeled images and tested on unseen data to evaluate its performance. The project is organized to handle data in various formats, including CSV for metadata, JPG for MRI images, and PDF documents for related reports.

## Key Features
- **Image-Based Tumor Classification**: Detects and classifies brain tumors from MRI images with high accuracy.
- **Multi-Format Data Handling**: Processes images in JPG format and metadata in CSV format, with auxiliary information in PDF format.
- **Model Evaluation**: Provides accuracy and classification metrics, enabling detailed performance analysis.
- **Confusion Matrix and Visualization**: Visualizes test results for qualitative insights.
- **Scalable Model Architecture**: Uses CNN architecture optimized for image classification.

## Technologies Used
- **Python**: For scripting and model building.
- **TensorFlow/Keras**: For deep learning and CNN implementation.
- **NumPy**: For numerical operations.
- **Pandas**: For data handling and manipulation.
- **Matplotlib**: For result visualization.
- **Scikit-Learn**: For model evaluation metrics.
- **OpenCV & PIL**: For image preprocessing.

## Project Structure
- **Brain_MRI_conditions.csv**: Contains metadata about the MRI images and associated tumor conditions.
- **Brain_MRI_tumor.pdf**: A PDF report or supplementary data relevant to the project.
- **ST000001**: Folder containing MRI images organized into subfolders by class:
  - **SE000001** to **SE000010**: Each subfolder corresponds to a specific tumor type or condition and contains relevant MRI images.
- **model_training.ipynb**: Jupyter notebook with code to train and evaluate the model.

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Prepare the Dataset**:
   - Place the CSV, PDF, and MRI image folders (`Brain_MRI_conditions.csv`, `Brain_MRI_tumor.pdf`, and `ST000001`) in the main project directory.
   - Ensure the images are organized into class folders within `ST000001` as described.

3. **Train the Model**:
   Open `model_training.ipynb` in Jupyter Notebook or Jupyter Lab and run all cells to train and evaluate the model.

4. **Test the Model**:
   Run the model on the test set or any new images by following the `Testing` section in the notebook.

## Results
- **Training Accuracy**: 97.86%
- **Validation Accuracy**: 95.71%
- **Model Evaluation**: The model achieves a high accuracy rate with minimal overfitting, as indicated by low training and validation losses (0.0928 and 0.1411 respectively).
- **Visualization**: Displays test images with predicted and actual labels, along with a confusion matrix for in-depth performance analysis.

## Future Improvement
- **Data Augmentation**: Adding more image transformations to increase model robustness.
- **Fine-Tuning**: Experiment with advanced architectures or hyperparameter tuning for potentially better performance.
- **Deployment**: Package the model into a web application to enable easy access for medical professionals.
- **Integration with Clinical Data**: Combine with patient data for a comprehensive diagnosis model.

--- 
