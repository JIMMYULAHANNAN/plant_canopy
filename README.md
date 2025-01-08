# plant_disease-canopy
### **Objectives**
This project builds and trains a Convolutional Neural Network (CNN) for multi-class image classification using TensorFlow and Keras, focusing on preprocessing, data augmentation, training, evaluation, and visualization of results.

---

### **Skills Used**
1. **Data Handling**:
   - Load and preprocess images using TensorFlow datasets.
   - Partition data into training, validation, and testing sets.

2. **Image Processing**:
   - Resizing and normalizing images.
   - Applying data augmentation techniques like flipping and rotation to improve generalization.

3. **Deep Learning**:
   - Designing a CNN architecture with convolutional, pooling, and dense layers.
   - Training and evaluating the model using categorical cross-entropy loss.

4. **Visualization**:
   - Displaying sample images from the dataset.
   - Plotting training accuracy and loss trends.
   - Visualizing predictions with confidence scores.

---

### **Libraries Imported**
1. **TensorFlow**: For building, training, and evaluating the CNN model.
2. **Matplotlib**: To visualize data and results.
3. **IPython.display**: To enhance interactive outputs.

---

### **Explanation of the Code**

#### **1. Setting Constants**
Defines hyperparameters like batch size, image size, number of channels, and epochs for training.

#### **2. Loading the Dataset**
- **Dataset Loading**:
  - Uses `image_dataset_from_directory` to load images from a directory.
  - Images are resized to \(256 \times 256\) and batched.
  - Labels are automatically inferred from subdirectory names.

- **Visualizing Images**:
  - Displays random images from the dataset with their corresponding class labels.

#### **3. Dataset Partitioning**
- Splits the dataset into training, validation, and test sets using a custom function (`get_dataset_partitions_tf`).
- Ensures shuffling for randomness and balanced splits.

#### **4. Dataset Optimization**
- **Caching**: Stores the data in memory to improve training speed.
- **Prefetching**: Ensures data is loaded while the model trains on the previous batch, reducing idle time.

#### **5. Preprocessing Layers**
- **Resizing and Normalization**:
  - Ensures all images are resized to a consistent size and normalized to a range of [0, 1].
- **Data Augmentation**:
  - Adds variability to the training data by flipping and rotating images.

#### **6. CNN Model Architecture**
- **Initial Layers**:
  - Resizing and normalization layers are integrated into the model to handle variations during inference.
- **Convolutional Layers**:
  - Multiple convolutional layers extract hierarchical features.
- **Pooling Layers**:
  - Reduce spatial dimensions to retain essential features.
- **Dense Layers**:
  - Fully connected layers map the features to class probabilities.
- **Output Layer**:
  - Uses softmax activation to generate class probabilities for multi-class classification.

#### **7. Model Compilation and Training**
- **Compilation**:
  - Adam optimizer is used for adaptive learning.
  - Loss function: Sparse categorical cross-entropy (suitable for multi-class classification with integer labels).
- **Training**:
  - Trains the model for 10 epochs, tracking accuracy and loss for both training and validation datasets.

#### **8. Evaluating the Model**
- Evaluates the model on the test dataset and prints accuracy metrics.

#### **9. Visualization of Training Performance**
- Plots training and validation accuracy/loss over epochs to identify overfitting or underfitting.

#### **10. Predictions and Results**
- **Inference on Sample Images**:
  - Displays actual and predicted labels for test images along with confidence scores.
- **Custom Prediction Function**:
  - Processes an image, predicts its class, and returns the label with confidence.

---

### **Project Summary**
This project demonstrates how to preprocess and augment image datasets, build a CNN for multi-class classification, and evaluate its performance. It emphasizes modular and efficient handling of data pipelines while ensuring robust model training and evaluation.

code:
