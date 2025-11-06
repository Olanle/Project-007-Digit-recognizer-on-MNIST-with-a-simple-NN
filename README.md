# Project-007-Digit-recognizer-on-MNIST-with-a-simple-NN

---

This project demonstrates how to build and optimize a **Neural Network (NN)** using **TensorFlow/Keras** to recognize handwritten digits from the **MNIST dataset**.
It walks through data preprocessing, model design, training, and prediction — and includes an improved architecture that reduces overfitting by adding an extra hidden layer.

---

## 📁 Project Overview

| Stage             | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| **1. Dataset**    | MNIST — 70,000 grayscale images of handwritten digits (0–9).               |
| **2. Task**       | Multi-class classification (10 classes).                                   |
| **3. Framework**  | TensorFlow/Keras.                                                          |
| **4. Model Type** | Feedforward Neural Network (Fully Connected).                              |
| **5. Goal**       | Classify handwritten digits accurately and generalize well to unseen data. |

---

## ⚙️ Model Architecture

| Layer    | Type  | Neurons | Activation | Purpose                                      |
| -------- | ----- | ------- | ---------- | -------------------------------------------- |
| Input    | Dense | 784     | —          | Flattened pixel inputs (28×28)               |
| Hidden 1 | Dense | 128     | ReLU       | Learns low-level stroke patterns             |
| Hidden 2 | Dense | 64      | ReLU       | Learns higher-level digit structures         |
| Hidden 3 | Dense | 16      | ReLU       | Adds regularization and generalization depth |
| Output   | Dense | 10      | Softmax    | Predicts probabilities for digits (0–9)      |

---

## 🧩 Key Model Improvement

To reduce **overfitting**, an additional hidden layer was added:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),   # 🆕 Added layer to improve generalization
    Dense(10, activation='softmax')
])
```

### Effects of This Change:

* **Better generalization** on unseen data.
* **Smoother validation accuracy** curve.
* **Reduced gap** between training and validation loss.

---

## Implementation Steps

1. **Data Preprocessing**

   * Flatten 28×28 images → 784 features.
   * Normalize pixel values (0–255 → 0–1).
   * One-hot encode labels (0–9).

2. **Model Compilation**

   * Optimizer: `Adam`
   * Loss: `Categorical Crossentropy`
   * Metric: `Accuracy`

3. **Training**

   * Epochs: 10
   * Batch size: 128
   * Validation split: 0.1

4. **Testing**

   * Evaluate model on unseen test data.
   * Predict handwritten digits from custom images.

---

## Core Code Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build improved model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
```

---

## Performance Results

| Metric              | Before     | After Improvement     |
| ------------------- | ---------- | --------------------- |
| Training Accuracy   | ~99%       | ~98%                  |
| Validation Accuracy | ~96%       | ~97.5%                |
| Overfitting         | Noticeable | Significantly reduced |

---

## Predicting on Custom Images

```python
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess your image (28x28 grayscale)
img = image.load_img("my_digit.png", color_mode="grayscale", target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = 255 - img_array   # Invert if needed (white background)
img_array = img_array.reshape(1, 784) / 255.0

plt.imshow(img_array.reshape(28,28), cmap='gray')
plt.title("Uploaded Digit")
plt.show()

prediction = model.predict(img_array)
digit = np.argmax(prediction)
print("Predicted Digit:", digit)
```

---

## Key Takeaways

* Neural networks can effectively classify handwritten digits with minimal architecture.
* Overfitting can be mitigated by **adding intermediate layers** or **regularization**.
* This project demonstrates the **end-to-end workflow** in AI Engineering:

  > Data → Preprocessing → Model → Training → Evaluation → Deployment

---
