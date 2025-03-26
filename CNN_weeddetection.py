import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Dataset Paths
weed_image_dir = "/content/drive/My Drive/weed"
crop_image_dir = "/content/drive/My Drive/crop"

# Image Size & Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not loaded: {img_path}")
            continue

        # Gaussian Blur for noise reduction
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # HSV Segmentation with Adaptive Range
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 30])    # Expanded lower limit for light green shades
        upper_green = np.array([100, 255, 255]) # Wider upper limit for varying greens

        # Mask creation
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Dilation to expand detected leaf areas
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours and filter based on area size
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_mask = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Retain only larger crop-like regions
                cv2.drawContours(crop_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply refined mask to the original image
        img = cv2.bitwise_and(img, img, mask=crop_mask)

        # Resize and Normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels

# Load dataset
weed_images, weed_labels = load_images_from_folder(weed_image_dir, 1)
crop_images, crop_labels = load_images_from_folder(crop_image_dir, 0)

# Combine and shuffle dataset
X = np.array(weed_images + crop_images)
y = np.array(weed_labels + crop_labels)

indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Split dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Function to visualize predictions
def visualize_predictions(X_test, y_test, model, n_images=10):
    plt.figure(figsize=(15, 12))
    predictions = (model.predict(X_test[:n_images]) > 0.5).astype("int32")

    for i in range(n_images):
        plt.subplot(2, n_images // 2, i + 1)
        plt.imshow(X_test[i])
        predicted_class = "weed" if predictions[i] == 1 else "crop"
        actual_class = "weed" if y_test[i] == 1 else "crop"
        plt.title(f"Predicted: {predicted_class}\nActual: {actual_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Display Test Predictions
visualize_predictions(X_test, y_test, model, n_images=10)
# Save Model
MODEL_PATH = "/content/drive/My Drive/weed_model_v3.h5"
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")