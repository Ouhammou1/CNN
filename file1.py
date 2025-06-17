import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import numpy as np
from PIL import Image

# --- Configuration ---
# Set the base directory where your 'real_vs_fake' folder is located
BASE_DIR = 'real-vs-fake' # Adjust this path if your 'real_vs_fake' is elsewhere
IMAGE_SIZE = (128, 128) # Resize images to this size (e.g., 128x128 pixels)
BATCH_SIZE = 32
EPOCHS = 10 # You might need more epochs for better accuracy
MODEL_FILENAME = 'model.h5' # The name of your pre-saved model file

# --- 1. Data Preparation using ImageDataGenerator ---
# ImageDataGenerator is great for loading images from directories and performing augmentation

print("Setting up data generators...")

# Data augmentation for training data to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation and test data (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # 'binary' because we have two classes: real/fake
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'valid'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Keep data in order for evaluation
)

# Check class indices (0 for 'fake', 1 for 'real' or vice-versa)
print(f"Class indices: {train_generator.class_indices}")
# You'll likely see {'fake': 0, 'real': 1} or {'real': 0, 'fake': 1} based on alphabetical order.

# --- 2. Model Definition (If you want to train a new model) ---
def create_cnn_model():
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten the 3D output to 1D
        Flatten(),

        # Dense (Fully Connected) Layer 1
        Dense(512, activation='relu'),
        Dropout(0.5), # Dropout for regularization to prevent overfitting

        # Output Layer (binary classification)
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer='adam', # Adam optimizer is a good default
        loss='binary_crossentropy', # Binary crossentropy for binary classification
        metrics=['accuracy'] # Track accuracy during training
    )
    return model

# --- 3. Model Training or Loading ---

model = None
if os.path.exists(MODEL_FILENAME):
    print(f"\n--- Loading existing model from {MODEL_FILENAME} ---")
    try:
        model = load_model(MODEL_FILENAME)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}. Training a new model instead.")
        model = create_cnn_model()
        print("\n--- Training new model ---")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator
        )
        model.save(MODEL_FILENAME) # Save the newly trained model
        print(f"New model saved as {MODEL_FILENAME}")
else:
    print("\n--- No existing model found. Training a new model ---")
    model = create_cnn_model()
    model.summary() # Print model summary
    print("\n--- Training new model ---")
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    # Save the trained model
    model.save(MODEL_FILENAME)
    print(f"New model saved as {MODEL_FILENAME}")

# --- 4. Model Evaluation ---
print("\n--- Evaluating the model on the test set ---")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 5. Making Predictions on new images ---
print("\n--- Making predictions ---")

# Function to predict a single image
def predict_single_image(model, image_path, target_size):
    try:
        img = Image.open(image_path).convert('RGB') # Open image and ensure RGB format
        img = img.resize(target_size) # Resize to target size
        img_array = np.array(img) # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Normalize pixel values

        prediction = model.predict(img_array)
        # Assuming 0 is 'fake' and 1 is 'real' based on alphabetical sorting by flow_from_directory
        predicted_class = 'real' if prediction[0][0] >= 0.5 else 'fake'
        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else (1 - prediction[0][0])
        return predicted_class, confidence
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

# Example usage for prediction:
# Create dummy paths for prediction examples if the user doesn't provide real ones
example_real_path = os.path.join(BASE_DIR, 'test', 'real', os.listdir(os.path.join(BASE_DIR, 'test', 'real'))[0]) if os.path.exists(os.path.join(BASE_DIR, 'test', 'real')) and os.listdir(os.path.join(BASE_DIR, 'test', 'real')) else None
example_fake_path = os.path.join(BASE_DIR, 'test', 'fake', os.listdir(os.path.join(BASE_DIR, 'test', 'fake'))[0]) if os.path.exists(os.path.join(BASE_DIR, 'test', 'fake')) and os.listdir(os.path.join(BASE_DIR, 'test', 'fake')) else None

if example_real_path:
    print(f"\n--- Prediction for a real image example: {example_real_path} ---")
    predicted_class, confidence = predict_single_image(model, example_real_path, IMAGE_SIZE)
    if predicted_class:
        print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}")

if example_fake_path:
    print(f"\n--- Prediction for a fake image example: {example_fake_path} ---")
    predicted_class, confidence = predict_single_image(model, example_fake_path, IMAGE_SIZE)
    if predicted_class:
        print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}")

if not example_real_path and not example_fake_path:
    print("\nCould not find example images for prediction. Please ensure your test directories contain images.")
    print("To predict on a new image, replace 'path/to/your/new_image.jpg' with the actual path:")
    print("    predicted_class, confidence = predict_single_image(model, 'path/to/your/new_image.jpg', IMAGE_SIZE)")

