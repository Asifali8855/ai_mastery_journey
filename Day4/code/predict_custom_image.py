import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model = load_model("mnist_digit_model.h5")

# Load and preprocess your handwritten image
image_path = "C:/ai_practice_days/Day4/code/my_digit.png"  # Replace with your image file
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 and invert colors (MNIST is white digit on black bg)
img_resized = cv2.resize(img, (28, 28))
img_inverted = cv2.bitwise_not(img_resized)
img_normalized = img_inverted / 255.0

# Reshape to match model input (1, 28, 28, 1)
img_reshaped = img_normalized.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_reshaped)
predicted_class = np.argmax(prediction)

# Show the result
plt.imshow(img_resized, cmap="gray")
plt.title(f"Predicted Digit: {predicted_class}")
plt.axis("off")
plt.show()
