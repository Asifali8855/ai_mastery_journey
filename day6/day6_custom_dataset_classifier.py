# Day 6: Neural Network on Custom Dataset
# =======================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load the custom dataset (CSV file)
data = pd.read_csv('student_data.csv')

# 2. Inspect the dataset (optional)
print(data.head())

# 3. Separate features (X) and target label (y)
X = data[['Hours_Studied', 'Class_Participation', 'Sleep_Hours']].values
y = data['Pass'].values

# 4. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature scaling (normalize the input data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build the neural network model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 3 features
    keras.layers.Dense(8, activation='relu'),                                    # Hidden layer
    keras.layers.Dense(1, activation='sigmoid')                                  # Output layer (binary classification)
])

# 7. Compile the model (optimizer, loss function, metrics)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 8. Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# 9. Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# 10. Visualize training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 11. Predict on new student data (example)
new_student = np.array([[5, 6, 7]])  # Hours_Studied, Class_Participation, Sleep_Hours
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)

# Convert probability to class (0 or 1)
predicted_class = (prediction >= 0.5).astype(int)
print("\nPrediction (Pass=1/Fail=0):", predicted_class[0][0])
