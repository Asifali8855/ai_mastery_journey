# student_pass_predictor.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Create dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 2. Split features and label
X = df[['Hours_Studied']]
y = df['Pass']

# 3. Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and print results
predictions = model.predict(X_test)
print("Predicted values:", predictions)
print("Actual values:   ", list(y_test))

# 6. Visualize result
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, predictions, color='red', label='Prediction Line')
plt.xlabel('Hours Studied')
plt.ylabel('Pass (0 or 1)')
plt.title('Student Pass Prediction Model')
plt.legend()
plt.show()
