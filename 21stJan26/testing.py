from sklearn.linear_model import LinearRegression
import numpy as np

# Input data (must be 2D)
x = np.array([[1], [2], [3], [4]])
y = np.array([30, 50, 70, 90])

# Create model
model = LinearRegression()

# Train (fit) the model
model.fit(x, y)

# Predict
predicted = model.predict([[5]])
print("Prediction for x=5:", predicted)

# Coefficients
print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)
