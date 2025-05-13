import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-6, 6, 100)
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label='Sigmoid: $\sigma(x) = \\frac{1}{1+e^{-x}}$', color='blue', linewidth=2)
plt.plot(x, y_derivative, 
         label="Derivative: $\sigma'(x) = \sigma(x)(1-\sigma(x))$", 
         color='red', linestyle='--', linewidth=2)

plt.title("Sigmoid Function and Its Derivative", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend(fontsize=12)
plt.savefig('sigmoid.png')