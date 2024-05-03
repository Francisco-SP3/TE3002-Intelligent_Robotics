# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Define parameters
max_depth = 3
min_samples_leaf = 12

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
max_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=max_depth)
min_tree = DecisionTreeRegressor(criterion='squared_error', min_samples_leaf=min_samples_leaf)
max_tree.fit(X, y)
min_tree.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = max_tree.predict(X_test)
y_2 = min_tree.predict(X_test)

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plot first graph on the first subplot
ax1.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
ax1.plot(X_test, y_1, color="cornflowerblue", label="prediction", linewidth=2)
ax1.set_xlabel("data")
ax1.set_ylabel("target")
ax1.set_title(f'Decision Tree Regression (max_depth={max_depth})')
ax1.legend()

# Plot second graph on the second subplot
ax2.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
ax2.plot(X_test, y_2, color="yellowgreen", label="prediction", linewidth=2)
ax2.set_xlabel("data")
ax2.set_ylabel("target")
ax2.set_title(f'Decision Tree Regression (min_samples_leaf={min_samples_leaf})')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Graficar el árbol de decisión con fuente pequeña y alta resolución
plt.figure(figsize=(24, 16))  # Aumenta el tamaño de la figura para mejorar la resolución
plot_tree(max_tree, filled=True, rounded=True, fontsize=8)  # Establece el tamaño de la fuente
plt.title(f'Decision Tree Regression (max_depth={max_depth})')
plt.show()

# Graficar el árbol de decisión con fuente pequeña y alta resolución
plt.figure(figsize=(24, 16))  # Aumenta el tamaño de la figura para mejorar la resolución
plot_tree(min_tree, filled=True, rounded=True, fontsize=8)  # Establece el tamaño de la fuente
plt.title(f'Decision Tree Regression (min_samples_leaf={min_samples_leaf})')
plt.show()

# Guardar la figura con alta resolución
# plt.savefig('regression_tree_high_dpi.png', dpi=300)  # Ajusta el dpi según tu preferencia
