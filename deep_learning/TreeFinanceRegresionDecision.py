import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Supongamos que tienes un conjunto de datos de precios de acciones en un DataFrame de pandas
# con las características relevantes para la predicción, como volumen de transacciones, 
# precios históricos, noticias financieras, etc.

volumen_min = 10
volumen_max = 40

# Aquí creamos un DataFrame de ejemplo para ilustrar el proceso
np.random.seed(1)
data = {
    'Volumen': np.random.randint(volumen_min, volumen_max, 100),
    'PrecioAnterior': np.random.uniform(50, 100, 100),
    'NoticiasPositivas': np.random.randint(0, 2, 100)  # Supongamos que esta es una variable binaria
}
df = pd.DataFrame(data)

# Agregamos la columna de PrecioActual (lo que queremos predecir)
df['PrecioActual'] = df['PrecioAnterior'] + np.random.uniform(-10, 10, 100)

# Dividimos los datos en conjuntos de entrenamiento y prueba (no se utiliza en este enfoque)
X = df.drop('PrecioActual', axis=1)
y = df['PrecioActual']

# Función para calcular sum squared residuals
def sum_squared_residuals(x, y):
    mean_y = np.mean(y)
    ssr = np.sum((y - mean_y) ** 2)
    return ssr

# Función para encontrar el mejor candidato para la raíz de un predictor
def find_best_threshold(feature, target):
    unique_values = np.unique(feature)
    thresholds = []  # Lista para almacenar los thresholds analizados
    best_ssr = float('inf')
    best_threshold = None
    for value in unique_values:
        left_indices = feature <= value
        right_indices = feature > value
        left_ssr = sum_squared_residuals(target[left_indices], target[left_indices])
        right_ssr = sum_squared_residuals(target[right_indices], target[right_indices])
        total_ssr = left_ssr + right_ssr
        
        # Almacenar el threshold analizado en la lista
        thresholds.append((value, total_ssr))

        if total_ssr < best_ssr:
            best_ssr = total_ssr
            best_threshold = value
            
    # Devolver el mejor threshold y la lista de thresholds analizados
    return best_threshold, thresholds, best_ssr

# Diccionario para almacenar los valores de total SSR de cada predictor
total_ssr_dict = {}

# Iteración sobre cada predictor
for predictor in X.columns:
    threshold, thresholds, total_ssr = find_best_threshold(X[predictor], y)
    print(f"Mejor threshold para {predictor}: {threshold}")
    print(f"Total Sum Squared Residuals para {predictor}: {total_ssr}")

    # Guardar el valor de total SSR en el diccionario
    total_ssr_dict[predictor] = total_ssr

    # Obtener los valores de threshold y sum squared residuals
    threshold_values, ssr_values = zip(*thresholds)

    # Crear subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Graficar puntos y el mejor threshold
    ax1.scatter(X[predictor], y, color='blue', label='Data Points')
    ax1.axvline(x=threshold, color='red', linestyle='--', label='Best Threshold')
    ax1.set_xlabel(predictor)
    ax1.set_ylabel('Precio Actual')
    ax1.set_title(f'Análisis de Threshold para {predictor}')
    ax1.legend()

    # Graficar thresholds analizados
    ax2.plot(threshold_values, ssr_values, marker='o', color='blue')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Sum Squared Residuals')
    ax2.set_title(f'Thresholds Analizados para {predictor}')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Ordenar el diccionario de total SSR de menor a mayor
sorted_total_ssr_dict = {k: v for k, v in sorted(total_ssr_dict.items(), key=lambda item: item[1])}
print("Total SSR de cada predictor (de menor a mayor), ESTE SERA EL ORDEN DE NUESTRO ARBOL:")
for predictor, total_ssr in sorted_total_ssr_dict.items():
    print(f"{predictor}: {total_ssr}")

# Especificar la profundidad del árbol
profundidad_arbol = 20

# Crear un árbol de decisión de regresión
regression_tree = DecisionTreeRegressor(max_depth=profundidad_arbol)

# Entrenar el árbol utilizando los datos
regression_tree.fit(X, y)

# Graficar el árbol de decisión con fuente pequeña y alta resolución
plt.figure(figsize=(24, 16))  # Aumenta el tamaño de la figura para mejorar la resolución
plot_tree(regression_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=6)  # Establece el tamaño de la fuente
plt.title(f'Árbol de Decisión de Regresión (Profundidad {profundidad_arbol})')

# Guardar la figura con alta resolución
plt.savefig('regression_tree_high_dpi.png', dpi=300)  # Ajusta el dpi según tu preferencia
plt.show()
