
# Regresión Lineal con Gradient Descent
# Dataset: Duración del ejercicio (min) vs Calorías quemadas

import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 


# Dataset: duración del ejercicio (min) y calorías quemadas
x_train = np.array([10, 15, 20, 25, 30])        # Duración en minutos
y_train = np.array([55, 71, 108, 116, 148])     # Calorías quemadas


# Visualización inicial de los datos
plt.scatter(x_train, y_train, color='red', label='Datos')
plt.xlabel('Duración (min)')
plt.ylabel('Calorías quemadas (kcal)')
plt.title('Regresión Lineal: tiempo ejercicio vs calorías quemadas')
plt.legend()
plt.grid()
plt.show()


# Función de costo
def compute_cost(x, y, w, b):
    """Calcula el costo cuadrático medio para regresión lineal"""
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return (1 / (2 * m)) * cost


# Gradientes
def compute_gradient(x, y, w, b):
    """Calcula los gradientes dj_dw y dj_db"""
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
        
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db



# Gradient Descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """Ejecuta el descenso de gradiente"""
    J_history = []
    p_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Actualización de parámetros
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        # Guardar historial de costos
        J_history.append(cost_function(x, y, w, b))
        p_history.append([w, b])
        
        # Mostrar progreso cada cierto número de iteraciones
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteración {i:5}: Costo {J_history[-1]:0.2e} | "
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e} | "
                  f"w: {w:0.4f}, b: {b:0.4f}")
 
    return w, b, J_history, p_history


# Entrenamiento del modelo
now = int(datetime.now().timestamp())
print(f"Seed para números aleatorios: {now}")
np.random.seed(now)

# Parámetros iniciales
w_init = np.random.rand() * 10
b_init = 0
alpha = 1e-3
iterations = 50000


# Ejecución de gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

print(f"(w_final, b_final) encontrados: ({w_final:.4f}, {b_final:.4f})")


# Visualización de predicciones
y_predict = np.dot(x_train, w_final) + b_final

plt.plot(x_train, y_predict, "r-", label="Predicción")
plt.plot(x_train, y_train, "b.", label="Datos")
plt.xlabel('Duración (min)')
plt.ylabel('Calorías quemadas (kcal)')
plt.title('Regresión Lineal: tiempo ejercicio vs calorías quemadas')
plt.legend()
plt.grid()
plt.show()


# Predicción para un caso específico
estimacion = w_final * 28 + b_final
print(f"Estimación para 28 mins de ejercicio: {estimacion:.1f} kcal")
