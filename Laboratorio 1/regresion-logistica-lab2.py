
import numpy as np
import matplotlib.pyplot as plt
import math
from utils.lab_01 import *
from utils.public_tests import *


# Cargamos y visualizar los datos
X_train, y_train = load_data("data/lab_01-2.txt")

print("Primeros 5 elementos de X_train:\n", X_train[:5])
print("Primeros 5 elementos de y_train:\n", y_train[:5])
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Cantidad de ejemplos m =", len(y_train))

# Visualización inicial
plot_data(X_train, y_train[:], pos_label="Admitido", neg_label="No admitido")
plt.ylabel('Puntaje 2do exámen')
plt.xlabel('Puntaje 1er exámen')
plt.legend(loc="upper right")
plt.show()



# Función Sigmoide
def sigmoid(z):
    """
    Calcula la función sigmoide g(z) = 1 / (1 + e^(-z))
    Soporta escalares, vectores y matrices.
    """
    g = 1 / (1 + np.exp(-z))
    return g

# Pruebas
print(f"sigmoid(0) = {sigmoid(0)}")
print("sigmoid([-1, 0, 1, 2]) =", sigmoid(np.array([-1, 0, 1, 2])))
sigmoid_test(sigmoid)


# Función de costo para regresión logística
def compute_cost(X, y, w, b, *argv):
    """
    Calcula el costo promedio J(w,b) de la regresión logística
    """
    m, n = X.shape
    total_cost = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        total_cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    total_cost = total_cost / m
    return total_cost


# Pruebas de costo
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.
print('Costo inicial (ceros):', compute_cost(X_train, y_train, initial_w, initial_b))

test_w = np.array([0.2, 0.2])
test_b = -24.
print('Costo con w,b prueba:', compute_cost(X_train, y_train, test_w, test_b))
compute_cost_test(compute_cost)



# Gradiente para regresión logística
def compute_gradient(X, y, w, b, *argv): 
    """
    Calcula los gradientes de J(w,b) respecto a w y b
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        diff = f_wb - y[i]

        dj_db += diff
        dj_dw += diff * X[i]

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw



# Pruebas del gradiente
dj_db, dj_dw = compute_gradient(X_train, y_train, np.zeros(n), 0.)
print("Gradiente inicial (zeros):", dj_db, dj_dw.tolist())
compute_gradient_test(compute_gradient)



# Descenso por el gradiente
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Entrena el modelo por descenso del gradiente
    """
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in)
            J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w_in, b_in, J_history, w_history


# Entrenamiento del modelo
from datetime import datetime
now = int(datetime.now().timestamp())
np.random.seed(now)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

print(f"Seed: {now}")
print(f"Parámetros iniciales: {initial_w}, {initial_b}")

iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                      compute_cost, compute_gradient, alpha, iterations, 0)



# Frontera de decisión
plot_decision_boundary(w, b, X_train, y_train)
plt.ylabel('Puntaje exámen nr 2')
plt.xlabel('Puntaje exámen nr 1')
plt.legend(loc="upper right")
plt.show()


# Predicciones y precisión
def predict(X, w, b): 
    """
    Realiza predicciones binarias (0 o 1) según un umbral de 0.5
    """
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):   
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0
    
    return p

p = predict(X_train, w, b)
accuracy = np.mean(p == y_train) * 100
print(f'Precisión entrenamiento (aprox): {accuracy:.2f}%')
