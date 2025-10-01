import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# X es ahora una MATRIZ de valores, los cuales son: superficie de la casa, numero de habitaciones, piso y antiguedad (en años)
# Podemos apreciar como x es una matriz con 3 array, cada uno con las 4 caracteristicas mencionadas
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])

# x_0(0) = superficie_casa_1, x_1(0) = habitaciones_casa_1, etc..
# x_0(1) = superficie_casa_2, x_1(1) = habitaciones_casa_2, etc..

# para y obtenemos un vector como unico resultado, donde sus componentes serán la prediccion del valor en USD para cada casa
y_train = np.array([460, 232, 178])


# Printeamos los datos de entrada para ver bien la matriz:
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


# En este tipo de regresion, cada columna tiene su propio W y B, por lo que estos valores ahora serán 
# W -> un vector, donde cada elemento del mismo correspondera a cada casa
# B -> un parámetro escalar

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


# La siguiente funcion recibe la matriz con las distintas entradas, el W y el bias de cada elemento, y predice el resultado para cada ejemplo
# SIN EMBARGO, esto es super lento, y exiten formas mas rápidas de hacerlo.
"""
def predict_single_loop(x, w, b):

    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p
"""


# Podemos realizar la misma funcion usando el .dot de numpy:
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
