# Pequeña practica para probar un modelo de regresión lineal en python.

# operaciones con arrays y matrices (grandes vol)
import numpy as np

# librería para graficar
import matplotlib.pyplot as plt

# x_train son los valores iniciales (1000 m2) - empezamos con 1000 y 2000 m2
x_train = np.array([1.0, 2.0])
print(f"Miles de m2 - x_train = {x_train}")

# y_train son los valores objetivo de entrenamieento; alimentamos el modelo con los valores iniciales de 300k usd para 1000m2 y 500k para 2000m2.
y_train = np.array([300.0, 500.0])
print(f"Miles de dolares - y_train = {y_train}")

# .shape se utiliza para obtener las dimensiones o tamaño de un array o matriz. 
# En este caso, m toma el tamaño de las columnas (el del array)
print(f"x_train.shape: {x_train.shape}")

# m es el tamaño de la muestra, osea el numero de variables independientes que ingresaremos al modelo
m = x_train.shape[0]
print(f"Number of training examples is: {m}")


# elemento i-esimo de la entrada (cada subindice) - empieza en 0
i = 0 

x_i = x_train[i]  ## 1.0
y_i = y_train[i]  ## 300.0

print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


### Función que simula el comportamiento de la función modelo de regresión lineal
# Calcula los valores de salida del modelo lineal (le pasamos la var independiente, el weight y la bias)
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


# La idea principal es que nuestros valores de entrada 
w = 200
b = 100

# Se asemejen lo maximo posible a los valores esperados y_train
tmp_f_wb = compute_model_output(x_train, w, b,)


# Genera la línea representante de la función
plt.plot(x_train, tmp_f_wb, c='b', label='Predicción')

# Datos de entrada para generar el gráfico
plt.scatter(x_train, y_train, marker='x', c='r', label='Valores reales')

# Titulo del grafico
plt.title("Precio casas")

# Etiqueta del eje Y
plt.ylabel('Precio (en 1000s de dólares)')

# Etiqueta del eje X
plt.xlabel('Superficie (1000 m2)')

plt.legend()
plt.show()


### Encontramos que los valores w=200 y b=100 nos dejan una linea recta que coincide a la perfección con los valores independientes ingresados al principio.
# Ahora si, gracias a este ajuste, podemos empezar a jugar con las variables independientes desconocidas y predecir mas valores a la larga.

# Si quisieramos saber el valor para una casa de 1200m2, entonces:  
x_i = 1.2
cost_1200m2 = w * x_i + b     # Llamamos a la funcion lineal.. con los valores ajustados de w y b

print(f"Una casa de {x_i}k m2 costaría algo de ${cost_1200m2:.0f} mil dólares")

# Para probar distintas predicciones, basta con modificar el valor de x_i