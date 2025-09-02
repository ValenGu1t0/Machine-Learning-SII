# probando python

# compilamos con python <archivo.py>

# --- Variables y tipos básicos ---
entero = 10
decimal = 3.14
texto = "Hola Python"
booleano = True

print("Entero:", entero)
print("Decimal:", decimal)
print("Texto:", texto)
print("Booleano:", booleano)

# --- Listas / Arrays ---
numeros = [1, 2, 3, 4, 5]
print("Lista:", numeros)
print("Primer elemento:", numeros[0])

# --- Diccionarios / Objetos ---
persona = {"nombre": "Valentino", "edad": 22}
print("Diccionario:", persona)
print("Nombre:", persona["nombre"])

# --- Condicionales ---
if entero > 5:
    print("El número es mayor que 5")
else:
    print("El número es 5 o menor")

# --- Bucles ---
print("Ciclo for sobre lista:")
for num in numeros:
    print(num)

print("Ciclo while:")
i = 0
while i < 3:
    print("i =", i)
    i += 1

# --- Funciones ---
def saludar(nombre):
    return f"Hola, {nombre}!"       # f es un formatted, como ${} en js

print(saludar("Valentino"))

# --- Entrada de usuario ---
usuario = input("¿Cómo te llamas? ")
print(saludar(usuario))
