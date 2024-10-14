import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Simulación de un dataset
data = {
    'hora_dia': [8, 12, 18, 6, 9, 15, 19, 22],
    'dia_semana': ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo', 'lunes'],
    'cantidad_pasajeros': [200, 340, 400, 120, 450, 390, 500, 220],
    'estacion': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'clima': ['soleado', 'lluvioso', 'soleado', 'nublado', 'nublado', 'lluvioso', 'soleado', 'soleado'],
    'tiempo_espera': [5, 10, 2, 15, 7, 12, 3, 5]
}

df = pd.DataFrame(data)

# Convertir variables categóricas en numéricas
df = pd.get_dummies(df, columns=['dia_semana', 'estacion', 'clima'])

# Definir variables de entrada (X) y de salida (y)
X = df.drop('cantidad_pasajeros', axis=1)
y = df['cantidad_pasajeros']

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar y entrenar el árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Hacer predicciones
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")



plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, filled=True, rounded=True)
plt.show()
