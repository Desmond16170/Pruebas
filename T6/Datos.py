# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Cargar y preparar los datos
df = pd.read_csv("traffic.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'] + ' ' + df['Time'])
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df.drop(['DateTime', 'Time', 'Junction', 'ID'], axis=1, inplace=True)

# Visualizacion de datos para clustering
sns.pairplot(df[['year', 'month', 'day', 'hour', 'Vehicles']])
plt.title('Pairplot for Clustering')
plt.show()

# Implementacion de K-means
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df[['year', 'month', 'day', 'hour', 'Vehicles']])
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Metodo del Codo')
plt.xlabel('Numero de clusters')
plt.ylabel('Inertia')
plt.show()

# Aplicar K-means con el numero optimo de clusters
num_clusters = 3  # Reemplazar con el numero optimo encontrado
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['year', 'month', 'day', 'hour', 'Vehicles']])

# Visualizar los clusters
sns.scatterplot(x='hour', y='Vehicles', hue='cluster', data=df)
plt.show()

# Preparacion para regresion
X = df[['year', 'month', 'day', 'hour']]
y = df['Vehicles']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresion lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
r2 = modelo.score(X_test, y_test)
print("Coeficiente de Determinacion (R2): {}".format(r2))



# Crear una figura para el grafico de regresion
plt.figure()
# Graficar la regresion en la figura
sns.regplot(x=y_test, y=y_pred, data=df, line_kws={"color": "C1"})
# Personaliza la figura segun sea necesario
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Grafico de Regresion')
# Muestra la figura
plt.show()

prediccion = modelo.predict([[2023 ,10,10, 13]])
print("Cantidad estimada de autos: {}".format(prediccion[0]))
