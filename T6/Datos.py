import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ruta del archivo csv
df = pd.read_csv("trafico.csv")

# Eliminar columnas innecesarias
df.drop(["Operating airline IATA Code",
        "Published Airline IATA Code", "Activity Type Code", "Cargo Type Code"],
        axis=1, inplace=True)

# Histograma de una columna
plt.figure(figsize=(10, 6))
plt.hist(df['Cargo Weight LBS'], bins=30, color='blue', edgecolor='black')
plt.title('Distribucion del Peso de Carga (LBS)')
plt.xlabel('Peso de Carga (LBS)')
plt.ylabel('Frecuencia')
plt.show()

# Grafico de dispersion entre dos variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cargo Weight LBS', y='Cargo Metric TONS', data=df)
plt.title('Relacion entre Peso de Carga en LBS y Toneladas Metricas')
plt.xlabel('Peso de Carga (LBS)')
plt.ylabel('Peso de Carga (Toneladas Metricas)')
plt.show()



silhouette_coefficients = []
for k in range(2, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df[['Cargo Weight LBS', 'Cargo Metric TONS']])
        score = silhouette_score(df[['Cargo Weight LBS', 'Cargo Metric TONS']], kmeans.labels_)
        silhouette_coefficients.append(score)

# Graficar los coeficientes de silueta
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Numero de Clusters")
plt.ylabel("Coeficiente de Silueta")
plt.show()

# asumiendo que el numero optimo de clusters es 'n', reemplaza 'n' con el numero correspondiente
kmeans = KMeans(n_clusters=n)
df['Cluster'] = kmeans.fit_predict(df[['Cargo Weight LBS', 'Cargo Metric TONS']])

# Visualizar los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cargo Weight LBS', y='Cargo Metric TONS', hue='Cluster', data=df, palette='viridis')
plt.title('Clustering K-means de Peso de Carga')
plt.xlabel('Peso de Carga (LBS)')
plt.ylabel('Peso de Carga (Toneladas Metricas)')
plt.show()

# Preparar los datos para la regresion
X = df[['Cargo Weight LBS']]
y = df['Cargo Metric TONS']

# Crear y entrenar el modelo de regresion lineal
model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)
r2 = model.score(X, y)

# Graficar la regresion lineal
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cargo Weight LBS', y='Cargo Metric TONS', data=df)
plt.plot(X, y_pred, color='red')
plt.title('Regresion Lineal de Peso de Carga')
plt.xlabel('Peso de Carga (LBS)')
plt.ylabel('Peso de Carga (Toneladas Metricas)')
plt.show()


print("Coeficiente de Determinacion (R2): {}".format(r2))