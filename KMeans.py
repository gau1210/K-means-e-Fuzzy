import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Carrega o datatset
dataset = pd.read_csv('C:\\Users\\glaub\\PycharmProjects\\TrabalhoIA2\\Wholesalecustomersdata.csv', delimiter=";")

print(dataset.dtypes)

dataset['Total Gastos'] = dataset['Fresh'] + dataset['Milk'] + dataset['Grocery'] + dataset['Frozen'] + \
                          dataset['Detergents_Paper'] + dataset['Delicassen']
print(dataset)

X = dataset.values  # Atribui os valores de entrada em X
print(X)  # Printa os dados do dataset

print(dataset.shape)  # Informa a quantidade de dados na base

# Construíndo a máquina preditiva utilizado o algorítimo PCA que -
# transforma as 8 variáveis em 2 principais juntando todas por semelhança.
pca = PCA(n_components=2).fit_transform(dataset)

# Determinando um range do Hyperparamentro do Kmeans
k_range = range(1, 48)
print(k_range)
#Criando 12 máquinas preditivas para treinar e salvar na variável k
k_means_var = [KMeans(n_clusters=k).fit(pca) for k in k_range]

#Curva de Elbow calcula a distáncia Euclidiana
#Ajuste da centróide do cluster para cada modelo
centroids = [X.cluster_centers_ for X in k_means_var]

#Calculando a distância euclidiana de cada pondo para o centróide
k_euclid = [cdist(pca, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]

#Soma dos quadrados das distâncias dentro do cluster
soma_quad_cluster = [sum(d**2) for d in dist]

#soma do total dos quadados
soma_tot_cluster = sum(pdist(pca)**2)/pca.shape[0]

#soma dos quadados entre os clusters
soma_quad_cluster = soma_tot_cluster - soma_quad_cluster

#curva de Elbow
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, soma_quad_cluster/soma_tot_cluster * 100, 'b*-')
ax.set_ylim((0, 100))
plt.grid(True)
plt.xlabel('N° de Clusters')
plt.ylabel('% de Variância')
plt.title('Variância para cada valor de k')
plt.show()

#Criando o modelo com k = 2
modelo_v1 = KMeans(n_clusters=2)
print(modelo_v1.fit(pca))

labels = modelo_v1.labels_
print(silhouette_score(pca, labels, metric='euclidean'))

modelo_v2 = KMeans(n_clusters=3)
print(modelo_v2.fit(pca))

labels = modelo_v2.labels_
print(silhouette_score(pca, labels, metric='euclidean'))

modelo_v3 = KMeans(n_clusters=4)
print(modelo_v3.fit(pca))

labels = modelo_v3.labels_
print(silhouette_score(pca, labels, metric='euclidean'))

modelo_v4 = KMeans(n_clusters=5)
print(modelo_v4.fit(pca))

labels = modelo_v4.labels_
print(silhouette_score(pca, labels, metric='euclidean'))

modelo_v5 = KMeans(n_clusters=6)
print(modelo_v5.fit(pca))

labels = modelo_v5.labels_
print(silhouette_score(pca, labels, metric='euclidean'))

names = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'Total Gastos']

#Inclui o número de clusters no dataset
cluster_map = pd.DataFrame(dataset, columns=names)
cluster_map['Total Gastos'] = pd.to_numeric(cluster_map['Total Gastos'])
cluster_map['cluster'] = modelo_v1.labels_

print(cluster_map)

#calcula a média de gastos separado por cluster
print(cluster_map.groupby('cluster')['Total Gastos'].mean())

