import pandas as pd
import skfuzzy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Carrega o datatset
dataset = pd.read_csv('C:\\Users\\glaub\\PycharmProjects\\TrabalhoIA2\\Wholesalecustomersdata.csv', delimiter=';')

#Inclui um novo atributo com a soma dos gastos na base de dados
dataset['Total Gastos'] = dataset['Fresh'] + dataset['Frozen'] + dataset['Detergents_Paper'] + dataset['Delicassen']
print(dataset)

#Pré processamento, buscando os registro da base e seleciona dois atributos da base
X = dataset.iloc[:, [1, 8]].values

#Normalização dos dados para diminui os valores das classses e colocando em 0 e 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print("Intevalo da classe normalizada: ", X.min(), X.max())

#Acensando os pacotes e incluíndo os hiperparametros nas variáveis.
agrupamento = skfuzzy.cmeans(data=X.T, c=5, m=2, error=0.005, maxiter=1000, init=None)
print("Tamanho da variável: ", len(agrupamento))
#Variável para incluíndo o primeiro grupo
previsoes_porcentagem = agrupamento[1]

#Acesso dos valores dos grupos individualmente e idintificação das probabilidade do registro 1
previsoes_porcentagem[0][0], previsoes_porcentagem[1][0], previsoes_porcentagem[2][0], previsoes_porcentagem[3][0], \
previsoes_porcentagem[4][0]

#Soma das probabilidades que será = 1
previsoes_porcentagem[0][0] + previsoes_porcentagem[1][0] + previsoes_porcentagem[2][0] + previsoes_porcentagem[3][0] + \
previsoes_porcentagem[4][0]

#Retorna o maior valor que está em cada uma das linhas
previsoes = previsoes_porcentagem.argmax(axis=0)

#Conta os grupos únicos que existem na base
cont_grupo = np.unique(previsoes, return_counts=True)
print("Mostra q quantidade de registros por grupo: ", cont_grupo)

#Gráfico dos grupos gerados
plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], c='red', label='Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], c='green', label='Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], c='blue', label='Cluster 3')
plt.scatter(X[previsoes == 3, 0], X[previsoes == 3, 1], c='yellow', label='Cluster 4')
plt.scatter(X[previsoes == 4, 0], X[previsoes == 4, 1], c='orange', label='Cluster 5')
plt.xlabel('grupos')
plt.ylabel('Gastos')
plt.legend()
plt.show()

colors = ['blue', 'orange', 'green', 'red', 'yellow', 'black', 'brown', 'cyan', 'magenta', 'forestgreen']


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fpcs = []
#Coeficiente de partições fuzzy, identifica o melhor numero de grupos para ser escolhido como parâetro de c.
for n_clusters, ax in enumerate(axes.reshape(-1), 2):

    centroides, previsoes, _, _, _, _, fpc = skfuzzy.cmeans(data=X.T, c=n_clusters, m=2, error=0.005, maxiter=1000,
                                                            init=None)
    fpcs.append(fpc)

    previsoes = np.argmax(previsoes, axis=0)

    for i in range(n_clusters):
        ax.plot(X[previsoes == i, 0], X[previsoes == i, 1], '.', color=colors[i])

    for centro in centroides:
        ax.plot(centro[0], centro[1], 'rs')

    ax.set_title('Centros = {}; FPC = {}'.format(n_clusters, fpc))
    ax.axis('off')
plt.show()

print("Valores de FPC:", fpcs)

#Gera os gráfico de cada FPCs gerados
fig, ax = plt.subplots()
ax.plot(range(1, 10), fpcs)
ax.set_xlabel('Número de clusters')
ax.set_ylabel('Coeficiente de partição difusa')
plt.show()


#Aplicando o número de cluster ideal identificado pela métrica FPC utilizada anteriormente
agrupamento = skfuzzy.cmeans(data=X.T, c=2, m=2, error=0.005, maxiter=1000, init=None)

previsoes_porcentagem = agrupamento[1]
previsoes = previsoes_porcentagem.argmax(axis=0)
previsoes, np.unique(previsoes, return_counts=True)

count_grupo = np.unique(previsoes, return_counts=True)
print("Mostra q quantidade de registros por grupo: ", count_grupo)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s=100, c='orange', label='Cluster 2')
plt.xlabel('cliente')
plt.ylabel('Gastos')
plt.legend()

centroides = agrupamento[0]
centroides = scaler.inverse_transform(centroides)
centroides = pd.DataFrame(data=centroides, columns=['Cliente', 'Gastos'])
print(centroides)
