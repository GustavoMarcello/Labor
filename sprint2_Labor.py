#Dataset utilizado: Labor
#Esse Dataset foi usado para aprender as descrições de contratos aceitos e não aceitos

#*******descobrir como testar e treinar o código, a partir das aulas da alura*********

from re import X
from typing import NamedTuple
from numpy import NaN
from pandas.core.algorithms import mode
import pymongo
from scipy.sparse import data
from sklearn import datasets
import sklearn
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import openml
import pandas as pd
import matplotlib.pyplot as plt

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#conectando ao servidor/conectando ao banco/criando coleção
myClient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myClient["labor"]
myCol = mydb["contratos"]

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#buscando dataset: Labor https://www.openml.org/d/4
dataset = openml.datasets.get_dataset(4)
#printando informações do dataset
#print(dataset)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#adicionando as informações do dataset a variável info
x, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
# print(y)
#isolando coluna standby-py
standby_pay = x['standby-pay']
#print(standby_pay)
#isolando coluna duration
duration = x['duration']
#print(duration)
#isolando classe
classe = y
#print(classe)
#print(classe.shape) #verificando quantidade de dados

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#cria o gráfico
sns.scatterplot(x=standby_pay, y=duration, hue=classe, data=x)
#imprime o gráfico
#plt.show()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#começando treino

#separando dados x e y
raw_dados_x = x[["duration", "standby-pay"]]
# print(raw_dados_x)
raw_dados_y = pd.DataFrame(y) #transformando y de Series para DataFrame
# print(raw_dados_y)

raw_dados_x['class'] = raw_dados_y
print(raw_dados_x)

# #verificando tipos de dados raw_dados_x/raw_dados_y
# print(type(raw_dados_x))
# print('')
# print(type(raw_dados_y))

# #excluindo "standby-pay" = NaN
# dados_x = raw_dados_x.dropna()
# print(dados_x.shape)


# dados_x = []
# for k in len(raw_dados_x):
#     if raw_dados_x["standby-pay"] != NaN:
#         dados_x.append(k)
# print(dados_x)


# #qtde de treino (total 57)
# treino_x = dados_x[:43]
# treino_y = dados_y[:43]
# #qtde de teste (total 57)
# teste_y = dados_x[43:]
# teste_x = dados_y[43:]
# print(f'Treinaremos com {len(treino_x)} e testaremos com {len(teste_x)} elementos')

# #rodando e treinando o modelo
# modelo = LinearSVC()
# modelo.fit(treino_x, treino_y)
# previsoes = modelo.predict(teste_x)
# acuracia = accuracy_score(teste_y, previsoes) * 100
# print(f'A acurácia foi de {acuracia:.2f}')

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

