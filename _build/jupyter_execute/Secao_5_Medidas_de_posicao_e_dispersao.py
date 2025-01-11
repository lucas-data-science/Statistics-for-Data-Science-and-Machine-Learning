#!/usr/bin/env python
# coding: utf-8

# https://www.udemy.com/course/estatistica-para-ciencia-de-dados-machine-learning/l

# # Medidas de posição e dispersão

# ## Base de dados

# In[1]:


import numpy as np
import statistics
from scipy import stats
import math


# In[2]:


dados = np.array([150, 151, 152, 152, 153, 154, 155, 155, 155, 155, 156, 156, 156,
                  157, 158, 158, 160, 160, 160, 160, 160, 161, 161, 161, 161, 162,
                  163, 163, 164, 164, 164, 165, 166, 167, 168, 168, 169, 170, 172,
                  173])


# ## Média aritmética simples
#  
# ![image.png](attachment:image.png) 

# In[3]:


dados.sum() / len(dados)


# In[4]:


dados.mean()


# In[5]:


statistics.mean(dados)


# ## Moda

# In[6]:


statistics.mode(dados)


# In[7]:


stats.mode(dados)


# ## Mediana

# In[8]:


dados_impar = [150, 151, 152, 152, 153, 154, 155, 155, 155]


# ### Cálculo manual (ímpar)

# In[9]:


posicao = len(dados_impar) / 2
posicao


# In[10]:


posicao = math.ceil(posicao)
posicao


# In[11]:


dados_impar[posicao - 1]


# ### Cálculo manual (par)

# In[12]:


posicao = len(dados) // 2
posicao


# In[13]:


dados[posicao - 1], dados[posicao]


# In[14]:


mediana = (dados[posicao - 1] + dados[posicao]) / 2
mediana


# ### Bibliotecas

# In[15]:


np.median(dados_impar)


# In[16]:


np.median(dados)


# In[17]:


statistics.median(dados_impar)


# In[18]:


statistics.median(dados)


# ## Média aritmética ponderada
# 
# ![image.png](attachment:image.png)

# In[19]:


notas = np.array([9, 8, 7, 3])
pesos = np.array([1, 2, 3, 4])


# In[20]:


(9 * 1 + 8 * 2 + 7 * 3 + 3 * 4) / (1 + 2 + 3 + 4)


# In[21]:


media_ponderada = (notas * pesos).sum() / pesos.sum()
media_ponderada


# In[22]:


np.average(notas, weights=pesos)


# ## Média aritmética, moda e mediana com distribuição de frequência (dados agrupados)

# In[23]:


dados_2 = {'inferior': [150, 154, 158, 162, 166, 170],
         'superior': [154, 158, 162, 166, 170, 174],
         'fi': [5, 9, 11, 7, 5, 3]}


# In[24]:


import pandas as pd
dataset = pd.DataFrame(dados_2)
dataset


# In[25]:


dataset['xi'] = (dataset['superior'] + dataset['inferior']) / 2
dataset


# In[26]:


dataset['fi.xi'] = dataset['fi'] * dataset['xi']
dataset


# In[27]:


dataset['Fi'] = 0
dataset


# In[28]:


frequencia_acumulada = []
somatorio = 0
for linha in dataset.iterrows():
  #print(linha[1])
  #print(linha[1][2])
  somatorio += linha[1][2]
  frequencia_acumulada.append(somatorio)


# In[29]:


frequencia_acumulada


# In[30]:


dataset['Fi'] = frequencia_acumulada
dataset


# ### Média

# In[31]:


dataset['fi'].sum(), dataset['fi.xi'].sum()


# In[32]:


dataset['fi.xi'].sum() / dataset['fi'].sum()


# ### Moda

# In[33]:


dataset['fi'].max()


# In[34]:


dataset[dataset['fi'] == dataset['fi'].max()]


# In[35]:


dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]


# ### Mediana

# In[36]:


dataset


# In[37]:


fi_2 = dataset['fi'].sum() / 2
fi_2


# In[38]:


limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
for linha in dataset.iterrows():
  #print(linha)
  limite_inferior = linha[1][0]
  frequencia_classe = linha[1][2]
  id_frequencia_anterior = linha[0]
  if linha[1][5] >= fi_2:
    id_frequencia_anterior -= 1
    break


# In[39]:


limite_inferior, frequencia_classe, id_frequencia_anterior


# In[40]:


Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
Fi_anterior


# In[41]:


mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classe
mediana


# ### Função completa

# In[42]:


def get_estatisticas(dataframe):
  media = dataset['fi.xi'].sum() / dataset['fi'].sum()
  moda = dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]

  fi_2 = dataset['fi'].sum() / 2
  limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
  for i, linha in enumerate(dataset.iterrows()):
    limite_inferior = linha[1][0]
    frequencia_classe = linha[1][2]
    id_frequencia_anterior = linha[0]
    if linha[1][5] >= fi_2:
      id_frequencia_anterior -= 1
      break
  Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
  mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classe

  return media, moda, mediana


# In[43]:


get_estatisticas(dataset)


# ## Média geométrica, harmônica e quadrática

# ### Média geométrica
# 
# Média geométrica de um conjunto de dados $ \{a_{1},a_{2},\ldots ,a_{n}\}}\{a_{1},a_{2},\ldots ,a_{n}$:
# 
# $$ \left(\prod _{i=1}^{n}a_{i}\right)^{1/n}={\sqrt[{n}]{a_{1}a_{2}\cdots a_{n}}}.\left(\prod _{{i=1}}^{n}a_{i}\right)^{{1/n}}={\sqrt[ {n}]{a_{1}a_{2}\cdots a_{n}}}.$$

# In[44]:


from scipy.stats import gmean


# In[45]:


gmean(dados)


# ### Média harmônica
# 
# $${\displaystyle {\bar {h}}={\frac {n}{{\frac {1}{x_{1}}}+{\frac {1}{x_{2}}}+\cdots +{\frac {1}{x_{n}}}}}={\frac {n}{\sum _{i=1}^{n}{\frac {1}{x_{i}}}}}={\frac {n\cdot \prod _{j=1}^{n}x_{j}}{\sum _{i=1}^{n}{\frac {\prod _{j=1}^{n}x_{j}}{x_{i}}}}}.}$$

# In[46]:


from scipy.stats import hmean


# In[47]:


hmean(dados)


# ### Média quadrática
# 
# $${  x_{q}={\sqrt {\frac {x_{1}^{2}+x_{2}^{2}+\ldots +x_{n}^{2}}{n}}}\,}$$

# In[48]:


def quadratic_mean(dados):
  return math.sqrt(sum(n * n for n in dados) / len(dados))


# In[49]:


quadratic_mean(dados)


# ## Quartis

# In[50]:


dados_impar = [150, 151, 152, 152, 153, 154, 155, 155, 155]


# ### Cálculo manual

# In[51]:


np.median(dados_impar)


# In[52]:


posicao_mediana = math.floor(len(dados_impar) / 2)
posicao_mediana


# In[53]:


esquerda = dados_impar[0:posicao_mediana]
esquerda


# In[54]:


np.median(esquerda)


# In[55]:


direita = dados_impar[posicao_mediana + 1:]
direita


# In[56]:


np.median(direita)


# ### Bibliotecas

# #### numpy

# In[57]:


np.quantile(dados_impar, 0.5)


# In[58]:


np.quantile(dados_impar, 0.75)


# In[59]:


np.quantile(dados_impar, 0.25)


# In[60]:


esquerda2 = dados_impar[0:posicao_mediana + 1]
esquerda2


# In[61]:


np.median(esquerda2)


# In[62]:


np.quantile(dados, 0.25), np.quantile(dados, 0.50), np.quantile(dados, 0.75)


# #### scipy

# In[63]:


stats.scoreatpercentile(dados, 25), stats.scoreatpercentile(dados, 50), stats.scoreatpercentile(dados, 75)


# #### pandas

# In[64]:


dataset


# In[65]:


dataset.quantile([0.25, 0.5, 0.75])


# In[66]:


dataset.describe()


# ## Quartis com distribuição de frequência (dados agrupados)

# In[67]:


dataset


# In[68]:


def get_quartil(dataset, q1=True):
    if q1 == True:
        fi_4 = dataset['fi'].sum() / 4
    else:
        fi_4 = (3 * dataset['fi'].sum()) / 4

    limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
    for linha in dataset.iterrows():
        limite_inferior = linha[1][0]
        frequencia_classe = linha[1][2]
        id_frequencia_anterior = linha[0]
        if linha[1][5] >= fi_4:
            id_frequencia_anterior -= 1
            break
    Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
    q = limite_inferior + ((fi_4 - Fi_anterior) * 4) / frequencia_classe

    return q


# In[69]:


get_quartil(dataset), get_quartil(dataset, q1 = False)


# ## Percentis

# In[70]:


np.median(dados)


# In[71]:


np.quantile(dados, 0.5)


# In[72]:


np.percentile(dados, 50)


# In[73]:


np.percentile(dados, 5), np.percentile(dados, 10), np.percentile(dados, 90)


# In[74]:


stats.scoreatpercentile(dados, 5), stats.scoreatpercentile(dados, 10), stats.scoreatpercentile(dados, 90)


# In[75]:


import pandas as pd
dataset = pd.DataFrame(dados)
dataset.head()


# In[76]:


dataset.quantile([0.05, 0.10, 0.90])


# ## Exercício

# In[77]:


dataset_2 = pd.read_csv('data/census.csv')


# In[78]:


dataset_2.head()


# In[79]:


dataset_2['age'].mean()


# In[80]:


stats.hmean(dataset_2['age'])


# In[81]:


from scipy.stats.mstats import gmean
gmean(dataset_2['age'])


# In[82]:


quadratic_mean(dataset_2['age'])


# In[83]:


dataset_2['age'].median()


# In[84]:


statistics.mode(dataset_2['age'])


# ## Medidas de dispersão

# ### Amplitude total e diferença interquartil

# In[85]:


dados


# In[86]:


dados.max() - dados.min()


# In[87]:


q1 = np.quantile(dados, 0.25)
q3 = np.quantile(dados, 0.75)
q1, q3


# In[88]:


diferenca_interquartil = q3 - q1
diferenca_interquartil


# In[89]:


inferior = q1 - (1.5 * diferenca_interquartil)
inferior


# In[90]:


superior = q3 + (1.5 * diferenca_interquartil)
superior


# ### Variância, desvio padrão e coeficiente de variação

# In[91]:


dados_impar = np.array([150, 151, 152, 152, 153, 154, 155, 155, 155])


# #### Cálculo manual

# In[92]:


media = dados_impar.sum() / len(dados_impar)
media


# In[93]:


desvio = abs(dados_impar - media)
desvio


# In[94]:


desvio = desvio ** 2
desvio


# In[95]:


soma_desvio = desvio.sum()
soma_desvio


# In[96]:


v = soma_desvio / len(dados_impar)
v


# In[97]:


dp = math.sqrt(v)
dp


# In[98]:


cv = (dp / media) * 100
cv


# In[99]:


def get_variancia_desvio_padrao_coeficiente(dataset):
  media = dataset.sum() / len(dataset)
  desvio = abs(dados_impar - media)
  desvio = desvio ** 2
  soma_desvio = desvio.sum()
  variancia = soma_desvio / len(dados_impar)
  dp = math.sqrt(variancia)
  return variancia, dp, (dp / media) * 100


# In[100]:


get_variancia_desvio_padrao_coeficiente(dados_impar)


# #### Bibliotecas

# In[101]:


np.var(dados_impar)


# In[102]:


np.std(dados_impar)


# In[103]:


np.var(dados)


# In[104]:


np.std(dados)


# In[105]:


statistics.variance(dados)


# In[106]:


statistics.stdev(dados)


# In[107]:


from scipy import ndimage
ndimage.variance(dados)


# In[108]:


stats.tstd(dados, ddof = 0)


# In[109]:


stats.variation(dados_impar) * 100


# In[110]:


stats.variation(dados) * 100


# ### Desvio padrão com dados agrupados

# In[111]:


dataset


# In[112]:


dataset['xi_2'] = dataset['xi'] * dataset['xi']
dataset


# In[333]:


dataset['fi_xi_2'] = dataset['fi'] * dataset['xi_2']
dataset


# In[334]:


dataset.columns


# In[335]:


colunas_ordenadas = ['inferior', 'superior', 'fi', 'xi', 'fi.xi', 'xi_2', 'fi_xi_2', 'Fi']


# In[336]:


dataset = dataset[colunas_ordenadas]
dataset


# In[340]:


dp = math.sqrt(dataset['fi_xi_2'].sum() / dataset['fi'].sum() - math.pow(dataset['fi.xi'].sum() / dataset['fi'].sum(), 2))
dp


# ## Testes com algoritmos de classificação

# In[341]:


import pandas as pd
dataset = pd.read_csv('data/credit_data.csv')


# In[342]:


dataset.dropna(inplace=True)
dataset.shape


# In[343]:


dataset


# In[344]:


X = dataset.iloc[:, 1:4].values
X


# In[345]:


y = dataset.iloc[:, 4].values
y


# In[346]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[347]:


resultados_naive_bayes = []
resultados_logistica = []
resultados_forest = []
for i in range(30):
  X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2,
                                                                    stratify = y, random_state = i)
  naive_bayes = GaussianNB()
  naive_bayes.fit(X_treinamento, y_treinamento)
  resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

  logistica = LogisticRegression()
  logistica.fit(X_treinamento, y_treinamento)
  resultados_logistica.append(accuracy_score(y_teste, logistica.predict(X_teste)))

  random_forest = RandomForestClassifier()
  random_forest.fit(X_treinamento, y_treinamento)
  resultados_forest.append(accuracy_score(y_teste, random_forest.predict(X_teste)))


# In[348]:


print(resultados_naive_bayes)


# In[349]:


print(resultados_logistica)


# In[350]:


print(resultados_forest)


# In[351]:


type(resultados_naive_bayes)


# In[353]:


resultados_naive_bayes = np.array(resultados_naive_bayes)
resultados_logistica = np.array(resultados_logistica)
resultados_forest = np.array(resultados_forest)


# In[354]:


type(resultados_naive_bayes)


# ### Média

# In[355]:


resultados_naive_bayes.mean(), resultados_logistica.mean(), resultados_forest.mean()


# ### Moda

# In[356]:


statistics.mode(resultados_naive_bayes)


# In[357]:


stats.mode(resultados_naive_bayes), stats.mode(resultados_logistica), stats.mode(resultados_forest)


# ### Mediana

# In[358]:


np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resultados_forest)


# ### Variância

# In[359]:


np.set_printoptions(suppress=True)
np.var(resultados_naive_bayes), np.var(resultados_logistica), np.var(resultados_forest)


# In[360]:


np.min([8.756250000000001e-05, 0.00020933333333333337, 2.9229166666666637e-05])


# In[361]:


np.max([8.756250000000001e-05, 0.00020933333333333337, 2.9229166666666637e-05])


# In[362]:


resultados_forest


# ### Desvio padrão

# In[363]:


np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resultados_forest)


# ### Coeficiente de variação

# In[364]:


stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resultados_forest) * 100


# ### Exercício: validação cruzada

# In[365]:


from sklearn.model_selection import cross_val_score, KFold


# In[366]:


resultados_naive_bayes_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    naive_bayes = GaussianNB()
    scores = cross_val_score(naive_bayes, X, y, cv=kfold)
    resultados_naive_bayes_cv.append(scores.mean())

    logistica = LogisticRegression()
    scores = cross_val_score(logistica, X, y, cv=kfold)
    resultados_logistica_cv.append(scores.mean())

    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, X, y, cv=kfold)
    resultados_forest_cv.append(scores.mean())


# In[367]:


scores, 10 * 30


# In[368]:


scores.mean()


# In[369]:


print(resultados_naive_bayes_cv)


# In[370]:


print(resultados_logistica_cv)


# In[371]:


print(resultados_forest_cv)


# In[372]:


stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resultados_forest) * 100


# In[373]:


stats.variation(resultados_naive_bayes_cv) * 100, stats.variation(resultados_logistica_cv) * 100, stats.variation(resultados_forest_cv) * 100


# ### Seleção de atributos utilizando variância

# In[374]:


np.random.rand(50)


# In[375]:


np.random.randint(0, 2)


# In[376]:


base_selecao = {'a': np.random.rand(20),
                'b': np.array([0.5] * 20),
                'classe': np.random.randint(0, 2, size = 20)}


# In[377]:


base_selecao


# In[378]:


dataset = pd.DataFrame(base_selecao)
dataset.head()


# In[379]:


dataset.describe()


# In[380]:


math.sqrt(0.08505323963215053)


# In[381]:


np.var(dataset['a']), np.var(dataset['b'])


# In[382]:


X = dataset.iloc[:, 0:2].values
X


# In[383]:


from sklearn.feature_selection import VarianceThreshold


# In[384]:


selecao = VarianceThreshold(threshold=0.07)
X_novo = selecao.fit_transform(X)


# In[385]:


X_novo, X_novo.shape


# In[386]:


selecao.variances_


# In[387]:


indices = np.where(selecao.variances_ > 0.07)
indices


# #### Exercício seleção de atributos utilizando variância

# In[388]:


dataset = pd.read_csv('data/credit_data.csv')


# In[389]:


dataset.dropna(inplace=True)


# In[390]:


dataset.head()


# In[391]:


dataset.describe()


# In[392]:


X = dataset.iloc[:, 1:4].values
X


# In[393]:


y = dataset.iloc[:, 4].values
y


# In[394]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[395]:


X


# In[396]:


selecao = VarianceThreshold(threshold=0.027)
X_novo = selecao.fit_transform(X)


# In[397]:


X_novo


# In[398]:


np.var(X[0]), np.var(X[1]), np.var(X[2])


# In[399]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
naive_sem_selecao = GaussianNB()
naive_sem_selecao.fit(X, y)
previsoes = naive_sem_selecao.predict(X)
accuracy_score(previsoes, y)


# In[400]:


naive_com_selecao = GaussianNB()
naive_com_selecao.fit(X_novo, y)
previsoes = naive_com_selecao.predict(X_novo)
accuracy_score(previsoes, y)


# ## Valores faltantes com média e moda

# ### Média

# In[401]:


import pandas as pd
dataset = pd.read_csv('data/credit_data.csv')


# In[402]:


dataset.isnull().sum()


# In[403]:


nulos = dataset[dataset.isnull().any(axis=1)]
nulos


# In[404]:


dataset['age'].mean(), dataset['age'].median()


# In[405]:


dataset['age'] = dataset['age'].replace(to_replace = np.nan, value = dataset['age'].mean())


# In[406]:


dataset[dataset.isnull().any(axis=1)]


# In[407]:


dataset


# ### Moda

# In[409]:


dataset = pd.read_csv('data/autos.csv', encoding='ISO-8859-1')


# In[410]:


dataset.head()


# In[411]:


dataset.isnull().sum()


# In[412]:


dataset['fuelType'].unique()


# In[413]:


stats.mode(dataset['fuelType'])


# In[414]:


statistics.mode(dataset['fuelType'])


# In[415]:


dataset['fuelType'] = dataset['fuelType'].replace(to_replace = np.nan, value = statistics.mode(dataset['fuelType']))


# In[416]:


dataset['fuelType'].unique()


# In[ ]:




