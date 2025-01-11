#!/usr/bin/env python
# coding: utf-8

# https://www.udemy.com/course/estatistica-para-ciencia-de-dados-machine-learning

# In[1]:


import pandas as pd


# # Amostragem

# ## Carregamento da base de dados

# ### Biblioteca

# In[2]:


import pandas as pd 
import random
import numpy as np   
#


# ### Base de dados

# In[3]:


dataset = pd.read_csv('census.csv')


# In[ ]:


dataset.shape


# In[6]:


dataset.head(10)


# In[7]:


dataset.tail(5)


# ## Amostragem aleatória simples

# In[8]:


df_amostra_aleatoria_simples = dataset.sample(n = 100, random_state = 1)


# In[9]:


df_amostra_aleatoria_simples.shape


# In[10]:


df_amostra_aleatoria_simples.head()


# In[11]:


def amostragem_aleatoria_simples(dataset, amostras):
  return dataset.sample(n = amostras, random_state=1)


# In[13]:


df_amostra_aleatoria_simples = amostragem_aleatoria_simples(dataset, 100)
df_amostra_aleatoria_simples.shape


# In[14]:


df_amostra_aleatoria_simples.head()


# ## Amostragem sistemática

# In[15]:


dataset.shape


# In[16]:


len(dataset) // 100


# In[17]:


random.seed(1)
random.randint(0, 325)


# In[18]:


68 + 325


# In[19]:


393 + 325


# In[20]:


np.arange(68, len(dataset), step = 325)


# In[21]:


def amostragem_sistematica(dataset, amostras):
  intervalo = len(dataset) // amostras
  random.seed(1)
  inicio = random.randint(0, intervalo)
  indices = np.arange(inicio, len(dataset), step = intervalo)
  amostra_sistematica = dataset.iloc[indices]
  return amostra_sistematica


# In[22]:


df_amostra_sistematica = amostragem_sistematica(dataset, 100)
df_amostra_sistematica.shape


# In[23]:


df_amostra_sistematica.head()


# ## Amostragem por grupos

# In[24]:


len(dataset) / 10


# In[43]:


# separa a base em 10 grupos 
grupos = []
id_grupo = 0
contagem = 0
for _ in dataset.iterrows():
  grupos.append(id_grupo)
  contagem += 1
  if contagem > 3256:
    contagem = 0
    id_grupo += 1


# In[44]:


print(grupos)


# In[45]:


np.unique(grupos, return_counts=True)


# In[46]:


np.shape(grupos), dataset.shape


# In[47]:


dataset['grupo'] = grupos


# In[48]:


dataset.head()


# In[49]:


dataset.tail()


# In[50]:


random.randint(0, 9)


# In[51]:


df_agrupamento = dataset[dataset['grupo'] == 7]
df_agrupamento.shape


# In[52]:


df_agrupamento['grupo'].value_counts()


# In[53]:


def amostragem_agrupamento(dataset, numero_grupos):
  intervalo = len(dataset) / numero_grupos

  grupos = []
  id_grupo = 0
  contagem = 0
  for _ in dataset.iterrows():
    grupos.append(id_grupo)
    contagem += 1
    if contagem > intervalo:
      contagem = 0
      id_grupo += 1

  dataset['grupo'] = grupos
  random.seed(1)
  grupo_selecionado = random.randint(0, numero_grupos)
  return dataset[dataset['grupo'] == grupo_selecionado]


# In[54]:


len(dataset) / 325


# In[55]:


325 * 100


# In[56]:


df_amostra_agrupamento = amostragem_agrupamento(dataset, 325)
df_amostra_agrupamento.shape, df_amostra_agrupamento['grupo'].value_counts()


# In[57]:


df_amostra_agrupamento.head()


# ## Amostra estratificada

# In[58]:


# separa a base respeitando a proporção de determinada componente


# In[59]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[60]:


dataset['income'].value_counts()


# In[61]:


7841 / len(dataset), 24720 / len(dataset)


# In[62]:


0.2408095574460244 + 0.7591904425539756


# In[63]:


100 / len(dataset)


# In[64]:


split = StratifiedShuffleSplit(test_size=0.0030711587481956942)
for x, y in split.split(dataset, dataset['income']):
  df_x = dataset.iloc[x]
  df_y = dataset.iloc[y]


# In[65]:


df_x.shape, df_y.shape


# In[66]:


df_y.head()


# In[67]:


df_y['income'].value_counts()


# In[68]:


def amostragem_estratificada(dataset, percentual):
  split = StratifiedShuffleSplit(test_size=percentual, random_state=1)
  for _, y in split.split(dataset, dataset['income']):
    df_y = dataset.iloc[y]
  return df_y


# In[69]:


df_amostra_estratificada = amostragem_estratificada(dataset, 0.0030711587481956942)
df_amostra_estratificada.shape


# ## Amostragem de reservatório

# In[70]:


stream = []
for i in range(len(dataset)):
  stream.append(i)


# In[ ]:


print(stream)


# In[78]:


def amostragem_reservatorio(dataset, amostras):
  stream = []
  for i in range(len(dataset)):
    stream.append(i)

  i = 0
  tamanho = len(dataset)

  reservatorio = [0] * amostras
  for i in range(amostras):
    reservatorio[i] = stream[i]

  while i < tamanho:
    j = random.randrange(i + 1)
    if j < amostras:
      reservatorio[j] = stream[i]
    i += 1

  return dataset.iloc[reservatorio]
  


# In[79]:


df_amostragem_reservatorio = amostragem_reservatorio(dataset, 100)
df_amostragem_reservatorio.shape


# In[80]:


df_amostragem_reservatorio.head()


# ## Comparativo dos resultados

# In[81]:


dataset['age'].mean()


# In[82]:


df_amostra_aleatoria_simples['age'].mean()


# In[83]:


df_amostra_sistematica['age'].mean()


# In[84]:


df_amostra_agrupamento['age'].mean()


# In[85]:


df_amostra_estratificada['age'].mean()


# In[86]:


df_amostragem_reservatorio['age'].mean()


# In[ ]:





# In[ ]:




