#!/usr/bin/env python
# coding: utf-8

# https://www.udemy.com/course/estatistica-para-ciencia-de-dados-machine-learning

# # Distribuição de frequência

# ## Importação das bibliotecas e dados originais

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.templates.default = "simple_white"
pio.renderers.default = "notebook_connected"


# In[2]:


dados = np.array([160, 165, 167, 164, 160, 166, 160, 161, 150, 152, 173, 160, 155,
                  164, 168, 162, 161, 168, 163, 156, 155, 169, 151, 170, 164,
                  155, 152, 163, 160, 155, 157, 156, 158, 158, 161, 154, 161, 156, 172, 153])


# ## Ordenação

# In[3]:


dados = np.sort(dados)


# In[4]:


dados


# In[5]:


minimo = dados.min()
minimo


# In[6]:


maximo = dados.max()
maximo


# In[7]:


np.unique(dados, return_counts=True)


# In[8]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=dados, 
    name='altura',  
    marker_color='blue',
    opacity=1
)) 
fig.update_layout(
    title_text='Distribuição de altura de '+str(len(dados))+' pessoas',  
    xaxis_title_text='Valor altura (cm)',  
    yaxis_title_text='Quantidade', 
    bargap=0.1, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)


# ## Número de classes
# 
# - i = 1 + 3.3 log n

# In[9]:


n = len(dados)
n


# In[10]:


i = 1 + 3.3 * np.log10(n)
i


# In[11]:


i = round(i)
i


# ## Amplitude do intervalo
# 
# - h = AA / i
# - AA = Xmax - Xmin

# In[12]:


AA = maximo - minimo
AA


# In[13]:


h = AA / i
h


# In[14]:


import math
h = math.ceil(h)
h


# ## Construção da distribuição de frequência

# In[15]:


intervalos = np.arange(minimo, maximo + 2, step = h)
intervalos


# In[16]:


intervalo1, intervalo2, intervalo3, intervalo4, intervalo5, intervalo6 = 0,0,0,0,0,0
for i in range(n):
  if dados[i] >= intervalos[0] and dados[i] < intervalos[1]:
    intervalo1 += 1
  elif dados[i] >= intervalos[1] and dados[i] < intervalos[2]:
    intervalo2 += 1
  elif dados[i] >= intervalos[2] and dados[i] < intervalos[3]:
    intervalo3 += 1
  elif dados[i] >= intervalos[3] and dados[i] < intervalos[4]:
    intervalo4 += 1
  elif dados[i] >= intervalos[4] and dados[i] < intervalos[5]:
    intervalo5 += 1
  elif dados[i] >= intervalos[5] and dados[i] < intervalos[6]:
    intervalo6 += 1


# In[17]:


lista_intervalos = []
lista_intervalos.append(intervalo1)
lista_intervalos.append(intervalo2)
lista_intervalos.append(intervalo3)
lista_intervalos.append(intervalo4)
lista_intervalos.append(intervalo5)
lista_intervalos.append(intervalo6)
lista_intervalos


# In[18]:


lista_classes = []
for i in range(len(lista_intervalos)):
  lista_classes.append(str(intervalos[i]) + '-' + str(intervalos[i + 1]))


# In[19]:


lista_classes


# In[20]:


fig = px.histogram(lista_classes, lista_intervalos)
fig.show() 


# ## Distribuição de frequência e histograma com numpy e matplotlib
# 
# - https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
# - https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges

# In[21]:


dados = np.array([160, 165, 167, 164, 160, 166, 160, 161, 150, 152, 173, 160, 155,
                  164, 168, 162, 161, 168, 163, 156, 155, 169, 151, 170, 164,
                  155, 152, 163, 160, 155, 157, 156, 158, 158, 161, 154, 161, 156, 172, 153])


# In[22]:


frequencia, classes = np.histogram(dados)


# In[23]:


frequencia, classes, len(classes)


# In[24]:


fig = px.histogram(dados, nbins = len(classes))
fig.show() 


# In[25]:


frequencia, classes = np.histogram(dados, bins=5)
frequencia, classes


# In[26]:


fig = px.histogram(dados, nbins=5)
fig.show() 
dados, classes


# In[27]:


plt.hist(dados, classes);


# In[28]:


frequencia, classes = np.histogram(dados, bins = 'sturges')
frequencia, classes


# In[29]:


plt.hist(dados, classes);


# ## Distribuição de frequência e histograma com pandas e seaborn

# In[30]:


type(dados)


# In[31]:


dataset = pd.DataFrame({'dados': dados})


# In[32]:


dataset.head()


# In[33]:


dataset.plot.hist();


# In[34]:


sns.distplot(dados, hist = True, kde = True);


# ## Exercício - idade census.csv

# In[35]:


import pandas as pd
dataset = pd.read_csv('data/census.csv')


# In[36]:


dataset.head()


# In[37]:


dataset['age'].max(), dataset['age'].min()


# In[38]:


fig = px.histogram(dataset, x="age",nbins=100)
fig.show()


# In[39]:


dataset['age'] = pd.cut(dataset['age'], bins=[0, 17, 25, 40, 60, 90], 
                        labels=['Faixa1', "Faixa2", "Faixa3", "Faixa4", "Faixa5"])


# In[40]:


dataset.head()


# In[41]:


dataset['age'].unique()


# ## Regras de associação

# In[42]:


dataset.head()


# In[43]:


dataset_apriori = dataset[['age', 'workclass', 'education', 'marital-status', 'relationship', 'occupation',
                            'sex', 'native-country', 'income']]


# In[44]:


dataset_apriori.head()


# In[45]:


dataset.shape


# In[46]:


dataset_apriori = dataset_apriori.sample(n = 1000)
dataset_apriori.shape


# In[47]:


transacoes = []
for i in range(dataset_apriori.shape[0]):
  transacoes.append([str(dataset_apriori.values[i, j]) for j in range(dataset_apriori.shape[1])])


# In[48]:


len(transacoes)


# In[49]:


transacoes[:2]


# In[50]:


get_ipython().system('pip install apyori')


# In[51]:


from apyori import apriori


# In[52]:


regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.2)
resultados = list(regras)


# In[53]:


len(resultados)


# In[54]:


resultados


# In[55]:


resultados[12]


# In[ ]:





# In[ ]:





# In[ ]:




