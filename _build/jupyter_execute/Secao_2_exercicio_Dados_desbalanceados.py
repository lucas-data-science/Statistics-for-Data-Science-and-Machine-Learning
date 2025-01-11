#!/usr/bin/env python
# coding: utf-8

# Este exercício é parte do curso: https://www.udemy.com/course/estatistica-para-ciencia-de-dados-machine-learning/

# # Classificação com dados desbalanceados

# ## Carregamento da base de dados

# In[1]:


import pandas as pd
import random
import numpy as np


# In[2]:


dataset = pd.read_csv('csv_result-ebay_confianca_completo.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.head(30)


# In[5]:


dataset['blacklist'] = dataset['blacklist'] == 'S'


# In[6]:


import seaborn as sns
sns.countplot(dataset['reputation']);


# In[7]:


len(dataset.columns)


# In[8]:


X = dataset.iloc[:,0:74].values


# In[9]:


X.shape


# In[10]:


X


# In[11]:


y = dataset.iloc[:,74].values


# In[12]:


y


# In[13]:


np.unique(y, return_counts=True)


# ## Base de treinamento e teste

# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y)


# In[16]:


X_treinamento.shape, y_treinamento.shape


# In[17]:


X_teste.shape, y_teste.shape


# ## Classificação com Random Forest

# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


modelo = RandomForestClassifier()
modelo.fit(X_treinamento, y_treinamento)


# In[20]:


previsoes = modelo.predict(X_teste)
from sklearn.metrics import accuracy_score
accuracy_score(previsoes, y_teste)


# ## Subamostragem (undersampling) - Tomek links
# 
# - https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html

# In[21]:


get_ipython().system('pip install imblearn')
from imblearn.under_sampling import TomekLinks


# In[22]:


tl = TomekLinks(sampling_strategy='majority')
X_under, y_under = tl.fit_resample(X, y)


# In[23]:


X_under.shape, y_under.shape


# In[24]:


np.unique(y, return_counts=True)


# In[25]:


np.unique(y_under, return_counts=True)


# In[26]:


X_treinamento_u, X_teste_u, y_treinamento_u, y_teste_u = train_test_split(X_under, y_under, 
                                                                          test_size = 0.2, stratify = y_under)
X_treinamento_u.shape, X_teste_u.shape


# In[27]:


modelo_u = RandomForestClassifier()
modelo_u.fit(X_treinamento_u, y_treinamento_u)
previsoes_u = modelo_u.predict(X_teste_u)
accuracy_score(previsoes_u, y_teste_u)


# ## Sobreamostragem (oversampling) - SMOTE

# In[28]:


from imblearn.over_sampling import SMOTE


# In[29]:


smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X, y)


# In[30]:


X_over.shape, y_over.shape


# In[31]:


np.unique(y, return_counts=True)


# In[32]:


np.unique(y_over, return_counts=True)


# In[33]:


X_treinamento_o, X_teste_o, y_treinamento_o, y_teste_o = train_test_split(X_over, y_over, 
                                                                          test_size = 0.2, stratify = y_over)
X_treinamento_o.shape, X_teste_o.shape


# In[34]:


modelo_o = RandomForestClassifier()
modelo_o.fit(X_treinamento_o, y_treinamento_o)
previsoes_o = modelo_o.predict(X_teste_o)
accuracy_score(previsoes_o, y_teste_o)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




