#!/usr/bin/env python
# coding: utf-8

# https://www.udemy.com/course/estatistica-para-ciencia-de-dados-machine-learning

# In[1]:


import pandas as pd


# ## Percentuais

# In[2]:


dados = {'emprego': ['Adminstrador_banco_dados', 'Programador', 'Arquiteto_redes'],
         'nova_jersey': [97350, 82080, 112840],
         'florida': [77140, 71540, 62310]}


# In[3]:


type(dados)


# In[4]:


dados


# In[5]:


dataset = pd.DataFrame(dados)


# In[6]:


dataset


# In[7]:


dataset['nova_jersey'].sum()


# In[8]:


dataset['florida'].sum()


# In[9]:


dataset['%_nova_jersey'] = (dataset['nova_jersey'] / dataset['nova_jersey'].sum()) * 100


# In[10]:


dataset


# In[11]:


dataset['%_florida'] = (dataset['florida'] / dataset['florida'].sum()) * 100


# In[12]:


dataset


# In[ ]:





# In[ ]:





# ## Exercício percentuais

# In[13]:


dataset = pd.read_csv('data/census.csv')


# In[14]:


dataset.head()


# In[15]:


dataset2 = dataset[['income', 'education']]
dataset2


# In[16]:


dataset3 = dataset2.groupby(['education', 'income'])['education'].count()


# In[17]:


dataset3


# In[18]:


dataset3.index


# In[19]:


dataset3[' Bachelors', ' <=50K'], dataset3[' Bachelors', ' >50K']


# In[20]:


3134 + 2221


# In[21]:


# % >50K
(2221 / 5355) * 100


# In[22]:


# % <=50K
(3134 / 5355) * 100


# ## Exercício coeficientes e taxas

# In[23]:


dados = {'ano': ['1', '2', '3', '4', 'total'],
        'matriculas_marco': [70, 50, 47, 23, 190],
        'matriculas_novembro': [65, 48, 40, 22, 175]}


# In[24]:


dados


# In[25]:


dataset = pd.DataFrame(dados)
dataset


# In[26]:


dataset['taxa_evasao'] = ((dataset['matriculas_marco'] - dataset['matriculas_novembro']) / dataset['matriculas_marco']) * 100


# In[27]:


dataset

