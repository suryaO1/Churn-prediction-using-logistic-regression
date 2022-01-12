#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('Cellphone. csv')


# In[4]:


data


# In[5]:


pd.options.display.float_format = '{:,.4f}'.format
corr_matrix1 = data.corr()
corr_matrix1


# In[6]:


corr_matrix1['Churn'].sort_values(ascending=False)


# In[7]:


plt.figure(figsize=(16,8))
a = sns.heatmap(corr_matrix1, square = True,annot = True,fmt ='.2f', linecolor ='black')
plt.show()
    


# In[8]:


data.Churn==1


# In[9]:


data.columns


# In[10]:


data.DataPlan.unique()


# In[11]:


data_data_list = ['1']


# In[12]:


data_sample = data[data.DataPlan.isin(data_data_list)]
data_sample.head(100)


# In[13]:


x = data[['ContractRenewal','CustServCalls', 'DataUsage','RoamMins','DayCalls','OverageFee']]
y = data[['Churn']]


# In[14]:


x.isnull().sum()


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[16]:


from sklearn.linear_model import LogisticRegression
elff = LogisticRegression()


# In[17]:


grid=[
    {'penalty':['l1','l2','elasticnet','none'],
     'C':np.logspace(-4,10,20),
     'solver':['lbfgs','newton-cg','liblinear','sag','saga'],
     'max_iter':[100,1000,5000]
    }
]


# In[18]:


from sklearn.model_selection import GridSearchCV


# In[19]:


elf =GridSearchCV(elff,param_grid = grid, cv = 3, verbose=True, n_jobs=-1)


# In[20]:


best_elf = elf.fit(x_train,y_train)


# In[47]:


y_test


# In[48]:


x_test


# In[34]:


best_elf.best_estimator_


# In[35]:


best_elf.score(x_train,y_train)


# In[36]:


best_elf.score(x_test,y_test)

