
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


PATH = 'D:/Data_Science/marketing_data.xlsx'
df = pd.read_excel(PATH)


# In[3]:


df.head()


# In[4]:


df.shape


# df.columns

# In[5]:


df.columns


# In[6]:


df = df.loc[df['ID'].notnull()]


# In[7]:


print(df.shape)


# In[ ]:


columns = ['Ignore1', 'Ignore2', 'Ignore3', 'Ignore4', 'Ignore5', 'Ignore6', 'PRESMED', 'TOP', 'PE', 'SPECIALITY', 

