
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


PATH = 'D:/Data_Science/Iris.xlsx'
df = pd.read_excel(PATH)


# In[3]:


df.head


# In[4]:


df.tail()


# In[5]:


df


# In[6]:


df.head(2)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.Species


# In[10]:


df.Species.value_counts()


# In[11]:


df.head()


# In[12]:


df.drop('Id', axis=1,inplace=True)


# In[13]:


df.head()


# In[15]:


fig = df[df.Species=='Iris-setosa'].plot(kind='scatter', x= 'SepalLengthCm', y= 'SepalWidthCm', color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter', x= 'SepalLengthCm', y= 'SepalWidthCm', color='blue', label='versicolor', ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter', x= 'SepalLengthCm', y= 'SepalWidthCm', color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_xlabel("Sepal Width")
fig.set_title("Sepal Length vs Width")
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[16]:


fig = df[df.Species=='Iris-setosa'].plot(kind='scatter', x= 'PetalLengthCm', y= 'PetalWidthCm', color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter', x= 'PetalLengthCm', y= 'PetalWidthCm', color='blue', label='versicolor', ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter', x= 'PetalLengthCm', y= 'PetalWidthCm', color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_xlabel("Petal Width")
fig.set_title("Petal Length vs Width")
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[18]:


from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[20]:


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[21]:


plt.figure(figsize=(7,4))
sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
plt.show()


# In[22]:


train, test = train_test_split(df, test_size=0.3)
print(train.shape)
print(test.shape)


# In[23]:


train.head()


# In[24]:


test.head()


# In[25]:


x_train = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_train = train.Species
x_test = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_test = test.Species


# In[26]:


x_train.head(5)


# In[27]:


x_test.head(5)


# In[28]:


y_train.head(5)


# In[29]:


y_test.head(5)


# In[30]:


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)


# In[31]:


train_prediction = knn_model.predict(x_train)
test_prediction = knn_model.predict(x_test)
train_accuracy = metrics.accuracy_score(train_prediction,y_train)
test_accuracy = metrics.accuracy_score(test_prediction,y_test)
print('The training accuracy of the KNN model is ', train_accuracy)
print('The test accuracy of the KNN model is ', test_accuracy)


# In[32]:


train_prediction = knn_model.predict(x_train)
test_prediction = knn_model.predict(x_test)
train_accuracy = metrics.accuracy_score(train_prediction,y_train)
test_accuracy = metrics.accuracy_score(test_prediction,y_test)
print('The training accuracy of the KNN model is ', train_accuracy)
print('The test accuracy of the KNN model is ', test_accuracy)


# In[33]:


lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

