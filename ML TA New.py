#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans 
from sklearn.metrics import r2_score 
from scipy import optimize
from scipy.cluster.hierarchy import dendrogram, linkage
from umap import UMAP


# # Read Data

# In[2]:


Data = pd.read_csv('Data_SimNew.csv')


# In[3]:


Data


# In[4]:


Data = Data.drop(["Start Loaded (Month)"], axis=1)
Data


# In[5]:


Data.corr()


# # Heatmap

# In[6]:


plt.figure(figsize = (10, 6))
heatmap = sns.heatmap(Data.corr(), vmin = -1, vmax = 1, annot = True)
heatmap.set_title('Heatmap', fontdict = {'fontsize':20}, pad = 12);


# In[7]:


Turner = Data.drop(['Well Status', 'Modified Sigmoid ((x/1000)/(1+e^-x))'], axis = 1)
heatmapTurner = sns.heatmap(Turner.corr()[['Turner Rate (MMSCFD)']].sort_values(by = 'Turner Rate (MMSCFD)', ascending = False), vmin = -1, vmax = 1, annot = True, cmap = 'YlGnBu')
heatmapTurner.set_title('Heatmap Turner Rate', fontdict = {'fontsize':20}, pad = 12);


# In[8]:


Loaded = Data.drop(['Well Status', 'Turner Rate (MMSCFD)'], axis = 1)
heatmapLoaded = sns.heatmap(Loaded.corr()[['Modified Sigmoid ((x/1000)/(1+e^-x))']].sort_values(by = 'Modified Sigmoid ((x/1000)/(1+e^-x))', ascending = False), vmin = -1, vmax = 1, annot = True, cmap = 'YlGnBu')
heatmapLoaded.set_title('Heatmap Start Loaded', fontdict = {'fontsize':20}, pad = 12);


# # Clustering

# # First Stage Clustering

# In[9]:


inputdata = Data.drop(["Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))"], axis = 1)
#Using UMAP
umap = UMAP(n_components = 2, init = 'random', random_state = 0)
proj = umap.fit_transform(inputdata)
graph = px.scatter(proj, x = 0, y = 1)
graph.show()


# Looking for Optimum Cluster

# In[10]:


#Elbow Method
X = Data
X = Data.drop(["Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))"], axis = 1)
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show


# In[11]:


#Dendogram
linked = linkage(proj, 'single')

labelList = range(1, 56)

plt.figure(figsize=(15, 8))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# Clustering

# In[12]:


#First Stage using 3 clusters
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
label = cluster.fit_predict(proj)
Data.insert(9, 'Label', label, False)


# # Second Stage Clustering

# In[13]:


inputdata = Data.drop(["Reference Pressure (psia)", "Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))"], axis = 1)
#Using UMAP
umap = UMAP(n_components = 2, init = 'random', random_state = 0)
proj = umap.fit_transform(inputdata)
graph = px.scatter(proj, x = 0, y = 1)
graph.show()


# Looking for Optimum Cluster

# In[14]:


#Elbow Method
X = Data
X = Data.drop(["Reference Pressure (psia)", "Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))"], axis = 1)
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show


# In[15]:


#Dendogram
linked = linkage(proj, 'single')

labelList = range(1, 56)

plt.figure(figsize=(15, 8))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# Clustering

# In[16]:


#Second Stage using 2 clusters
Data = Data.drop(["Label"], axis =1)
kmeans = KMeans(n_clusters = 2)
label = kmeans.fit_predict(proj)
Data.insert(9, 'Label', label, False)

Label0 = proj[label == 0]
Label1 = proj[label == 1]


plt.scatter(Label0[:, 0], Label0[:, 1], color = "red")
plt.scatter(Label1[:, 0], Label1[:, 1], color = "black")


plt.show()


# # Regression

# In[17]:


Data


# DATA LABEL 0

# In[18]:


Data0 = Data[Data['Label'] == 0]
Data0


# In[19]:


#Scalling
Data0 = Data[Data['Label'] == 0]
xdata = Data0
xdata = Data0.drop(["Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))", 'Label'], axis = 1)
scaler = StandardScaler()
xdata_scaled0 = xdata.copy()
#xdata_scaled0 = scaler.fit_transform(xdata_scaled0)
xdata_scaled0 = pd.DataFrame(xdata_scaled0, index = xdata.index, columns = xdata.columns)
xdata_scaled0.describe()


# In[20]:


#Check correlation using Heatmap
plt.figure(figsize = (10, 6))
check = sns.heatmap(xdata_scaled0.corr(), vmin = -1, vmax = 1, annot = True)
check.set_title('Heatmap Data 0', fontdict = {'fontsize':20}, pad = 12);


# DATA LABEL 1

# In[21]:


Data1 = Data[Data['Label'] == 1]
Data1


# In[22]:


#Scalling
Data1 = Data[Data['Label'] == 1]
xdata = Data1
xdata = Data1.drop(["Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))", 'Label'], axis = 1)
scaler = StandardScaler()
xdata_scaled1 = xdata.copy()
#xdata_scaled1 = scaler.fit_transform(xdata_scaled1)
xdata_scaled1 = pd.DataFrame(xdata_scaled1, index = xdata.index, columns = xdata.columns)
xdata_scaled1.describe()


# In[23]:


#Check correlation using Heatmap
plt.figure(figsize = (10, 6))
check = sns.heatmap(xdata_scaled1.corr(), vmin = -1, vmax = 1, annot = True)
check.set_title('Heatmap Data 1', fontdict = {'fontsize':20}, pad = 12);


# # Stats Model

# DATA LABEL 0

# In[24]:


X0 = xdata_scaled0
Y0 = Data0["Modified Sigmoid ((x/1000)/(1+e^-x))"]
model = sm.OLS(Y0, X0)
results = model.fit()
print(results.summary())


# In[25]:


X0 = xdata_scaled0
Y0 = Data0["Turner Rate (MMSCFD)"]
model = sm.OLS(Y0, X0)
results = model.fit()
print(results.summary())


# DATA LABEL 1

# In[26]:


X1 = xdata_scaled1
Y1 = Data1["Modified Sigmoid ((x/1000)/(1+e^-x))"]
model = sm.OLS(Y1, X1)
results = model.fit()
print(results.summary())


# In[27]:


X1 = xdata_scaled1
Y1 = Data1["Turner Rate (MMSCFD)"]
model = sm.OLS(Y1, X1)
results = model.fit()
print(results.summary())


# # Gradient Boosting

# # DATA LABEL 0

# In[28]:


xdata = Data0
xdata = Data0.drop(["Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))", 'Label'], axis = 1)
scaler = StandardScaler()
xdata_scaled = xdata.copy()
xdata_scaled = pd.DataFrame(xdata_scaled, index = xdata.index, columns = xdata.columns)

X0 = xdata_scaled
Y0 = Data0["Modified Sigmoid ((x/1000)/(1+e^-x))"]

x_train, x_test, y_train, y_test = train_test_split(X0, Y0, test_size = 0.3, random_state=4) 

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

regression = GradientBoostingRegressor(**params)
regression.fit(x_train, y_train)
y_pred_test = regression.predict(x_test)
y_pred_train = regression.predict(x_train)
print("Rsquare Train Data : ", r2_score(y_train, y_pred_train))
print("Rsquare Test Data : ", r2_score(y_test, y_pred_test))


# In[29]:


#Validation
params = {'n_estimators': [50, 100, 200, 300, 400, 500],
          'max_depth': [2, 4, 6, 8, 10],
          'min_samples_split': [2, 4, 8, 10],
          'learning_rate': [0.001, 0.01, 0.1, 1],
          }
grid = GridSearchCV(GradientBoostingRegressor(), params, cv = 2)
grid.fit(X0, Y0)
print("Best score : ", grid.best_score_)
print(grid.best_params_)


# # DATA LABEL 1

# In[30]:


xdata = Data1
xdata = Data1.drop(["Rel Perm (No. Data)", "Turner Rate (MMSCFD)", "Well Status", "Modified Sigmoid ((x/1000)/(1+e^-x))", 'Label'], axis = 1)
scaler = StandardScaler()
xdata_scaled = xdata.copy()
xdata_scaled = pd.DataFrame(xdata_scaled, index = xdata.index, columns = xdata.columns)

X1 = xdata_scaled
Y1 = Data1["Modified Sigmoid ((x/1000)/(1+e^-x))"]

x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.3, random_state=5) 

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

regression = GradientBoostingRegressor(**params)
regression.fit(x_train, y_train)
y_pred_test = regression.predict(x_test)
y_pred_train = regression.predict(x_train)
print("Rsquare Train Data : ", r2_score(y_train, y_pred_train))
print("Rsquare Test Data : ", r2_score(y_test, y_pred_test))


# In[31]:


#Validation
params = {'n_estimators': [100, 200, 300, 400, 500],
          'max_depth': [2, 4, 6, 8, 10],
          'min_samples_split': [2, 4, 8, 10],
          'learning_rate': [0.001, 0.01, 0.1, 1],
          }
grid = GridSearchCV(GradientBoostingRegressor(), params, cv = 4)
grid.fit(X1, Y1)
print("Best score : ", grid.best_score_)
print(grid.best_params_)


# In[ ]:




