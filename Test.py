#!/usr/bin/env python
# coding: utf-8

# In[129]:


from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# In[130]:


import pandas as pd
import json
with open("C:/Users/DELL/Downloads/algoparams_from_ui.json") as f:
    data = json.load(f)
    print(data["design_state_data"]["target"]["target"])
    print(data["design_state_data"]["target"]["type"])
data


# In[131]:


data["design_state_data"]


# In[132]:


df=pd.read_csv("D:\iris.csv")
df.head(10)


# # Read the features (which are column names in the csv) 

# In[ ]:





# In[133]:


df.isnull().sum()


# In[134]:


df.describe()


# In[135]:


df.info()


# In[136]:


df.shape


# In[137]:


df.columns


# In[138]:


df["species"].value_counts()


# In[139]:


df['sepal_length'].hist()


# In[140]:


df['sepal_width'].hist()


# In[141]:


df['petal_length'].hist()


# In[142]:


df['petal_width'].hist()


# In[143]:


df.corr()


# In[144]:


corr= df.corr()
fig, ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax)


# In[145]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[146]:


df['species'] = le.fit_transform(df['species'])
df.head()


# In[147]:


X= df.drop(columns=['species'])
Y= df['species']
x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.30)


# # Performing PCA on Dataset

# In[148]:


from sklearn.decomposition import PCA


# In[149]:


X.shape


# In[150]:


Y.shape


# In[151]:


pca= PCA(n_components=2)


# In[152]:


pca.fit(X)


# In[153]:


pca.components_


# In[154]:


Z= pca.transform(X)


# In[155]:


Z.shape


# In[156]:


plt.scatter(Z[:,0],Z[:,1],c=Y)


# # Run the fit and predict on each model – keep in mind that you need to do hyper parameter tuning i.e., use GridSearchCV

# In[157]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm

clf = GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
}, cv=5, return_train_score=False)

clf.fit(df.X,df.Y)
clf.cv_results_


# In[158]:


#search_space= {
    #"n_estimators":[100,200,500],
    #"max_depth": [3,6,9],
    #"gamma":[0.01,0.1],
    #"learning_rate":[0.001,0.01,0.1,1]
#}


# In[168]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)


# In[169]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm,datasets
parameters = [{'C': [1,10,100,1000], 'kernel':['linear']},
              {'C': [1,10,100,1000], 'kernel':['rbf'],'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search= GridSearchCV(estimator= classifier,
                param_grid=parameters,
                scoring='accuracy',
                cv=10,
                n_jobs=-1)
grid_search= GridSearchCV.fit(x_train, y_train)


# In[161]:


grid_search.best_score_


# In[162]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[163]:


model.fit(x_train, y_train)


# In[164]:


print("Accuracy: ",model.score(x_test, y_test)*100)


# In[ ]:





# In[ ]:





# In[ ]:




