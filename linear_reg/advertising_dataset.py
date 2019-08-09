#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
dataset_advt = pd.read_csv("/home/negi/negi/Machine_Learn/Simple_Linear_regression/Advertising.csv")
print(dataset_advt)


# In[2]:


X = dataset_advt[["TV", "radio", "newspaper", "sales"]]
print(X)


# In[3]:


q = X.describe()
print(q)


# In[4]:


x = X.head()
print(x)


# In[5]:


y = dataset_advt.sales
print(y)


# In[12]:


sns.jointplot(x='sales',y='TV',data=dataset_advt,kind='scatter')


# In[13]:


sns.distplot(dataset_advt['newspaper'])


# In[14]:


sns.jointplot(x='sales',y='TV',data=dataset_advt,kind='hex')


# In[15]:


sns.jointplot(x='sales',y='TV',data=dataset_advt,kind='reg')


# In[17]:


sns.pairplot(dataset_advt,palette='coolwarm')


# In[18]:


sns.rugplot(dataset_advt['sales'])


# In[19]:


sns.lmplot(x='sales',y='TV',size=2,aspect=4,data=dataset_advt)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_advt[["TV", "radio", "newspaper"]],dataset_advt.sales,test_size=0.3,random_state = 42 )
print("X_train value:" , X_train)
print("X_test value :",X_test)
print("y_train value :",y_train)
print("y_test value :",y_test)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# In[7]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


regressor.intercept_
print("intercept Value : ", regressor.intercept_)

regressor.coef_
print(regressor.coef_ )


# In[9]:


y_pred = regressor.predict( X_test )
print("y_predction value : ",y_pred)


# In[20]:


test_pred_df = pd.DataFrame({ 'actual': y_test, 'predicted ': np.round(y_pred, 2), 'residuals': y_test - y_pred })
print(test_pred_df)

###################
from sklearn import metrics
rmse = np.sqrt( metrics.mean_squared_error(y_test,y_pred))
round( rmse, 2 )

print("root mean square value : ",round(rmse,2))
## calculating R-squared  it's means that if the value is closer to 100 % then our model is good

metrics.r2_score( y_test, y_pred )
print("R- squared value ",metrics.r2_score( y_test, y_pred ))


# In[23]:


residuals = (y_test-y_pred)
print("residuals value: ",residuals)


# In[21]:


sns.distplot(residuals)


# In[ ]:




