#!/usr/bin/env python
# coding: utf-8

# In[113]:


# Simple Linear Regression (ML-I) ;
get_ipython().run_line_magic('reset', '')


# In[144]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as seaborninstance
from sklearn.model_selection import train_test_split 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[115]:


dataset = pd.read_csv("/Users/ankurgupta/Documents/Online Courses/ML India Catalogue/ML-Algorithms-master/Regression/SimpleLinearRegression/student_scores.csv")
dataset 
dataset.shape
dataset.describe()


# In[116]:


dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[130]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values 


# In[131]:


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0, test_size = 0.2) 
regressor = LinearRegression()
regressor.fit(X_train, y_train) 


# In[132]:


print(regressor.intercept_)
print(regressor.coef_)


# In[133]:


y_pred = regressor.predict(X_test)
y_pred
# y_pred is a numpy array


# In[141]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[139]:


from sklearn import metrics as mt
print('Mean Absolute Error (MAE):', mt.mean_absolute_error(y_test, y_pred))
print('MSE :', mt.mean_squared_error(y_test, y_pred))
print('RMSE :', np.sqrt(mt.mean_squared_error(y_test,y_pred)))


# In[143]:


plt.scatter(X_test, y_test, color = "gray")
plt.plot(X_test, y_pred, color ="red", linewidth = 2)
plt.show()


# In[ ]:


# RMSE is less than the 10% of the mean of y values 


# In[ ]:


# Multiple Linear Regression
# https://medium.com/codingninjas-blog/step-by-step-guide-to-execute-linear-regression-in-python-97122e2cd8bd


# In[147]:


dataset = pd.read_csv("/Users/ankurgupta/Documents/Online Courses/ML India Catalogue/ML-Algorithms-master/Regression/SimpleLinearRegression/petrol_consumption.csv")


# In[148]:


dataset.describe()


# In[151]:


dataset.isnull().any()


# In[153]:


dataset = dataset.fillna(method = 'ffill')
dataset.head()


# In[152]:


X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']


# In[155]:


from sklearn.linear_model import LinearRegression 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)


# In[157]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[159]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns = ['Coefficients'])
coeff_df


# In[172]:


intercept = regressor.intercept_
intercept


# In[162]:


y_pred = regressor.predict(X_test)


# In[167]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[168]:


from sklearn import metrics as mt
print('Mean Absolute Error (MAE):', mt.mean_absolute_error(y_test, y_pred))
print('MSE :', mt.mean_squared_error(y_test, y_pred))
print('RMSE :', np.sqrt(mt.mean_squared_error(y_test,y_pred)))


# In[ ]:


# RMSE is slightly higher than the 10% of the mean estimated 


# In[171]:


df1 = df.head(25)
df1.plot(kind=’bar’,figsize=(10,8))
plt.grid(which=’major’, linestyle=’-‘, linewidth=’0.5′, color=’green’)
plt.grid(which=’minor’, linestyle=’:’, linewidth=’0.5′, color=’black’)
plt.show()

