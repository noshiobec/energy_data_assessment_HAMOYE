#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


ls


# In[3]:


pd.set_option('display.max_columns',None)
df_energy = pd.read_csv('energydata_complete (1).csv')
df_energy.head()


# In[4]:


df_energy.describe


# In[5]:


df_energy.isna().sum()


# In[6]:


df_energy.info


# In[7]:


sns.distplot(df_energy['T6'])


# Question 12 - Spliting my data into X and y, 
#                Split the data into train test and split test, 
#                Training the model
#                

# In[8]:


X = df_energy['T2']
y = df_energy['T6']


# In[9]:


X=pd.DataFrame(X)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


#train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[12]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[13]:


lm.fit(X, y)


# In[14]:


predicted_values = lm.predict(X_test)


# In[15]:


from sklearn.metrics import r2_score
round(r2_score(y_test,predicted_values),2)


# # Therefore answer to question 12 on R squared value = 0.64

# BELOW IS THE START OF QUESTION 13 

# Spliting my data into X and y, 
#                Split the data into train test and split test, 
#                Training the model
#                

# In[16]:


d_energy=df_energy.drop(['date','lights'],axis=1)
d_energy


# In[17]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df=pd.DataFrame(scaler.fit_transform(d_energy),columns=d_energy.columns)
target=normalised_df['Appliances']
features_df = normalised_df.drop('Appliances',axis=1)


# In[18]:


features_df


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.3, random_state=42)


# In[20]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[21]:


lm.fit(X_train, y_train)


# In[22]:


predicted_values = lm.predict(X_test)


# In[23]:


predicted_values


# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[25]:


round(mean_absolute_error(y_test, predicted_values),2)


# # Question 13 answer on MAE = 0.05

# Question 15 on RMSE

# In[26]:


round(np.sqrt(mean_squared_error(y_test,predicted_values )),3)


# # Question 15 answer on RMSE = 0.088

# Question 14 on Residual sum of squares

# In[27]:


RSS= ((y_test-predicted_values )**2).sum()
round(RSS,2)


# # Question 14 answer on rss value = 45.35

# Question 16 on Coefficient of Determination also called R squared 

# In[28]:


from sklearn.metrics import r2_score
round(r2_score(y_test,predicted_values),2)


# # Question 16 answer on coefficiaent of determination = 0.15

# Question 17 on Feature weights

# In[29]:


lm_feature_weights=pd.DataFrame(lm.coef_,X_train.columns,columns=['weights']).sort_values('weights')
lm_feature_weights


# # Question 17 answer on feature weights = RH_2 and RH_1 respectively

# Question 18 on traininng a ridge regtression model

# In[30]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X_train, y_train)


# In[31]:


ridge_reg.coef_


# In[32]:


predicted_values = ridge_reg.predict(X_test)


# In[33]:


round(np.sqrt(mean_squared_error(y_test,predicted_values )),3)


# # Question 18 answer on RMSE value after traininng the ridge regtression model = NO  as there is no change

# Question 19 answer on count of non zero feature weights after training a lasso regression model

# In[34]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)


# In[35]:


lc=lasso_reg.coef_
lc


# In[36]:


np.count_nonzero(lc)


# # Question 19 answer on count of non zero feature weights = 4

# QUESTION 20 on RMSE value after lasso regression model

# In[37]:


predicted_values = lasso_reg.predict(X_test)


# In[38]:


round(np.sqrt(mean_squared_error(y_test,predicted_values )),3)


# # Question 20 answer on RMSE after training a lasso model = 0.094
