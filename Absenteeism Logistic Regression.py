#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# In[3]:


##Reading file
csv_file = pd.read_csv('C:/Users/user/Documents/Data Science/df_preprocessed.csv')


# In[4]:


csv_file


# In[5]:


df = csv_file.copy()


# In[6]:


df.head()


# In[11]:


##To display entire dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',  None)
df


# In[8]:


df.info()


# In[9]:


##Setting Target
Target = df['Absenteeism Time in Hours']
m = Target.mean()
m


# In[10]:


Targets = np.where(Target>m, 1, 0)


# In[12]:


from sklearn.preprocessing import StandardScaler
Var = StandardScaler()


# In[27]:


##Fitting
df = df= df.drop(['Absenteeism Time in Hours'], axis = 1)
unscaled_inputs = df
Scaled_inputs = Var.fit(unscaled_inputs)


# In[28]:


##Transforming
Scaled_inputs = Var.transform(unscaled_inputs)


# In[29]:


##Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Scaled_inputs, Targets, train_size = 0.8, random_state = 20)


# In[30]:


##Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
Reg = LogisticRegression()
Reg.fit(x_train, y_train)


# In[31]:


Reg.score(x_train, y_train)


# In[32]:


##Checking accuracy
model_outputs = Reg.predict(x_train)
model_outputs == y_train


# In[33]:


Reg.intercept_


# In[34]:


Reg.coef_


# In[35]:


Feature_name = df.columns.values
Feature_name


# In[36]:


##Summary Table
Summary_table = pd.DataFrame(columns = ['Feature_name'], data = Feature_name)
Summary_table['Coefficient'] = np.transpose(Reg.coef_)
Summary_table


# In[37]:


##Adding Intercept
Summary_table.index = Summary_table.index + 1
Summary_table.loc[0] = ['intercept', Reg.intercept_[0]]
Summary_table = Summary_table.sort_index()
Summary_table


# In[38]:


y_pred = Reg.predict(x_test)


# In[39]:


y_pred


# In[40]:


##Accuracy of Logistic Regression
Reg.score(x_test, y_test)


# In[ ]:





# In[42]:


##Confusion matrix
from sklearn.metrics import confusion_matrix as con


# In[44]:


##Showing numbers of correct and incorrect predictions
con_matrix = con(y_test, y_pred)
con_matrix


# In[45]:


##Compute Precision
from sklearn.metrics import classification_report as rep


# In[46]:


class_rep = rep(y_test, y_pred)


# In[47]:


class_rep


# In[ ]:


##the above shows that 76% of the entire datset


# In[48]:


##ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[52]:


Reg.predict(x_train)


# In[53]:


Reg.predict_proba(x_train)

Reg.score(x_train, y_train)
# In[54]:


Reg.score(x_train, y_train)


# In[ ]:




