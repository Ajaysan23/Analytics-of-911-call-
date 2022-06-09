#!/usr/bin/env python
# coding: utf-8

# # my-capstone-project-on-911-call
# 
# Use the "Run" button to execute the code.

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


# Execute this to save new versions of the notebook
jovian.commit(project="my-capstone-project-on-911-call")

911 Calls Capstone Project - Solutions
For this capstone project we will be analyzing some 911 call data from Kaggle. The data contains the following fields:

lat : String variable, Latitude
lng: String variable, Longitude
desc: String variable, Description of the Emergency Call
zip: String variable, Zipcode
title: String variable, Title
timeStamp: String variable, YYYY-MM-DD HH:MM:SS
twp: String variable, Township
addr: String variable, Address
e: String variable, Dummy variable (always 1)
Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!
# In[1]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ls


# In[7]:


df=pd.read_csv('911.csv')


# In[8]:


df.info()


# In[131]:


df.head()


# In[18]:


df['zip'].value_counts().head()


# In[19]:


df['twp'].value_counts().head()


# In[26]:


df['title']


# In[27]:


df['Reason']=df['title'].apply(lambda title:title.split(':')[0])


# In[31]:


df['Reason'].value_counts()


# In[33]:


sns.countplot(x='Reason',data=df)


# In[37]:


type(df['timeStamp'].iloc[0])


# In[39]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# In[140]:


type(df['timeStamp'].iloc[0])


# In[144]:


time=df['timeStamp'].iloc[0]
time


# In[157]:


df['Hour']=df['timeStamp'].apply(lambda time:time.hour)
df['Month']=df['timeStamp'].apply(lambda time:time.month)
df['Day of week']=df['timeStamp'].apply(lambda time:time.dayofweek)


# In[59]:


df['Hour'].unique()


# In[158]:


df['Month'].unique()


# In[64]:


df['Day of week'].unique()


# In[65]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[66]:


df['Day of week']=df['Day of week'].map(dmap)


# In[86]:


df['Day of week'].unique()


# In[84]:


sns.countplot(x='Day of week',data=df,hue='Reason')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[159]:


sns.countplot(x='Month',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.04,1),loc=2,borderaxespad=0.)


# In[160]:


byMonth=df.groupby('Month').count()


# In[161]:


byMonth


# In[162]:


byMonth['twp'].plot()


# In[101]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[118]:


df['Date']=df['timeStamp'].apply(lambda t:t.date())


# In[119]:


df['Date'].head()


# In[120]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[149]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[150]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[163]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()

