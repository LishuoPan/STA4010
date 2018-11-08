
# coding: utf-8

# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[4]:


# read the csv file in to dataframe
df1 = pd.read_csv("./data/Day1.csv",header = 0,sep = ',')
df2 = pd.read_csv("./data/Day2.csv",header = 0,sep = ',')
df3 = pd.read_csv("./data/Day3.csv",header = 0,sep = ',')
df4 = pd.read_csv("./data/Day4.csv",header = 0,sep = ',')


# In[5]:


## change the data from dataframe into the array for further usage
# n: dataset size; d: dimension
Data1 = df1.values
Data2 = df2.values
Data3 = df3.values
Data4 = df4.values
Data = np.vstack((Data1,Data2,Data3,Data4))
# get rid of first line: the time
Data = np.delete(Data, 0, axis=1)
n = Data.shape[0]
d = Data.shape[1]

# complete the dataset:
# principle: in the next line, if there is a nan,
# keep the value same as the last line within same column
for i in range(n):
    for j in range(d):
        if math.isnan(Data[i,j]):
            Data[i,j] = Data[i-1,j]


# In[9]:


print(Data,n)


# In[19]:


time = np.linspace(0,n,n)
BidP = Data[:,0]
Askp = Data[:,2]
Bidsize = Data[:,1]
Asksize = Data[:,3]
plt.plot(time,BidP,label='Bid price')
plt.plot(time,Askp,label='Ask price')
plt.legend()
plt.show()
plt.plot(time,Bidsize,label='Bid size')
plt.plot(time,Asksize,label='Ask size')
plt.legend()
plt.show()


# In[8]:


a = np.linspace(1,10,10)
print(a)

