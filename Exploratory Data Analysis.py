#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from textblob import TextBlob


# In[2]:


traindata = pd.read_csv("train.csv", index_col = [0])


# In[3]:


traindata.head()


# In[4]:


traindata.info() # have 7613 text and 7552 keyword, maybe removed the text that does not providing keyword?
                 # But the text without location is huge number, maybe just leave it as this? 


# In[5]:


sns.heatmap(traindata.isnull(),yticklabels = False, cbar = False, cmap = "viridis")


# In[6]:


df_train = pd.DataFrame(traindata)


# In[7]:


df_train["target"].value_counts() # 1 means real disaster and 0 means fake ones. More real ones than fake ones


# In[8]:


df_train["keyword"].value_counts() # most 'popular' word here is fatalities, and we have about 221 different
                                   # words, each with different frequency. We want to look more than just keyword, 
                                   # but could start with keywords


# In[9]:


df_train["length"] = df_train["text"].apply(len)
df_train.head()


# In[10]:


df_train.hist(column = "length", by = "target", bins = 50, figsize = (20,10)) # rough idea of the distribution 
                                                                              # of twitter length for 0 and 1


# In[11]:


df_train['target_count'] = df_train.groupby('keyword')['target'].transform('mean')


# In[12]:


fig = plt.figure(figsize=(8, 72), dpi=100)
sns.countplot(y = df_train.sort_values(by='target_count', ascending = False)["keyword"],
              hue=df_train.sort_values(by='target_count', ascending=False)['target'])  
# This is directly from the competition coding script, I think it's really nice to have, but may need to write 
# it with different style


# In[18]:


corpus = open("train.csv").read()
ngram_object = TextBlob(corpus)
gram_2 = ngram_object.ngrams(n = 2)
gram_3 = ngram_object.ngrams(n = 3)


# In[19]:


print(gram_2[:10])


# In[20]:


print(gram_3[:10])


# In[ ]:




