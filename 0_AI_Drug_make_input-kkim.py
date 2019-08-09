#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
os.getcwd()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#cellLine_list = pd.read_csv('../list file/cellLine_list.txt').values
#mutation_list = pd.read_csv('../list file/mutation_list.txt').values
#drug_list = pd.read_csv('../list file/drug_list.txt').values


# In[4]:


drug_data = pd.read_excel('../..//drugInfo_.xlsx')


# In[5]:


drug_data.shape


# In[6]:


drug_data.head()


# In[7]:


drug_vecs = {}
for _,row in drug_data.iterrows():
    assert(len(row['feature.bin']) == 6543)
    vec = np.asarray([int(x) for x in row['feature.bin']])
    drug_vecs[row['drug.list']] = vec


# In[8]:


cell_line_data = pd.read_excel('../match_count2.xlsx', index_col='Mutation')
cell_line_data = cell_line_data.T


# In[9]:


cell_line_data.head()


# In[10]:


cell_line_vecs = {}
for i,row in cell_line_data.iterrows():
    vec = row.values
    assert(len(vec) == 21213)
    cell_line_vecs[row.name] = vec


# In[11]:


responses_data = pd.read_excel('../learning data.xlsx')


# In[12]:


responses_data.head()


# In[13]:


responses_data['key'] = '%s_%s' % (responses_data['cellLine'],
                                   responses_data['drug'])


# In[14]:


responses_data['key'] = responses_data.apply(lambda x: '%s_%s' % (x.cellLine, x.drug), axis=1)


# In[15]:


responses_data.head()


# In[16]:


clean = responses_data.drop_duplicates(subset=['key'], keep=False)


# In[17]:


clean.shape


# In[18]:


dup = responses_data[~responses_data['key'].isin(clean['key'])]


# In[19]:


dup.head()


# In[20]:


dup1 = dup.drop_duplicates(subset=['key'], keep='first')
dup2 = dup.drop_duplicates(subset=['key'], keep='last')


# In[21]:


dup_merge = dup1.merge(dup2, on='key')


# In[22]:


dup_merge.head()


# In[23]:


dup_merge.to_excel('duplicated_learning_data.xlsx')


# In[24]:


dup_merge.plot.scatter(x='IC50(LN)_x', y = 'IC50(LN)_y')


# In[25]:


# decide to drop all of duplicated keys from responses_data
responses_clean = clean


# In[26]:


responses_data.shape


# In[27]:


responses_clean.shape


# In[28]:


keys = []
cvecs = []
dvecs = []
ln_ic50s = []

count = 0
for n, (i,row) in enumerate(responses_clean.iterrows()):
    if n % 10000 == 0:
        print('%d / %d | %d %d' % (n, len(responses_clean), len(keys),                                   count))
    name = row['cellLine']
    drug = row['drug']
    ln_ic50 = float(row['IC50(LN)'])
    
    try:
        cell_line_vec = cell_line_vecs[name]
        drug_vec = drug_vecs[drug]
    except Exception:
        count += 1
        continue;
    key = '%s_%s' % (name, drug)
    cvecs.append(cell_line_vec)
    dvecs.append(drug_vec)
    ln_ic50s.append(ln_ic50)
    keys.append(key)
print('%d / %d | %d %d' % (n, len(responses_clean), len(keys),                            count))


# In[29]:


len(keys), len(cvecs), len(dvecs), len(ln_ic50s)


# In[30]:


keys = np.asarray(keys)
cvecs = np.asarray(cvecs, dtype='int32')
dvecs = np.asarray(dvecs, dtype='int32')
ln_ic50s = np.asarray(ln_ic50s, dtype='float32')


# In[31]:


keys.shape, cvecs.shape, dvecs.shape, ln_ic50s.shape


# In[32]:


from sklearn.preprocessing import KBinsDiscretizer


# In[33]:


kbd10 = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
kbd24 = KBinsDiscretizer(n_bins=24, encode='ordinal', strategy='uniform')


# In[34]:


kbd10.fit(ln_ic50s.reshape(-1,1))


# In[35]:


kbd24.fit(ln_ic50s.reshape(-1,1))


# In[36]:


cls10 = kbd10.transform(ln_ic50s.reshape(-1,1)).reshape(-1).astype(np.int32)
cls24 = kbd24.transform(ln_ic50s.reshape(-1,1)).reshape(-1).astype(np.int32)


# In[37]:


from tensorflow.keras.utils import to_categorical


# In[38]:


cls10_onehot = to_categorical(cls10).astype('int32')
cls24_onehot = to_categorical(cls24).astype('int32')


# In[39]:


cls10


# In[40]:


cls10_onehot


# In[41]:


np.savez_compressed('20190217_kkim_cls10.npz', x=np.c_[cvecs,dvecs], y=cls10_onehot, y_labels=cls10, y_lnIC50=ln_ic50s)


# In[42]:


np.savez_compressed('20190217_kkim_cls24.npz', x=np.c_[cvecs,dvecs], y=cls24_onehot, y_labels=cls24, y_lnIC50=ln_ic50s)


# with open('training_image_array.csv','w') as t_image:
#     print(training_image_array)
# with open('training_label_array_2.csv','w') as l_image:
#     print(training_label_array)
