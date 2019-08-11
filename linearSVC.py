#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from pandas import DataFrame
from datetime import datetime
from sklearn.externals import joblib
import pickle

os.getcwd()


# In[2]:


# loading your test dataset file 
testset = np.load('/path/test_dataset.npz')


# In[4]:


# test dataset 
test_image_array, test_label_array = testset['x'], testset['y']


# In[7]:


num_classes = 3


# In[8]:


test_X, test_y = test_image_array, test_label_array


# In[9]:


test_y_arg = np.argmax(test_y,axis=1)


# In[11]:


filename = 'linearsvc_model.sav'


# In[ ]:


joblib_model = joblib.load(filename)


# In[14]:


from sklearn.metrics import accuracy_score
job_svm_y_predict = joblib_model.predict(test_X) 
score = accuracy_score(test_y_arg, job_svm_y_predict)
print(score) #1.0


# In[25]:


svm_y_predict_proba = joblib_model.predict_proba(test_X)


# In[15]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y_arg, job_svm_y_predict, labels=[0,1,2])


# In[28]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y_arg, svm_y_predict, target_names=target_names))


# In[29]:


import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
svm_fpr = dict()
svm_tpr = dict()
svm_roc_auc = dict()
for i in range(num_classes):
    svm_fpr[i], svm_tpr[i], _ = roc_curve(test_y[:, i], svm_y_predict_proba[:, i])
    svm_roc_auc[i] = auc(svm_fpr[i], svm_tpr[i])


# In[30]:


svm_fpr["micro"], svm_tpr["micro"], _ = roc_curve(test_y.ravel(), svm_y_predict_proba.ravel())
svm_roc_auc["micro"] = auc(svm_fpr["micro"], svm_tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
svm_all_fpr = np.unique(np.concatenate([svm_fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
svm_mean_tpr = np.zeros_like(svm_all_fpr)
for i in range(num_classes):
    svm_mean_tpr += interp(svm_all_fpr, svm_fpr[i], svm_tpr[i])

# Finally average it and compute AUC
svm_mean_tpr /= (num_classes)

svm_fpr["macro"] = svm_all_fpr
svm_tpr["macro"] = svm_mean_tpr
svm_roc_auc["macro"] = auc(svm_fpr["macro"], svm_tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(svm_fpr["micro"], svm_tpr["micro"],
         label='micro-average (area = {0:0.2f})'
               ''.format(svm_roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(svm_fpr["macro"], svm_tpr["macro"],
         label='macro-average (area = {0:0.2f})'
               ''.format(svm_roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(svm_fpr[i], svm_tpr[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
             ''.format(i, svm_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GDSC SVM ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[31]:


svm_fpr["micro"], svm_tpr["micro"]


# In[32]:


svm_fpr["macro"], svm_tpr["macro"]


# In[33]:


svm_fpr[0] , svm_tpr[0]


# In[34]:


svm_fpr[1] , svm_tpr[1]


# In[35]:


svm_fpr[2] , svm_tpr[2]


# In[ ]:




