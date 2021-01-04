#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('merged.csv')


# In[4]:


data.head()


# # Template of our dataset 

# In[5]:


temp = data.drop(['Column_id','tdate_x','Unnamed: 0'], axis = 1)


# In[7]:


temp.head()


# In[11]:


y = temp['ttype_x']
y.shape


# In[9]:


temp.drop('ttype_x', axis = 1, inplace = True)


# In[12]:


temp.shape


# In[13]:


from sklearn.naive_bayes import GaussianNB


# In[14]:


model = GaussianNB()


# # Convert our words to vectors

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


bow_transformer = CountVectorizer().fit(temp['ttext_x'])
print(len(bow_transformer.vocabulary_))


# In[19]:


bow4=bow_transformer.fit_transform(temp['ttext_x'])
print(bow4)
print(bow4.shape)


# In[20]:


messages_bow = bow_transformer.fit_transform(temp['ttext_x'])


# In[21]:


print('Shape of Sparse Matrix: ',messages_bow.shape)
print('Amount of non-zero occurences:',messages_bow.nnz)


# In[22]:


sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
print('sparsity:{}'.format(round(sparsity)))


# # TF-IDF

# In[23]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.fit_transform(bow4)
print(tfidf4)


# In[24]:


messages_tfidf=tfidf_transformer.fit_transform(messages_bow)
print(messages_tfidf.shape)


# # Train model

# In[25]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(messages_tfidf,y)


# In[28]:


print('predicted:',model.predict(tfidf4)[:10])
print('expected:',y[3])


# # F1 score

# In[29]:


all_predictions = model.predict(messages_tfidf)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y,all_predictions))
print(confusion_matrix(y,all_predictions))


# # Add new column "predict"

# In[31]:


data['predict'] = model.predict(tfidf4)
data.drop(['ttype_x','pred'], axis = 1, inplace = True)


# In[32]:


data.head()


# # Test with parsed news

# In[175]:


df = pd.read_csv('nur.csv', encoding='utf-8', comment='#', sep = ';')


# In[176]:


df.drop('Unnamed: 1', axis = 1, inplace = True)


# In[177]:


df.head()


# # Preparing parsed data

# In[178]:


bow_transformer.transform(df['Column_id'])
print(len(bow_transformer.vocabulary_))


# In[179]:


bow5=bow_transformer.transform(df['Column_id'])
print(bow5)
print(bow5.shape)


# In[180]:


new_messages_bow = bow_transformer.transform(df['Column_id'])


# In[181]:


print('Shape of Sparse Matrix: ',new_messages_bow.shape)
print('Amount of non-zero occurences:',new_messages_bow.nnz)


# In[182]:


tfidf_transformer.fit(new_messages_bow)
tfidf5 = tfidf_transformer.transform(bow5)
print(tfidf5)


# In[183]:


new_messages_tfidf=tfidf_transformer.transform(new_messages_bow)
print(new_messages_tfidf.shape)


# In[224]:


df['predicted'] = model.predict(tfidf5)


# In[226]:


df


# # Save trained model

# In[33]:


import numpy as np
from flask import Flask, request, jsonify
import pickle


# In[35]:


pickle.dump(spam_detect_model, open('model.pkl','wb'))


# In[219]:


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


# In[220]:


@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)


# In[228]:


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# In[227]:


import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':1.8,})
print(r.json())

