#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


df = pd.read_csv('movie_genre.csv')
df.head(10)

movies=df[0:20000]
plt.figure(figsize=(12,12))
sns.countplot(x='genre', data=movies)
plt.xlabel('Movie Genres')
plt.ylabel('Count')
plt.title('Genre Plot')
plt.show()


# In[85]:


movies['text'][200]


# In[63]:


movie_genre = list(movies['genre'].unique())
movie_genre.sort()
movie_genre


# In[64]:


genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 
                'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
movies['genre'] = movies['genre'].map(genre_mapper)
movies.head(10)


# In[65]:


movies.drop('id', axis=1, inplace=True)


# In[66]:


corpus = []
ps = PorterStemmer()

for i in range(0, movies.shape[0]):
    dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=movies['text'][i]) # Cleaning special character from the dialog/script
    dialog = dialog.lower() # Converting the entire dialog/script into lower case
    words = dialog.split() # Tokenizing the dialog/script by words
    dialog_words = [word for word in words if word not in set(stopwords.words('english'))] # Removing the stop words
    words = [ps.stem(word) for word in dialog_words] # Stemming the words
    dialog = ' '.join(words) # Joining the stemmed words
    corpus.append(dialog) # Creating a corpus


# drama_words = []
# for i in list(movies1[movies1['genre_new']==3].index):
#     drama_words.append(corpus[i])

# action_words = []
# for i in list(movies1[movies1['genre_new']==1].index):
#     action_words.append(corpus[i])

# comedy_words = []
# for i in list(movies1[movies1['genre_new']==2].index):
#     comedy_words.append(corpus[i])


# In[93]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=300, ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()

y = movies['genre'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))


# In[73]:


print(X.shape)


# In[94]:


from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)


# In[95]:


nb_y_pred = nb_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score1 = accuracy_score(y_test, nb_y_pred)
print("---- Score ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))


# In[92]:


import pickle
pickle.dump(nb_classifier, open('movie1.pkl', 'wb'))


# In[21]:


import pickle
pickle.dump(cv, open('count_vec.pkl', 'wb'))


# In[ ]:




