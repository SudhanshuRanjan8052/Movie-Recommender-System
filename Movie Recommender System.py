#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# # Data Pre-Processing

# In[120]:


credits = pd.read_csv("tmdb_5000_credits.csv")


# In[121]:


keywords = pd.read_csv("tmdb_5000_movies.csv")


# In[124]:


movies = credits.merge(keywords,on="title")


# In[125]:


movies


# In[126]:


#important columns: genres, id, title, keywords, overview, cast, crew
movies = movies[['movie_id','title','genres','overview','keywords','cast','crew']]


# In[127]:


movies #working dataset with important features


# In[128]:


#checking null values
movies.isnull().sum()


# In[129]:


movies.dropna(inplace=True) #removing null values


# In[130]:


movies.shape #After removing null values


# In[131]:


#checking duplicate values
movies.duplicated().sum()


# In[132]:


movies=movies.drop_duplicates() #removing duplicate values


# In[133]:


movies.shape #After removing duplicate values


# In[134]:


type(movies.iloc[0].genres) #as it is in string form so we have to convert this string into list so that we can extract information


# In[135]:


import ast
def convert(obj):
    r=[]
    for i in ast.literal_eval(obj): #to convert string literals into list
        r.append(i['name'])
    return r


# In[136]:


movies['genres']= movies['genres'].apply(convert) #applying on genres column


# In[137]:


movies.head()


# In[138]:


movies['keywords']= movies['keywords'].apply(convert) #applying on keywords column


# In[139]:


movies.head()


# In[140]:


def convert_cast(obj): #for cast column
    c=[]
    cnt=0
    for i in ast.literal_eval(obj): #to convert string literals into list
        if cnt!=5:   
            c.append(i['name'])
            cnt+=1
        else:
            break
    return c


# In[141]:


movies['cast'] = movies['cast'].apply(convert_cast) #applying on cast column


# In[142]:


movies.head()


# In[143]:


def convert_crew(obj):
    d=[]
    for i in ast.literal_eval(obj): #to convert string literals into list
        if i['job'] == 'Director':
            d.append(i['name'])
    return d


# In[144]:


movies['crew'] = movies['crew'].apply(convert_crew) #applying on crew column


# In[145]:


movies.head()


# In[146]:


movies['overview'] = movies['overview'].apply(lambda x:x.split()) #converting string into list so that we can concatenate it with others list


# In[147]:


movies.head()


# In[148]:


'''
removing spaces between chracters so that if we finding some movies then it will provide much accurate like if we are
finding tom holland and there is a tom hanks then model will get confuse whom to find but if there are no spaces then tom 
holland is considered as a whole word like tomholland so it will recommend the correct movie.
'''
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[149]:


movies.head()


# In[150]:


#putting all neccessary tags into one column
movies['tags'] =movies['overview'] + movies['keywords'] + movies['genres'] + movies['cast'] + movies['crew']


# In[151]:


movies.head()


# In[153]:


final_df = movies[['movie_id','title','tags']] 


# In[154]:


final_df


# In[155]:


final_df['tags'] = final_df['tags'].apply(lambda x:" ".join(x))


# In[156]:


final_df.head()


# In[157]:


final_df['tags'] = final_df['tags'].apply(lambda x:x.lower())


# In[158]:


final_df.head()


# In[159]:


from nltk import PorterStemmer 
ps = PorterStemmer()


# In[160]:


def stem(text): #stemming the tags columns so that after vectorization we would not gett repeated words 
    lst =[]
    for i in text.split():
        lst.append(ps.stem(i))
    return " ".join(lst)


# In[161]:


final_df['tags'] = final_df['tags'].apply(stem)


# In[165]:


#vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[166]:


vec = cv.fit_transform(final_df['tags']).toarray()


# In[167]:


vec.shape


# In[168]:


#similarlity calculation
#it will calculate the similarlity on the basis of angle between them
#range lies between 0 and 1 if it is 1 then full similar and if it is 0 then disimilar and in this way it depends on values
from sklearn.metrics.pairwise import cosine_similarity 


# In[169]:


sim_vec = cosine_similarity(vec) 


# In[170]:


sim_vec.shape


# # Model Building

# In[171]:


#function to recommend movie
def recommend(movie):
    mov_ind = final_df[final_df['title']==movie].index[0] #to get index of that movie
    dist = sim_vec[mov_ind] #to get the similarlity vector of that particular movie for comparison
    mov_lst = sorted(list(enumerate(dist)),reverse=True,key = lambda x:x[1])[1:6] #sorting the similarlity list to get the most similar movies and picking most 5 similar movies
    for i in mov_lst:
        print(final_df.iloc[i[0]].title)


# In[172]:


recommend('Superman')


# In[173]:


import pickle
pickle.dump(final_df.to_dict(),open('movie_dict.pkl','wb')) #for providing movie list to the website


# In[174]:


pickle.dump(sim_vec,open('sim_vec.pkl','wb')) #for providing similarlity vector 


# In[ ]:




