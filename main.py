import streamlit as st
import pickle
import pandas as pd
import requests

movies_dict= pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

sim_vec = pickle.load(open('sim_vec.pkl','rb'))

def fetch_poster(id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=9c58c24332472f67fc33aa878d7fa0ae&language=en-US'.format(id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

def recommender(movie):
    mov_ind = movies[movies['title']==movie].index[0] #to get index of that movie
    dist = sim_vec[mov_ind] #to get the similarlity vector of that particular movie for comparison
    mov_lst = sorted(list(enumerate(dist)),reverse=True,key = lambda x:x[1])[1:6] #sorting the similarlity list to get the most similar movies and picking most 5 similar movies
    recommended_name=[]
    recommended_poster=[]
    for i in mov_lst:
        id = movies.iloc[i[0]].movie_id
        recommended_name.append(movies.iloc[i[0]].title)
        recommended_poster.append(fetch_poster(id))
    return recommended_name,recommended_poster

st.title('Movie Recommender System')
select_movie = st.selectbox('Select Movie',movies['title'].values)

if st.button('Recommend'):
    names,poster = recommender(select_movie)
    
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(poster[0])
    with col2:
        st.text(names[1])
        st.image(poster[1])
    with col3:
        st.text(names[2])
        st.image(poster[2])
    with col4:
        st.text(names[3])
        st.image(poster[3])
    with col5:
        st.text(names[4])
        st.image(poster[4])