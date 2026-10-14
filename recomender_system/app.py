
import streamlit as st
import pickle
import requests
import os
import pandas as pd
import numpy as np
from collections import Counter


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


BASE_DIR = os.path.dirname(__file__)
MOVIE_DICT_PATH = os.path.join(BASE_DIR, 'movie_dict.pkl')
SIMILARITY_PATH = os.path.join(BASE_DIR, 'similarity.pkl')


movies_dict = pickle.load(open(MOVIE_DICT_PATH, 'rb'))
movies = pd.DataFrame(movies_dict)

def compute_similarity_from_tags(df, max_features=5000):
    tags = df['tags'].fillna('').astype(str).values
    # try sklearn first for speed
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        cv = CountVectorizer(max_features=max_features, stop_words='english')
        vectors = cv.fit_transform(tags).toarray()
        sim = cosine_similarity(vectors)
        return sim
    except Exception:
        counts = Counter()
        docs_tokens = []
        for t in tags:
            toks = [tok for tok in t.lower().split() if tok]
            docs_tokens.append(toks)
            counts.update(toks)

        vocab = [w for w,_ in counts.most_common(max_features)]
        idx = {w:i for i,w in enumerate(vocab)}

        vectors = np.zeros((len(tags), len(vocab)), dtype=np.float32)
        for i,toks in enumerate(docs_tokens):
            for tok in toks:
                j = idx.get(tok)
                if j is not None:
                    vectors[i,j] += 1

        norms = np.linalg.norm(vectors, axis=1)
        norms[norms==0] = 1.0
        sim = (vectors @ vectors.T) / (norms[:,None] * norms[None,:])
        return sim

if os.path.exists(SIMILARITY_PATH):
    try:
        similarity = pickle.load(open(SIMILARITY_PATH, 'rb'))
    except Exception:
        similarity = compute_similarity_from_tags(movies)
        pickle.dump(similarity, open(SIMILARITY_PATH, 'wb'))
else:
    similarity = compute_similarity_from_tags(movies)
    pickle.dump(similarity, open(SIMILARITY_PATH, 'wb'))


API_KEY = os.environ.get("TMDB_API_KEY")


def fetch_poster(movie_id):
    if not API_KEY:
        return "https://via.placeholder.com/500x750?text=No+Poster"

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"

        data = requests.get(url).json()

        poster_path = data.get('poster_path')

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"

    except:
        pass

    return "https://via.placeholder.com/500x750?text=No+Poster"


def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(
            movies.iloc[i[0]].title
        )

        recommended_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movies, recommended_posters


st.title("🎬 Movie Recommendation System")

if not API_KEY:
    st.info("Set the TMDB_API_KEY environment variable to load movie posters.")

st.markdown(
    """
    Select a movie and get recommendations based on
    genres, keywords, cast, director and overview.
    """
)

selected_movie = st.selectbox(
    "Choose a Movie",
    movies['title'].values
)

if st.button("Recommend Movies"):

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.caption(names[0])

    with col2:
        st.image(posters[1])
        st.caption(names[1])

    with col3:
        st.image(posters[2])
        st.caption(names[2])

    with col4:
        st.image(posters[3])
        st.caption(names[3])

    with col5:
        st.image(posters[4])
        st.caption(names[4])

st.markdown("---")
st.markdown("Built with Streamlit ❤️")