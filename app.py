import streamlit as st

# The recommender uses local CSV files and standard-library parsing.
from recommender import recommend, movies

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

# Pass a plain list of titles so the app does not need dataframe functionality.
movie_list = sorted(movie["title"] for movie in movies)
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend Movies"):
    results = recommend(selected_movie)
    st.subheader("Recommended Movies")
    if results:
        for number, movie in enumerate(results, start=1):
            st.write(f"**{number}.** 🎥 {movie}")
    else:
        st.warning("No recommendations found.")
