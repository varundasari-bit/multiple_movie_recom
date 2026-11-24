import streamlit as st
from urllib.parse import quote_plus

from recommender import recommend, movies
from revenue_prediction import model_metrics, predict_revenue

st.set_page_config(page_title="Movie ML Platform", page_icon="🎬")
st.title("🎬 Movie Intelligence Platform")

recommendation_tab, revenue_tab = st.tabs(["Movie Recommendation", "Revenue Prediction"])

with recommendation_tab:
    st.header("Movie Recommendation")
    # recommender.movies is a list of title dictionaries, not a pandas DataFrame.
    movie_titles = sorted(movie["title"] for movie in movies)
    selected_movie = st.selectbox("Choose a movie", movie_titles)
    if st.button("Recommend Movies"):
        results = recommend(selected_movie)
        if results:
            st.caption("Select a movie title to open its Wikipedia search in a new tab.")
            for number, title in enumerate(results, start=1):
                # Search Wikipedia by title so alternate titles and disambiguation pages work.
                wikipedia_url = (
                    "https://en.wikipedia.org/w/index.php?title=Special:Search&search="
                    + quote_plus(title)
                )
                # Escape markdown delimiters that may occur in a movie title.
                link_title = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
                st.markdown(f"{number}. [{link_title}]({wikipedia_url})")
        else:
            st.info("No recommendations found for that title.")

with revenue_tab:
    st.header("Movie Revenue Prediction")
    st.caption("A simple linear regression estimate based on the bundled TMDB dataset.")
    budget = st.number_input("Movie Budget ($)", min_value=1.0, value=10_000_000.0, step=1_000_000.0)
    runtime = st.number_input("Runtime (minutes)", min_value=1, value=120, step=1)
    release_year = st.number_input("Release Year", min_value=1800, max_value=2100, value=2026, step=1)

    if st.button("Predict Revenue"):
        try:
            revenue = predict_revenue(budget, runtime, release_year)
            st.success(f"Estimated Revenue: ${revenue:,.2f}")
        except (TypeError, ValueError) as error:
            st.error(str(error))

    with st.expander("Model evaluation"):
        st.write(f"Training rows: {model_metrics['training_rows']:,}")
        st.write(f"Test rows: {model_metrics['test_rows']:,}")
        st.write(f"Mean absolute error: ${model_metrics['mae']:,.2f}")
        st.write(f"Root mean squared error: ${model_metrics['rmse']:,.2f}")
        st.write(f"R²: {model_metrics['r2']:.3f}")
