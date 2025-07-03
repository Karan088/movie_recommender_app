import streamlit as st
import pandas as pd
import pickle

# Load models + data
@st.cache_resource
def load_svd_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_item_similarity():
    return pd.read_pickle('item_similarity_df.pkl')

@st.cache_data
def load_movie_metadata():
    return pd.read_pickle('movie_metadata.pkl')

svd_model = load_svd_model()
item_similarity_df = load_item_similarity()
movie_metadata = load_movie_metadata()

# Recommend SVD
def recommend_svd(user_id, top_n=10, min_pred_rating=3.5):
    rated = set(cf_ratings_agg.loc[cf_ratings_agg['user_id'] == user_id, 'movie_id'])
    unseen = [mid for mid in movie_metadata['movie_id'].unique() if mid not in rated]
    preds = [(mid, svd_model.predict(user_id, mid).est) for mid in unseen]
    preds = [p for p in preds if p[1] >= min_pred_rating]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    recs = movie_metadata[movie_metadata['movie_id'].isin([mid for mid, _ in top_preds])].copy()
    recs['predicted_rating'] = recs['movie_id'].map(dict(top_preds))
    return recs.sort_values(by='predicted_rating', ascending=False)

# Recommend CF
def recommend_cf(movie_id, top_n=10, min_sim=0.1):
    if movie_id not in item_similarity_df.index:
        return pd.DataFrame()
    scores = item_similarity_df[movie_id].sort_values(ascending=False).iloc[1:]
    scores = scores[scores >= min_sim].head(top_n)
    recs = movie_metadata[movie_metadata['movie_id'].isin(scores.index)].copy()
    recs['similarity_score'] = recs['movie_id'].map(scores)
    return recs.sort_values(by='similarity_score', ascending=False)

# Streamlit UI
st.title("🎬 Movie Recommender")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)
min_rating = st.slider("Minimum predicted rating (SVD)", 1.0, 5.0, 3.5, 0.1)
if st.button("Get Recommendations"):
    svd_recs = recommend_svd(user_id, top_n=10, min_pred_rating=min_rating)
    st.subheader("SVD Recommendations")
    st.dataframe(svd_recs)

    if not svd_recs.empty:
        movie_id = st.selectbox("Select a movie to see similar", svd_recs['movie_id'])
        cf_recs = recommend_cf(movie_id, top_n=10)
        st.subheader("Item-based CF Recommendations")
        st.dataframe(cf_recs)
