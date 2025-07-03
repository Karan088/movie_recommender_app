# Movie Recommendation System
This project builds a hybrid movie recommender system combining SVD collaborative filtering and item-based collaborative filtering. It predicts user ratings for unseen movies and suggests related movies based on similarity.

```
movie_recommender_app/
│
├── data/
│   └── Movie100K-Kaggle_link.txt        # Contains the Kaggle dataset URL
│
├── deployment/
│   ├── app.py                  # Streamlit app code
│   ├── item_similarity_df.pkl   # Saved item-based similarity matrix
│   ├── movie_metadata.pkl       # Metadata (title, genres, etc.)
│   ├── svd_model.pkl            # Trained SVD model
│   └── requirements.txt         # Dependencies for the app
│
│── movie-recommendation-cf.ipynb  # Notebook with full analysis
│
└── Movie Recommendation System.pdf   # Final project report

```


🚫 Deployment Status

The Streamlit app deployment did not succeed because:

The scikit-surprise package is difficult to build on many cloud environments (such as Streamlit Cloud) due to Cython compilation issues.

The build process fails when trying to compile required components during deployment.
