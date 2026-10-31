
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Audible Insights",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Audible Insights")
st.subheader("Intelligent Book Recommendation System")
st.write(
    "Explore Audible books, analyze catalogue trends, "
    "and discover books based on your interests."
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv("data/audible_cleaned.csv")

    for col in ["Book Name", "Author", "Description", "Genre"]:
        data[col] = data[col].fillna("Unknown").astype(str)

    data["Rating"] = pd.to_numeric(
        data["Rating"], errors="coerce"
    )
    data["Number of Reviews"] = pd.to_numeric(
        data["Number of Reviews"], errors="coerce"
    )

    data["Genre_List"] = (
        data["Genre"]
        .apply(lambda x: [
            g.strip() for g in x.split("|")
            if g.strip() and g.strip() != "Unknown"
        ])
    )

    return data


df = load_data()

# ==========================================
# PREPARE RECOMMENDATION FEATURES
# ==========================================

@st.cache_resource
def prepare_recommendation_features(data):

    valid = data[
        (data["Description"].str.lower() != "description not available")
        & (data["Description"].str.strip() != "")
    ].copy()

    valid = valid.reset_index(drop=True)

    valid["Text"] = (
        valid["Book Name"] + " " +
        valid["Author"] + " " +
        valid["Description"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=2
    )

    matrix = vectorizer.fit_transform(valid["Text"])

    return valid, matrix


valid_books, tfidf_matrix = prepare_recommendation_features(df)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Book Explorer",
        "Genre Recommendations",
        "Content-Based Recommendations",
        "EDA Dashboard"
    ]
)

# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.header("Welcome to Audible Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Books", f"{len(df):,}")
    col2.metric("Unique Authors", f"{df['Author'].nunique():,}")
    col3.metric("Average Rating", f"{df['Rating'].mean():.2f} ⭐")

    st.markdown("### About the Project")

    st.write(
        "Audible Insights is an intelligent book recommendation "
        "system developed using Python, Machine Learning, NLP, "
        "and Streamlit."
    )

    st.write(
        "The application helps users explore books, understand "
        "catalogue trends, and discover similar books."
    )

# ==========================================
# BOOK EXPLORER
# ==========================================

elif page == "Book Explorer":

    st.header("🔍 Book Explorer")

    search = st.text_input("Search by book name or author")

    filtered = df.copy()

    if search:
        filtered = filtered[
            filtered["Book Name"].str.contains(
                search, case=False, na=False
            )
            | filtered["Author"].str.contains(
                search, case=False, na=False
            )
        ]

    st.write(f"Books found: {len(filtered)}")

    st.dataframe(
        filtered[
            [
                "Book Name",
                "Author",
                "Rating",
                "Number of Reviews",
                "Genre"
            ]
        ].head(100),
        use_container_width=True
    )

# ==========================================
# GENRE RECOMMENDATIONS
# ==========================================

elif page == "Genre Recommendations":

    st.header("🎯 Genre-Based Recommendations")

    all_genres = sorted({
        genre
        for genres in df["Genre_List"]
        for genre in genres
    })

    selected_genre = st.selectbox(
        "Choose your favorite genre",
        all_genres
    )

    if st.button("Recommend Books"):

        recommendations = df[
            df["Genre_List"].apply(
                lambda genres: selected_genre in genres
            )
        ].copy()

        recommendations = recommendations.sort_values(
            ["Rating", "Number of Reviews"],
            ascending=[False, False]
        ).drop_duplicates(
            subset=["Book Name", "Author"]
        ).head(5)

        st.subheader(f"Top Books in {selected_genre}")

        if recommendations.empty:
            st.info("No books found for this genre.")
        else:
            st.dataframe(
                recommendations[
                    [
                        "Book Name",
                        "Author",
                        "Rating",
                        "Number of Reviews",
                        "Genre"
                    ]
                ],
                use_container_width=True
            )

# ==========================================
# CONTENT-BASED RECOMMENDATIONS
# ==========================================

elif page == "Content-Based Recommendations":

    st.header("📖 Content-Based Book Recommendations")

    book_names = sorted(valid_books["Book Name"].unique())

    selected_book = st.selectbox(
        "Select a book you like",
        book_names
    )

    if st.button("Find Similar Books"):

        matches = valid_books[
            valid_books["Book Name"] == selected_book
        ]

        selected_index = matches.index[0]

        scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()

        scores[selected_index] = -1

        top_indices = scores.argsort()[::-1][:5]

        results = valid_books.iloc[top_indices][
            ["Book Name", "Author", "Rating", "Genre"]
        ].copy()

        results["Similarity Score"] = scores[top_indices]

        st.subheader("Recommended Books")

        st.dataframe(
            results,
            use_container_width=True
        )

# ==========================================
# EDA DASHBOARD
# ==========================================

elif page == "EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        genre_counts = (
            df[df["Genre"] != "Unknown"]["Genre_List"]
            .explode()
            .value_counts()
            .head(10)
            .reset_index()
        )

        genre_counts.columns = ["Genre", "Number of Books"]

        fig_genre = px.bar(
            genre_counts,
            x="Number of Books",
            y="Genre",
            orientation="h",
            title="Top 10 Popular Genres"
        )

        st.plotly_chart(fig_genre, use_container_width=True)

    with col2:
        fig_rating = px.histogram(
            df,
            x="Rating",
            nbins=20,
            title="Book Rating Distribution"
        )

        st.plotly_chart(fig_rating, use_container_width=True)

    fig_reviews = px.scatter(
        df,
        x="Number of Reviews",
        y="Rating",
        title="Ratings vs Number of Reviews",
        hover_data=["Book Name", "Author"]
    )

    st.plotly_chart(fig_reviews, use_container_width=True)

    st.subheader("Dataset Summary")

    st.dataframe(
        df[
            ["Book Name", "Author", "Rating",
             "Number of Reviews", "Genre"]
        ].head(20),
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.sidebar.markdown("---")
st.sidebar.caption("Audible Insights | Intelligent Book Recommendations")
