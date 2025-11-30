import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1) Load & preprocess data
# -------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    df = pd.read_csv(csv_path)

    # Clean Keywords column
    df["Từ khóa"] = df["Từ khóa"].fillna("").str.replace(";", " ")

    # Combine text for TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("")
    )

    # Normalize image links
    if "Link ảnh" in df.columns:
        df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()

# -------------------------
# 2) TF-IDF + similarity matrix
# -------------------------
@st.cache_data
def build_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["FullText"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer(df)

# -------------------------
# 3) Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Demo CBF for small business",
    layout="wide"
)

st.title("Welcome to our store! Products from Adidas, Nike, Lacoste, Puma, Gucci")
st.write("Our system will find products based on your input (description/keywords).")

user_query = st.text_input("Please enter your product description or keywords:")

top_k = st.number_input("Number of recommendations (top K):", min_value=1, max_value=10, value=5)

threshold = 0.1  # minimum similarity to show

if user_query:
    # Transform user input
    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranking = scores.argsort()[::-1]

    # Check if the best match meets threshold
    if scores[ranking[0]] < threshold:
        st.warning("Can't find any product, please try again with different keywords.")
    else:
        st.subheader("Main product match:")

        best_idx = ranking[0]

        # Show main product info
        if "Link ảnh" in df.columns and df.loc[best_idx, "Link ảnh"]:
            st.image(df.loc[best_idx, "Link ảnh"], width=250)

        st.write(f"**Tên:** {df.loc[best_idx, 'Tên sản phẩm']}")
        st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
        st.write(f"**Giá:** {df.loc[best_idx, 'Giá']}")
        st.write(f"**Thương hiệu:** {df.loc[best_idx, 'Thương hiệu']}")
        st.write(f"Điểm đánh giá: {df.loc[best_idx, 'Điểm đánh giá']}")
        st.write(f"**Similarity:** `{scores[best_idx]:.3f}`")

        st.subheader("You may also like:")

        # Show top-K recommendations (excluding main product)
        count = 0
        for idx in ranking[1:]:
            if scores[idx] < threshold or count >= top_k:
                break

            if "Link ảnh" in df.columns and df.loc[idx, "Link ảnh"]:
                st.image(df.loc[idx, "Link ảnh"], width=180)

            st.write(f"**Tên:** {df.loc[idx, 'Tên sản phẩm']}")
            st.write(f"**Mô tả:** {df.loc[idx, 'Mô tả']}")
            st.write(f"Giá: {df.loc[idx, 'Giá']}")
            st.write(f"**Thương hiệu:** {df.loc[idx, 'Thương hiệu']}")
            st.write(f"Điểm đánh giá: {df.loc[idx, 'Điểm đánh giá']}")
            st.write(f"Similarity: `{scores[idx]:.3f}`")
            st.write("---")
            count += 1
