import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================
# 1. LOAD DATA
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("Gr6.csv")

    # Clean essential columns
    df["Tên sản phẩm"] = df["Tên sản phẩm"].astype(str).str.strip()
    df["Mô tả"] = df["Mô tả"].astype(str).fillna("")
    df["Thương hiệu"] = df["Thương hiệu"].astype(str).fillna("")
    df["Từ khóa"] = df["Từ khóa"].astype(str).fillna("")
    df["Loại sản phẩm"] = df["Loại sản phẩm"].astype(str).str.strip()

    # Create combined text for TF-IDF
    df["combined_text"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Thương hiệu"] + " " +
        df["Từ khóa"]
    ).str.lower()

    return df


df = load_data()

# Build TF-IDF model
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


# ===========================
# 2. FIND BEST PRODUCT MATCH
# ===========================
def find_best_match(query):
    query_clean = re.sub(r"[^\w\s]", " ", query.lower())
    vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()

    best_idx = scores.argmax()
    best_score = scores[best_idx]
    return best_idx, best_score


# ===========================
# 3. RECOMMEND SAME CATEGORY ONLY
# ===========================
def recommend_same_category(idx, top_k=10):
    target_cat = df.loc[idx, "Loại sản phẩm"]

    # All products in the same category
    same_cat_df = df[df["Loại sản phẩm"] == target_cat]

    same_cat_indices = same_cat_df.index.tolist()

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Filter same category + exclude itself
    filtered = [(i, sims[i]) for i in same_cat_indices if i != idx]

    # Sort by similarity
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)

    # Get top-k indices
    top_indices = [i for i, s in filtered_sorted[:top_k]]

    return top_indices


# ===========================
# 4. STREAMLIT UI
# ===========================
st.set_page_config(page_title="CBF Product Recommendation", layout="wide")

st.title("Content-Based Recommendation System")
st.write("Search products and get recommendations **within the same category only**.")

query = st.text_input("Enter product name or keywords:")

if query:

    # STEP 1 — Find closest match
    idx, score = find_best_match(query)

    st.subheader("Best Matched Product")
    st.write(df.loc[idx, ["Tên sản phẩm", "Loại sản phẩm", "Giá", "Thương hiệu"]])

    # STEP 2 — Recommend same category only
    st.subheader("Recommended Products (Same Category)")
    rec_indices = recommend_same_category(idx, top_k=10)

    if not rec_indices:
        st.warning("No similar products found in this category.")
    else:
        st.dataframe(df.loc[rec_indices, [
            "Tên sản phẩm",
            "Loại sản phẩm",
            "Giá",
            "Thương hiệu",
            "Mô tả"
        ]])
