import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================================================
# CONFIG
# ==========================================================================================
st.set_page_config(page_title="Demo CBF Recommendation system", layout="wide")

# ==========================================================================================
# LOAD DATA
# ==========================================================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Gr6.csv")

    # Chuẩn hóa text tránh lỗi TF-IDF
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str)
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Loại sản phẩm"] = df["Loại sản phẩm"].fillna("").astype(str)

    # Cột text final để TF-IDF
    df["text_clean"] = df["Tên sản phẩm"] + " " + df["Mô tả"]
    return df

df = load_data()

# ==========================================================================================
# TF-IDF MODEL
# ==========================================================================================
@st.cache_resource
def build_tfidf_model(texts):
    vect = TfidfVectorizer(stop_words="english")
    mat = vect.fit_transform(texts)
    return vect, mat

vectorizer, tfidf_matrix = build_tfidf_model(df["text_clean"])

# ==========================================================================================
# UTILS — QUERY PROCESSING
# ==========================================================================================
def process_query(q: str):
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

# ==========================================================================================
# STEP 1 — TÌM SẢN PHẨM GẦN NHẤT VỚI QUERY
# ==========================================================================================
def search_best_match(query):
    processed = process_query(query)
    q_vec = vectorizer.transform([processed])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    best_idx = sims.argmax()
    best_score = sims[best_idx]
    return best_idx, best_score, sims

# ==========================================================================================
# STEP 2 — LỌC THEO LOẠI SẢN PHẨM
# ==========================================================================================
def filter_same_category(idx, sims, top_k=10, threshold=0.15):
    target_cat = df.loc[idx, "Loại sản phẩm"]

    df["sim"] = sims
    df_sorted = df[df["sim"] >= threshold].sort_values("sim", ascending=False)

    # LỌC CÙNG CATEGORY
    same_cat = df_sorted[df_sorted["Loại sản phẩm"] == target_cat]

    # Nếu đủ top K → dùng luôn
    if len(same_cat) >= top_k + 1:
        return same_cat.iloc[1:top_k+1]  # bỏ sản phẩm chính

    # Nếu không đủ → fallback: lấy thêm sản phẩm khác loại
    fallback = df_sorted.iloc[1:top_k+1]

    return fallback

# ==========================================================================================
# STREAMLIT UI
# ==========================================================================================
st.title("Content-Based Filtering Recommendation Demo")

query = st.text_input("Enter the product you want to search for:")

top_k = st.slider("Top K", 5, 20, 10)
threshold = st.slider("Cosine similarity threshold", 0.05, 0.50, 0.15)

# ==========================================================================================
# PROCESS
# ==========================================================================================
if query.strip() != "":
    best_idx, best_score, sims = search_best_match(query)

    st.subheader("🔎 Most similar product in store:")
    st.write(f"**Tên sản phẩm:** {df.loc[best_idx, 'Tên sản phẩm']}")
    st.write(f"**Loại sản phẩm:** {df.loc[best_idx, 'Loại sản phẩm']}")
    st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
    st.write(f"**Similarity:** {best_score:.4f}")

    st.divider()

    # ======================================================================================
    # GET RECOMMENDATIONS
    # ======================================================================================
    rec_df = filter_same_category(best_idx, sims, top_k, threshold)

    st.subheader("Recommended products")

    for i, row in rec_df.iterrows():
        with st.container(border=True):
            st.write(f"### {row['Tên sản phẩm']}")
            st.write(f"**Loại:** {row['Loại sản phẩm']}")
            st.write(f"**Score:** {row['sim']:.4f}")
            st.write(row["Mô tả"])
