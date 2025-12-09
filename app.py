import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# 1) STREAMLIT CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Demo CBF Hybrid", layout="wide")


# ------------------------------------------------------
# 2) LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    df = pd.read_csv(csv_path)

    # Fill missing
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str).str.strip()
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Thương hiệu"] = df["Thương hiệu"].fillna("").astype(str)
    df["Loại sản phẩm"] = df["Loại sản phẩm"].fillna("").astype(str)

    if "Link ảnh" not in df.columns:
        df["Link ảnh"] = ""

    # Combine text
    df["FullText"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"]
    ).str.lower()

    return df


df = load_data()
if df.empty:
    st.error("Dataset empty.")
    st.stop()


# ------------------------------------------------------
# 3) TEXT PROCESSING (very stable)
# ------------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


def add_bigrams(tokens):
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams


df["tokens"] = df["FullText"].apply(preprocess_text)
df["tokens_bigram"] = df["tokens"].apply(add_bigrams)
df["processed_text"] = df["tokens_bigram"].apply(lambda x: " ".join(x))


# ------------------------------------------------------
# 4) TF-IDF MATRIX
# ------------------------------------------------------
@st.cache_data
def build_tfidf(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


vectorizer, tfidf_matrix = build_tfidf(df["processed_text"])
item_sim = cosine_similarity(tfidf_matrix)


# ------------------------------------------------------
# 5) MAIN FUNCTIONS
# ------------------------------------------------------
def find_best_match(query):
    tokens = preprocess_text(query)
    tokens = add_bigrams(tokens)
    txt = " ".join(tokens)
    vec = vectorizer.transform([txt])
    scores = cosine_similarity(vec, tfidf_matrix)[0]
    idx = scores.argmax()
    return idx, scores[idx]


def recommend(idx, topk=5, threshold=0.1):
    target_cat = df.loc[idx, "Loại sản phẩm"]
    scores = item_sim[idx]
    sorted_idx = scores.argsort()[::-1]

    recs = []
    for i in sorted_idx[1:]:
        if scores[i] < threshold:
            continue
        if df.loc[i, "Loại sản phẩm"] != target_cat:
            continue
        recs.append((i, scores[i]))
        if len(recs) >= topk:
            break
    return recs


# ------------------------------------------------------
# 6) STREAMLIT UI
# ------------------------------------------------------
st.title("Demo Content-Based Product Recommendation")

with st.sidebar:
    st.header("Settings")
    topk = st.number_input("Top-K", 1, 20, 5)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.1, 0.05)

mode = st.radio(
    "Choose mode:",
    ["Search Mode (keyword)", "Evaluation Mode (select product)"],
    horizontal=True
)

best_idx = None
best_score = 0.0
eval_mode = False


# ------------------------------------------------------
# SEARCH MODE
# ------------------------------------------------------
if mode == "Search Mode (keyword)":
    q = st.text_input("Enter product keyword:")
    if q.strip():
        best_idx, best_score = find_best_match(q)
        if best_score < threshold:
            st.warning("No matching product.")
            best_idx = None
        else:
            st.info(f"Matched: {df.loc[best_idx,'Tên sản phẩm']} ({best_score:.3f})")


# ------------------------------------------------------
# EVALUATION MODE
# ------------------------------------------------------
else:
    name = st.selectbox("Choose product:", df["Tên sản phẩm"])
    if name:
        eval_mode = True
        best_idx = df[df["Tên sản phẩm"] == name].index[0]
        best_score = 1.0


# ------------------------------------------------------
# SHOW MAIN PRODUCT
# ------------------------------------------------------
if best_idx is not None:
    st.subheader("Main Product")
    col1, col2 = st.columns([1, 3])

    with col1:
        img = df.loc[best_idx, "Link ảnh"]
        if img.strip():
            st.image(img, width=220)
        else:
            st.info("No image")

    with col2:
        st.markdown(f"**Tên:** {df.loc[best_idx,'Tên sản phẩm']}")
        st.markdown(f"**Mô tả:** {df.loc[best_idx,'Mô tả']}")
        st.markdown(f"**Thương hiệu:** {df.loc[best_idx,'Thương hiệu']}")
        st.markdown(f"**Loại:** {df.loc[best_idx,'Loại sản phẩm']}")
        st.markdown(f"**Giá:** {df.loc[best_idx,'Giá']}")
        st.markdown(f"**Rating:** {df.loc[best_idx,'Điểm đánh giá']}")

        if eval_mode:
            st.success("Evaluation Mode")
        else:
            st.success(f"Similarity: {best_score:.3f}")


# ------------------------------------------------------
# SHOW RECOMMENDATIONS
# ------------------------------------------------------
st.subheader("You may also like:")

if best_idx is not None:
    recs = recommend(best_idx, topk, threshold)
    if not recs:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(4)
        for j, (idx, score) in enumerate(recs):
            with cols[j % 4]:
                im = df.loc[idx, "Link ảnh"]
                if im.strip():
                    st.image(im, width=140)
                st.markdown(f"**{df.loc[idx,'Tên sản phẩm']}**")
                st.caption(df.loc[idx, "Mô tả"][:80] + "…")
                st.caption(f"Brand: {df.loc[idx,'Thương hiệu']}")
                st.caption(f"Price: {df.loc[idx,'Giá']}")
                st.info(f"{score:.3f}")
