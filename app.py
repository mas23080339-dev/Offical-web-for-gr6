import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- NLTK setup -------------------
import nltk
try:
    from nltk.corpus import stopwords
    en_stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    en_stopwords = set(stopwords.words('english'))

# ------------------- Underthesea setup -------------------
try:
    from underthesea import word_tokenize
    viet_tokenizer = True
except ImportError:
    viet_tokenizer = False
    st.warning("underthesea is not installed. Vietnamese tokenization disabled.")

# ------------------- Streamlit config -------------------
st.set_page_config(page_title="Demo CBF Hybrid", layout="wide")

# ------------------- 1) Load data -------------------
@st.cache_data
def load_data(csv="Gr6.csv"):
    try:
        df = pd.read_csv(csv)
    except:
        st.error("Cannot load dataset")
        return pd.DataFrame()

    # Ensure required fields exist
    required_cols = [
        "Tên sản phẩm", "Mô tả", "Từ khóa", "Thương hiệu", 
        "Loại sản phẩm", "Giá", "Điểm đánh giá"
    ]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()

    # Clean text
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str)
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Thương hiệu"] = df["Thương hiệu"].fillna("").astype(str)

    # Combine fields
    df["FullText"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"]
    )

    if "Link ảnh" not in df.columns:
        df["Link ảnh"] = ""

    return df

df = load_data()
if df.empty:
    st.stop()

# ------------------- 2) Text processing -------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()

    tokens = [t for t in tokens if t not in en_stopwords]

    if viet_tokenizer:
        tokens = word_tokenize(" ".join(tokens), format="text").split()

    return tokens

def add_bigrams(tokens):
    return tokens + [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]

df["tokens"] = df["FullText"].apply(preprocess_text)
df["tokens_bigram"] = df["tokens"].apply(add_bigrams)
df["processed_text"] = df["tokens_bigram"].apply(lambda x: " ".join(x))

# ------------------- 3) TF-IDF -------------------
@st.cache_data
def build_tfidf(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_tfidf(df["processed_text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# ------------------- 4) Recommendation logic -------------------
def find_best_match(query):
    q = query.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    tokens = preprocess_text(q)
    tokens = add_bigrams(tokens)
    q_final = " ".join(tokens)

    q_vec = vectorizer.transform([q_final])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    idx = scores.argmax()
    return idx, scores[idx]


def recommend(idx, top_k=5, threshold=0.1):
    target_cat = df.loc[idx, "Loại sản phẩm"]
    scores = similarity_matrix[idx]

    same_cat_idx = df[df["Loại sản phẩm"] == target_cat].index.tolist()
    same_cat_idx = [i for i in same_cat_idx if i != idx]

    valid = [(i, scores[i]) for i in same_cat_idx if scores[i] >= threshold]
    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)[:top_k]

    result = [
        {
            "index": i,
            "similarity": sim,
            "row": df.loc[i]
        }
        for i, sim in valid_sorted
    ]
    return result

# ------------------- Streamlit UI -------------------
st.title("Content-Based Recommendation System Demo")
st.write("Search by keyword or choose a product to get recommendations.")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K recommendations", 1, 20, 5)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.1, 0.05)

mode = st.radio(
    "Mode:",
    ["Search Mode (keyword)", "Evaluation Mode (select product)"],
    horizontal=True
)

best_idx = None
best_score = 0

# ------------------- Search Mode -------------------
if mode == "Search Mode (keyword)":
    query = st.text_input("Enter product or keyword:")
    if query:
        best_idx, best_score = find_best_match(query)
        if best_score < threshold:
            st.warning("No product matches your query.")
            best_idx = None
        else:
            st.info(f"Best match: **{df.loc[best_idx,'Tên sản phẩm']}** (Similarity: {best_score:.3f})")

# ------------------- Evaluation Mode -------------------
else:
    product = st.selectbox("Select product:", df["Tên sản phẩm"].unique())
    best_idx = df[df["Tên sản phẩm"] == product].index[0]
    best_score = 1.0

# ------------------- Show main product -------------------
if best_idx is not None:
    st.subheader(f"Main Product: {df.loc[best_idx, 'Tên sản phẩm']}")

    col1, col2 = st.columns([1, 3])
    with col1:
        img = df.loc[best_idx, "Link ảnh"]
        if img.strip():
            st.image(img, width=220)
        else:
            st.info("No image")

    with col2:
        row = df.loc[best_idx]
        st.markdown(f"**Description:** {row['Mô tả']}")
        st.markdown(f"**Brand:** {row['Thương hiệu']}")
        st.markdown(f"**Category:** {row['Loại sản phẩm']}")
        st.markdown(f"**Price:** {row['Giá']} | **Rating:** {row['Điểm đánh giá']}")
        st.success(f"Similarity score: {best_score:.3f}")

# ------------------- Show recommendations -------------------
st.subheader("You may also like:")

if best_idx is not None:
    recs = recommend(best_idx, top_k, threshold)

    if not recs:
        st.warning("No similar products above threshold.")
    else:
        cols = st.columns(min(len(recs), 4))
        for i, rec in enumerate(recs):
            r = rec["row"]
            with cols[i % len(cols)]:
                if r["Link ảnh"].strip():
                    st.image(r["Link ảnh"], width=140)
                else:
                    st.markdown(
                        "<div style='width:140px; height:120px; background:#444; "
                        "color:white; display:flex; align-items:center; "
                        "justify-content:center; border-radius:6px;'>No image</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(f"**{r['Tên sản phẩm']}**")
                st.caption(r["Mô tả"][:100] + "...")
                st.caption(f"Brand: {r['Thương hiệu']}")
                st.caption(f"Price: {r['Giá']}")
                st.info(f"Similarity: {rec['similarity']:.3f}")
