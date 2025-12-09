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
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    en_stopwords = set(stopwords.words('english'))

# ------------------- Underthesea setup -------------------
try:
    from underthesea import word_tokenize
    viet_tokenizer = True
except ImportError:
    viet_tokenizer = False
    st.warning("underthesea not installed. Vietnamese tokenization will be skipped.")

# ------------------- Streamlit config -------------------
st.set_page_config(page_title="Demo CBF Hybrid", layout="wide")

# ------------------- 1) Load data -------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
        return pd.DataFrame()

    # Fill missing and clean
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str).str.strip()
    df["Thương hiệu"] = df["Thương hiệu"].fillna("").astype(str)

    # NEW: Ensure category exists
    if "Loại sản phẩm" not in df.columns:
        st.error("Dataset missing column: 'Loại sản phẩm' (Category).")
        return pd.DataFrame()

    # Combine text fields
    df["FullText"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"]
    )

    if "Link ảnh" in df.columns:
        df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()
if df.empty:
    st.stop()

# ------------------- 2) Text processing -------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()

    # Remove English stopwords
    tokens = [t for t in tokens if t not in en_stopwords]

    # Vietnamese tokenization
    if viet_tokenizer:
        text_vn = " ".join(tokens)
        tokens = word_tokenize(text_vn, format="text").split()

    return tokens

def add_bigrams(tokens):
    bigrams = [tokens[i]+"_"+tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

df["tokens"] = df["FullText"].apply(preprocess_text)
df["tokens_with_bigrams"] = df["tokens"].apply(add_bigrams)
df["processed_text"] = df["tokens_with_bigrams"].apply(lambda x: " ".join(x))

# ------------------- 3) TF-IDF -------------------
@st.cache_data
def build_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_tfidf_matrix(df["processed_text"])
item_similarity_matrix = cosine_similarity(tfidf_matrix)

# ------------------- 4) Recommendation function -------------------
def recommend_product_by_index(idx, top_k=5, threshold=0.1):

    # NEW: filter by category
    target_category = df.loc[idx, "Loại sản phẩm"]

    scores = item_similarity_matrix[idx]
    sorted_idx = scores.argsort()[::-1]

    recommendations = []
    count = 0

    for i in sorted_idx[1:]:  # skip itself
        if scores[i] < threshold:
            continue

        # NEW: only recommend same category
        if df.loc[i, "Loại sản phẩm"] != target_category:
            continue

        recommendations.append({
            "index": i,
            "similarity": scores[i],
            "data": df.loc[i]
        })

        count += 1
        if count >= top_k:
            break

    return recommendations

def find_best_match(query_text):
    tokens = add_bigrams(preprocess_text(query_text))
    text_final = " ".join(tokens)
    vec = vectorizer.transform([text_final])
    scores = cosine_similarity(vec, tfidf_matrix)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    return best_idx, best_score

# ------------------- 5) Streamlit UI -------------------
st.title("Demo Content-Based Product Recommendation")
st.markdown("Search by keyword or choose a product to get recommendations!")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Number of recommendations (Top K)", 1, 20, 5)
    threshold = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.1, 0.05)

mode = st.radio(
    "Choose mode:",
    ("Search Mode (keyword)", "Evaluation Mode (select product)"),
    horizontal=True
)

best_idx = None
best_score = 0.0
is_eval_mode = False

# ------------------- SEARCH MODE -------------------
if mode == "Search Mode (keyword)":
    query_text = st.text_input("Enter product name or description:")
    if query_text:
        best_idx, best_score = find_best_match(query_text)
        if best_score < threshold:
            st.warning("No product found matching the query.")
            best_idx = None
        else:
            st.info(f"Best match: {df.loc[best_idx, 'Tên sản phẩm']} (Similarity: {best_score:.3f})")

# ------------------- EVALUATION MODE -------------------
elif mode == "Evaluation Mode (select product)":
    selected_product = st.selectbox("Select product:", df["Tên sản phẩm"].unique())
    if selected_product:
        is_eval_mode = True
        best_idx = df[df["Tên sản phẩm"] == selected_product].index[0]
        best_score = 1.0

# ------------------- SHOW MAIN PRODUCT -------------------
if best_idx is not None:
    st.subheader(f"Main Product: {df.loc[best_idx, 'Tên sản phẩm']}")

    col_img, col_info = st.columns([1, 3])
    with col_img:
        image_url = df.loc[best_idx, "Link ảnh"] if "Link ảnh" in df.columns else None
        if image_url and image_url.strip():
            st.image(image_url, width=200)
        else:
            st.info("No image available.")

    with col_info:
        st.markdown(f"**Name:** {df.loc[best_idx, 'Tên sản phẩm']}")
        st.markdown(f"**Description:** {df.loc[best_idx, 'Mô tả']}")
        st.markdown(f"**Brand:** {df.loc[best_idx, 'Thương hiệu']}")
        st.markdown(f"**Category:** {df.loc[best_idx, 'Loại sản phẩm']}")
        st.markdown(f"**Price:** {df.loc[best_idx, 'Giá']} | **Rating:** {df.loc[best_idx, 'Điểm đánh giá']}")

        if is_eval_mode:
            st.success("Evaluation mode: Recommendations are based on this product.")
        else:
            st.success(f"Similarity to query: {best_score:.3f}")

# ------------------- SHOW RECOMMENDATIONS -------------------
st.subheader("You may also like:")
recs = recommend_product_by_index(best_idx, top_k, threshold)

if recs:
    rec_cols = st.columns(min(len(recs), 4))
    for i, rec in enumerate(recs):
        idx = rec["index"]
        with rec_cols[i % len(rec_cols)]:
            img_url = df.loc[idx, "Link ảnh"] if "Link ảnh" in df.columns else None
            if img_url and img_url.strip():
                st.image(img_url, width=120)
            else:
                st.markdown(
                    "<div style='height:120px; background-color:#333; color:white; padding:10px; border-radius:5px; display:flex; align-items:center; justify-content:center; font-size:12px;'>Image missing</div>",
                    unsafe_allow_html=True
                )

            st.markdown(f"**{df.loc[idx, 'Tên sản phẩm']}**")
            st.caption(f"{df.loc[idx, 'Mô tả'][:100]}...")
            st.caption(f"Brand: {df.loc[idx, 'Thương hiệu']}")
            st.caption(f"Category: {df.loc[idx, 'Loại sản phẩm']}")
            st.caption(f"Price: {df.loc[idx, 'Giá']}")
            st.info(f"Similarity: {rec['similarity']:.3f}")
else:
    st.warning("No similar products found above the threshold.")
