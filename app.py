import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# Helper functions (from Colab pipeline)
# ---------------------------------------------------------------------
import underthesea
from nltk.corpus import stopwords

# Load Vietnamese + English stopwords
vi_stopwords = set([
    'và','là','của','này','khi','trên','có','cho','được','như','một','đi'
])
en_stopwords = set(stopwords.words('english'))

all_stopwords = vi_stopwords.union(en_stopwords)

def tokenize_and_filter(text):
    tokens = underthesea.word_tokenize(text)
    return [t for t in tokens if t.lower() not in all_stopwords]

def add_bigrams(tokens):
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

# ---------------------------------------------------------------------
# 1) Load & preprocess data
# ---------------------------------------------------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    df = pd.read_csv(csv_path)
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str).str.strip()
    df["Thương hiệu"] = df["Thương hiệu"].fillna("").astype(str).str.strip()
    
    # Combined text for TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"]
    )
    
    return df

df = load_data()

# ---------------------------------------------------------------------
# 2) Build TF-IDF matrix (preprocessed tokens + bigrams)
# ---------------------------------------------------------------------
@st.cache_data
def build_tfidf(df):
    processed_texts = []
    for text in df["FullText"]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = tokenize_and_filter(text)
        tokens_bigram = add_bigrams(tokens)
        processed_texts.append(" ".join(tokens_bigram))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_tfidf(df)

# ---------------------------------------------------------------------
# 3) Recommendation function
# ---------------------------------------------------------------------
def recommend_product(query, top_k=5, threshold=0.1):
    # Preprocess query
    query_text = query.lower()
    query_text = re.sub(r'[^\w\s]', ' ', query_text)
    tokens = tokenize_and_filter(query_text)
    tokens_bigram = add_bigrams(tokens)
    query_vec = vectorizer.transform([" ".join(tokens_bigram)])
    
    # Cosine similarity with all products
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Step 1: Top-1 product
    best_idx = scores.argmax()
    if scores[best_idx] < threshold:
        return None, []
    
    target_category = df.loc[best_idx, "Loại sản phẩm"] if "Loại sản phẩm" in df.columns else None
    
    # Step 2: Filter same category & exclude main product
    if target_category:
        mask = (df["Loại sản phẩm"] == target_category)
        mask.iloc[best_idx] = False
        filtered_scores = scores[mask]
        filtered_indices = df.index[mask]
    else:
        filtered_scores = np.delete(scores, best_idx)
        filtered_indices = np.delete(np.arange(len(df)), best_idx)
    
    # Step 3: Top-K recommendations
    top_indices = filtered_scores.argsort()[::-1][:top_k]
    recommendations = []
    for i in top_indices:
        idx = filtered_indices[i]
        recommendations.append({
            "Name": df.loc[idx, "Tên sản phẩm"],
            "Description": df.loc[idx, "Mô tả"],
            "Brand": df.loc[idx, "Thương hiệu"],
            "Price": df.loc[idx, "Giá"],
            "Rating": df.loc[idx, "Điểm đánh giá"],
            "Similarity": filtered_scores[i]
        })
    
    return best_idx, recommendations

# ---------------------------------------------------------------------
# 4) Streamlit UI
# ---------------------------------------------------------------------
st.title("CBF Product Recommendation Demo")
st.markdown("Tìm sản phẩm và xem các gợi ý tương đồng giống như Colab!")

top_k = st.sidebar.number_input("Top-K recommendations:", 1, 20, 5)
threshold = st.sidebar.slider("Minimum similarity threshold:", 0.0, 1.0, 0.1, 0.01)

query = st.text_input("Nhập tên sản phẩm hoặc mô tả:")

if query:
    best_idx, recommendations = recommend_product(query, top_k=top_k, threshold=threshold)
    
    if best_idx is None:
        st.warning("Không tìm thấy sản phẩm phù hợp với từ khóa.")
    else:
        st.subheader("Sản phẩm chính:")
        st.markdown(f"**{df.loc[best_idx, 'Tên sản phẩm']}**")
        st.write(f"Mô tả: {df.loc[best_idx, 'Mô tả']}")
        st.write(f"Thương hiệu: {df.loc[best_idx, 'Thương hiệu']}")
        st.write(f"Giá: {df.loc[best_idx, 'Giá']}")
        st.write(f"Rating: {df.loc[best_idx, 'Điểm đánh giá']}")
        
        st.subheader("Bạn cũng có thể thích:")
        for rec in recommendations:
            st.markdown(f"**{rec['Name']}**")
            st.write(f"Mô tả: {rec['Description'][:150]}...")
            st.write(f"Thương hiệu: {rec['Brand']}")
            st.write(f"Giá: {rec['Price']} | Rating: {rec['Rating']}")
            st.info(f"Tương đồng: {rec['Similarity']:.3f}")
