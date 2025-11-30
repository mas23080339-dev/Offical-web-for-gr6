import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer

# -------------------------------
# 1. TEXT PREPROCESSING PIPELINE
# -------------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Lowercasing
    text = re.sub(r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
                  r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
                  r"ýỳỷỹỵđ\s]", " ", text)  # Remove punctuation
    text = " ".join(ViTokenizer.tokenize(text).split())  # Tokenization
    return text

# -------------------------------
# 2. LOAD DATA + BUILD TF-IDF
# -------------------------------
@st.cache_data
def load_data(file_path="Gr6.csv"):
    df = pd.read_csv(file_path)
    df["Từ khóa"] = df["Từ khóa"].astype(str)
    df["Link ảnh"] = df["Link ảnh"].fillna("")
    
    # Preprocess keywords for similarity
    df["FullText"] = df["Từ khóa"].apply(preprocess_text)
    
    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["FullText"])
    
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data()

# -------------------------------
# 3. STREAMLIT UI
# -------------------------------
st.title("Demo CBF for Small Business")
st.subheader("Introduction")
st.write("This system recommends products based on your input description, keywords, or product name about Adidas, Lacoste, Gucci, Nike, and Puma products.")

user_query = st.text_input("Enter the product name or description:")

threshold = 0.01  # Lower threshold to avoid missing products

if user_query:
    # Preprocess user input
    user_query_processed = preprocess_text(user_query)
    
    # TF-IDF vector
    user_vec = vectorizer.transform([user_query_processed])
    
    # Cosine similarity
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]
    
    # Add similarity to dataframe
    df["similarity"] = similarities
    
    # Sort by similarity
    result = df.sort_values(by="similarity", ascending=False)
    
    # Filter by threshold
    filtered = result[result["similarity"] >= threshold]
    
    # -------------------------------
    # 4. DISPLAY RESULTS
    # -------------------------------
    if filtered.empty:
        st.warning("No matching products found. Try another description.")
    else:
        st.subheader("Best Match:")
        best = filtered.iloc[0]
        st.write(f"**{best['Tên sản phẩm']}** — similarity: {best['similarity']:.4f}")
        if best["Link ảnh"] != "":
            st.image(best["Link ảnh"], width=250)
        
        # Other suggestions
        st.subheader("Other Suggestions (Top 30):")
        for i, row in filtered.iloc[1:31].iterrows():
            st.write(f"- {row['Tên sản phẩm']} — `{row['similarity']:.4f}`")
            if row["Link ảnh"] != "":
                st.image(row["Link ảnh"], width=180)
    
    # Debug: show top 10 products with similarity
    st.subheader("Debug - Top 10 Similarities:")
    st.dataframe(result[["Tên sản phẩm", "similarity"]].head(10))
