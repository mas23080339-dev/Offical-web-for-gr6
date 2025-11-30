import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1) Load & xử lý dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("Gr6.csv")

    # Chuẩn hóa cột Từ khóa 
    df["Từ khóa"] = df["Từ khóa"].fillna("").str.replace(";", " ")

    # Gộp các cột để TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("")
    )

    # Chuẩn hóa cột Link ảnh
    if "Link ảnh" in df.columns:
        df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()


# 2) TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["FullText"])


# 3) Giao diện Streamlit
st.set_page_config(
    page_title="Demo CBF in small business",
    layout="wide"
)

st.title("Welcome to our store, we have many product from Adidas, Nike, Lacoste, Puma, Gucci")
st.write("Our system will find product rely on your input (description/keywords).")

user_query = st.text_input("Please enter your product:")

if user_query:
    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranking = scores.argsort()[::-1]

    threshold = 0.1

    if scores[ranking[0]] < threshold:
        st.warning("Can't find any product, try again please.")
    else:
        best_idx = ranking[0]

        st.subheader("Our product:")

        # Hiển thị ảnh
        if "Link ảnh" in df.columns and df.loc[best_idx, "Link ảnh"]:
            st.image(df.loc[best_idx, "Link ảnh"], width=250)

        st.write(f"**Tên:** {df.loc[best_idx, 'Tên sản phẩm']}")
        st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
        st.write(f"**Giá:** {df.loc[best_idx, 'Giá']}")
        st.write(f"**Thương hiệu:** {df.loc[best_idx, 'Thương hiệu']}")
        st.write(f"Điểm đánh giá: {df.loc[best_idx, 'Điểm đánh giá']}")
        st.write(f"**Similarity:** `{scores[best_idx]:.3f}`")

        st.subheader("Maybe you also like this product:")

        for idx in ranking[1:6]:
            if scores[idx] < threshold:
                break

            # Hiển thị ảnh gợi ý
            if "Link ảnh" in df.columns and df.loc[idx, "Link ảnh"]:
                st.image(df.loc[idx, "Link ảnh"], width=180)

            st.write(f"**Tên:** {df.loc[idx, 'Tên sản phẩm']}")
            st.write(f"**Mô tả:** {df.loc[idx, 'Mô tả']}") 
            st.write(f"Giá: {df.loc[idx, 'Giá']}")
            st.write(f"**Thương hiệu:** {df.loc[idx, 'Thương hiệu']}")
            st.write(f"Điểm đánh giá: {df.loc[idx, 'Điểm đánh giá']}")
            st.write(f"Similarity: `{scores[idx]:.3f}`")
            st.write("---")
