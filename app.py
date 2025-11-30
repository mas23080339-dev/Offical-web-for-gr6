import streamlit as st
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load data
# -------------------------
selling_data = pd.read_csv("Gr6.csv", encoding='utf-8-sig')

# Ground truth (related products)
ground_truth_df = pd.read_csv("try (2).csv", encoding='utf-8-sig')
ground_truth_dict = {row['product name']: row['related product'].split(';')
                     for _, row in ground_truth_df.iterrows()}

# Normalize product names for exact match
selling_data['name_clean'] = selling_data['Tên sản phẩm'].fillna("").str.strip().str.lower()

# -------------------------
# Build similarity matrix (cached)
# -------------------------
@st.cache_data
def build_similarity(df):
    df["combined_features"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"].fillna("") + " " +
        df["Thương hiệu"].fillna("")
    )
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity = build_similarity(selling_data)

# -------------------------
# Recommendation function
# -------------------------
def recommend_product(product_name, top_k=5):
    product_name_clean = product_name.strip().lower()
    
    if product_name_clean not in selling_data['name_clean'].values:
        return []
    
    idx = selling_data[selling_data['name_clean'] == product_name_clean].index[0]
    scores = similarity[idx]
    top_indices = scores.argsort()[::-1][1:top_k+1]  # skip itself
    return selling_data.loc[top_indices, 'Tên sản phẩm'].tolist()

# -------------------------
# Evaluation metrics
# -------------------------
def average_precision_at_k(recommended, ground_truth, k):
    if not ground_truth:
        return 0.0
    hits = 0
    sum_precisions = 0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(ground_truth)

def ndcg_at_k(recommended, ground_truth, k):
    if not ground_truth:
        return 0.0
    dcg = sum((1 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in ground_truth))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

# -------------------------
# Streamlit app UI
# -------------------------
st.title("Product Recommendation & Evaluation System")

user_input = st.text_input("Enter a product name:")

top_k = st.number_input("Number of recommendations (top K):", min_value=1, max_value=20, value=5)

if user_input:
    recommendations = recommend_product(user_input, top_k=top_k)
    true_related = [item.strip() for item in ground_truth_dict.get(user_input, []) if item.strip()]

    if recommendations:
        st.subheader("Recommended Products:")
        for rec in recommendations:
            st.write(f"- {rec}")

        # -------------------------
        # Evaluation (if ground truth exists)
        # -------------------------
        if true_related:
            hits = sum(1 for rec in recommendations if rec in true_related)
            hit_rate = int(hits > 0)
            precision = hits / top_k if top_k else 0
            recall = hits / len(true_related) if true_related else 0
            ap_k = average_precision_at_k(recommendations, true_related, top_k)
            ndcg_k = ndcg_at_k(recommendations, true_related, top_k)

            st.subheader("Evaluation Metrics:")
            st.write(f"Ground Truth: {true_related}")
            st.write(f"Hits: {hits}")
            st.write(f"Hit Rate@{top_k}: {hit_rate}")
            st.write(f"Precision@{top_k}: {precision:.2f}")
            st.write(f"Recall@{top_k}: {recall:.2f}")
            st.write(f"AP@{top_k}: {ap_k:.4f}")
            st.write(f"NDCG@{top_k}: {ndcg_k:.4f}")
        else:
            st.info("No ground truth available for this product.")
    else:
        st.warning("Product not found or no recommendations available.")
