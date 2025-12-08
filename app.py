import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ------------------------------------------
# 1) Cấu hình Streamlit
# ------------------------------------------
st.set_page_config(
    page_title="Demo CBF Hybrid",
    layout="wide"
)
st.title("Chào mừng đến với cửa hàng của chúng tôi!")
st.markdown("Hãy tìm kiếm sản phẩm hoặc chọn sản phẩm chính xác để nhận gợi ý!")

# ------------------------------------------
# 2) Load dữ liệu
# ------------------------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    """Tải dữ liệu và tiền xử lý cơ bản."""
    df = pd.read_csv(csv_path)

    # Làm sạch cột
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str).str.strip()

    # Gộp tất cả trường text để TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"] + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("").astype(str)
    )

    # Chuẩn hóa link ảnh nếu cột tồn tại
    if "Link ảnh" in df.columns:
        df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()
if df.empty:
    st.error("Dữ liệu rỗng hoặc lỗi tải file CSV. Kiểm tra lại đường dẫn.")
    st.stop()

# ------------------------------------------
# 3) Xây dựng TF-IDF vectorizer
# ------------------------------------------
@st.cache_data
def build_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["FullText"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer(df)

# ------------------------------------------
# 4) Hàm recommend giống Colab
# ------------------------------------------
def get_recommendations_by_query(query_vec, df, vectorizer, top_k=5, threshold=0.1):
    """
    Gợi ý sản phẩm dựa trên TF-IDF và Cosine Similarity giống Colab.
    """
    tfidf_all = vectorizer.transform(df["FullText"])
    scores = cosine_similarity(query_vec, tfidf_all)[0]
    ranking = scores.argsort()[::-1]

    recommendations = []
    count = 0
    for idx in ranking:
        score = scores[idx]
        if score < threshold or count >= top_k:
            break
        recommendations.append({
            "index": idx,
            "similarity": score,
            "data": df.loc[idx]
        })
        count += 1

    return recommendations

# ------------------------------------------
# 5) Sidebar: Thiết lập Top-K và Threshold
# ------------------------------------------
with st.sidebar:
    st.header("Thiết lập gợi ý")
    top_k = st.number_input("Số lượng gợi ý (Top K):", min_value=1, max_value=20, value=5)
    threshold = st.slider("Ngưỡng tương đồng tối thiểu:", 0.0, 1.0, 0.1, 0.05)

# ------------------------------------------
# 6) Chọn chế độ hoạt động
# ------------------------------------------
mode = st.radio(
    "Chọn chế độ hoạt động:",
    ("1. Tìm kiếm bằng từ khóa", "2. Chọn sản phẩm chính xác (Evaluation Mode)"),
    horizontal=True
)

best_idx = None
best_score = 0.0
is_accurate_mode = False

st.markdown("---")

# ------------------------------------------
# 7) Search Mode
# ------------------------------------------
if mode == "1. Tìm kiếm bằng từ khóa":
    user_query = st.text_input(
        "Nhập từ khóa hoặc mô tả sản phẩm:",
        key="query_input"
    )
    if user_query:
        query_vec = vectorizer.transform([user_query])
        recommendations = get_recommendations_by_query(
            query_vec, df, vectorizer, top_k=top_k, threshold=threshold
        )
        if len(recommendations) == 0:
            st.warning("Không tìm thấy sản phẩm nào đủ tương đồng.")
            best_idx = None
        else:
            best_idx = recommendations[0]["index"]
            best_score = recommendations[0]["similarity"]
            st.info(f"Đã tìm thấy sản phẩm chính: {df.loc[best_idx, 'Tên sản phẩm']} (Score: {best_score:.3f})")

# ------------------------------------------
# 8) Evaluation Mode (chọn sản phẩm)
# ------------------------------------------
elif mode == "2. Chọn sản phẩm chính xác (Evaluation Mode)":
    product_options = df["Tên sản phẩm"].unique()
    selected_product_name = st.selectbox(
        "Chọn sản phẩm chính xác:",
        options=product_options
    )
    if selected_product_name:
        is_accurate_mode = True
        best_idx = df[df["Tên sản phẩm"] == selected_product_name].index[0]
        best_score = 1.0  # tự mình
        # Lấy vector TF-IDF của sản phẩm
        product_vec = vectorizer.transform([df.loc[best_idx, "FullText"]])
        # Recommend Top-K bỏ chính nó
        recommendations = get_recommendations_by_query(
            product_vec, df, vectorizer, top_k=top_k+1, threshold=threshold
        )[1:]

# ------------------------------------------
# 9) Hiển thị sản phẩm chính và gợi ý
# ------------------------------------------
if best_idx is not None:
    st.subheader(f"Sản phẩm chính: {df.loc[best_idx, 'Tên sản phẩm']}")

    col_img, col_info = st.columns([1, 3])
    with col_img:
        image_url = df.loc[best_idx]["Link ảnh"] if "Link ảnh" in df.columns else None
        if image_url and image_url.strip():
            st.image(image_url, width=200)
        else:
            st.info("Không có hình ảnh.")

    with col_info:
        st.markdown(f"**Tên:** `{df.loc[best_idx, 'Tên sản phẩm']}`")
        st.markdown(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
        st.markdown(f"**Thương hiệu:** `{df.loc[best_idx, 'Thương hiệu']}`")
        st.markdown(f"**Giá:** `{df.loc[best_idx, 'Giá']}` | **Đánh giá:** `{df.loc[best_idx, 'Điểm đánh giá']}`")
        if is_accurate_mode:
            st.success("Chế độ Evaluation: Gợi ý dựa trên sản phẩm này.")
        else:
            st.success(f"Độ tương đồng với Query: `{best_score:.3f}`")

    st.markdown("---")
    st.subheader("Bạn có thể thích sản phẩm này:")

    if recommendations:
        rec_cols = st.columns(min(len(recommendations), 4))
        for i, rec in enumerate(recommendations):
            idx = rec["index"]
            with rec_cols[i % len(rec_cols)]:
                image_url = df.loc[idx]["Link ảnh"] if "Link ảnh" in df.columns else None
                if image_url and image_url.strip():
                    st.image(image_url, width=120)
                else:
                    st.markdown(
                        "<div style='height:120px; background-color:#333; color:white; padding:10px; "
                        "border-radius: 5px; display:flex; align-items:center; justify-content:center; "
                        "text-align:center; font-size:12px;'>Ảnh đang cập nhật</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(f"**{df.loc[idx, 'Tên sản phẩm']}**")
                st.caption(f"{df.loc[idx, 'Mô tả'][:100]}...")
                st.caption(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
                st.caption(f"Giá: {df.loc[idx, 'Giá']}")
                st.info(f"Tương đồng: `{rec['similarity']:.3f}`")
    else:
        st.warning(f"Không tìm thấy sản phẩm tương tự nào với ngưỡng {threshold:.2f}.")
