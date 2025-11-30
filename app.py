import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Demo CBF for small business",
    layout="wide"
)

# --------------------------------------------------------------------------------------
# 1) Tải & Tiền xử lý Dữ liệu
# Tải dữ liệu, làm sạch và tạo trường FullText để tính TF-IDF
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu tại đường dẫn '{csv_path}'. Vui lòng kiểm tra lại.")
        return pd.DataFrame() # Trả về DataFrame rỗng

    # Làm sạch cột Từ khóa
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    
    # Chuẩn hóa cột Mô tả để tránh lỗi khi gộp
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)

    # Gộp tất cả các trường văn bản lại để tính TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("").astype(str) + " " +
        df["Mô tả"] + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("").astype(str)
    )

    # Chuẩn hóa link ảnh nếu cột tồn tại
    if "Link ảnh" in df.columns:
        df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()

# --------------------------------------------------------------------------------------
# 2) Tính toán TF-IDF và Ma trận Tương đồng giữa các Sản phẩm (Item-to-Item Similarity Matrix)
# Đây là ma trận cốt lõi để thực hiện gợi ý Item-to-Item
# --------------------------------------------------------------------------------------
@st.cache_data
def build_similarity_matrices(df):
    """Tính TF-IDF và ma trận tương đồng giữa các sản phẩm."""
    if df.empty:
        return None, None
        
    vectorizer = TfidfVectorizer()
    # Ma trận TF-IDF của TẤT CẢ các sản phẩm
    tfidf_matrix = vectorizer.fit_transform(df["FullText"])
    
    # Ma trận Tương đồng giữa các Sản phẩm (Item-to-Item Similarity Matrix)
    # item_similarity_matrix[i, j] là độ tương đồng giữa sản phẩm i và sản phẩm j
    item_similarity_matrix = cosine_similarity(tfidf_matrix) 
    
    return vectorizer, item_similarity_matrix

if not df.empty:
    vectorizer, item_similarity_matrix = build_similarity_matrices(df)
else:
    vectorizer, item_similarity_matrix = None, None
    
# --------------------------------------------------------------------------------------
# 3) Hàm Gợi ý Sản phẩm Tương tự
# --------------------------------------------------------------------------------------
def get_item_recommendations(product_index, top_k, threshold):
    """
    Tìm các sản phẩm tương tự dựa trên Item-to-Item Similarity Matrix.
    """
    if item_similarity_matrix is None:
        return []
        
    # Lấy hàng tương đồng của sản phẩm chính
    item_scores = item_similarity_matrix[product_index]
    
    # Sắp xếp chỉ mục theo điểm số giảm dần
    ranking = item_scores.argsort()[::-1]
    
    recommendations = []
    count = 0
    # Bỏ qua sản phẩm đầu tiên (chính nó) -> bắt đầu từ ranking[1:]
    for idx in ranking[1:]:
        score = item_scores[idx]
        
        # Dừng lại nếu điểm số dưới ngưỡng hoặc đã đủ K sản phẩm
        if score < threshold or count >= top_k:
            break
            
        # Thêm sản phẩm được gợi ý vào danh sách
        recommendations.append({
            "index": idx,
            "similarity": score,
            "data": df.loc[idx]
        })
        count += 1
        
    return recommendations


# --------------------------------------------------------------------------------------
# 4) Streamlit UI & Logic
# --------------------------------------------------------------------------------------
st.title("Chào mừng đến với cửa hàng của chúng tôi! 🛍️")
st.markdown("Sử dụng công cụ này để tìm kiếm sản phẩm dựa trên từ khóa và nhận gợi ý sản phẩm tương tự.")

if df.empty or vectorizer is None:
    st.stop() # Dừng nếu dữ liệu chưa sẵn sàng

# Input của người dùng
user_query = st.text_input("Vui lòng nhập mô tả sản phẩm hoặc từ khóa (ví dụ: Áo thun co giãn, màu xanh, tập luyện cường độ cao):")

col_k, col_t = st.columns(2)
with col_k:
    top_k = st.number_input("Số lượng gợi ý (Top K):", min_value=1, max_value=20, value=5)
with col_t:
    threshold = st.slider("Ngưỡng tương đồng tối thiểu:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)


if user_query:
    st.markdown("---")
    
    # --- A. TÌM SẢN PHẨM PHÙ HỢP NHẤT VỚI QUERY NGƯỜI DÙNG ---
    query_vec = vectorizer.transform([user_query])
    query_scores = cosine_similarity(query_vec, vectorizer.transform(df["FullText"]))[0]
    ranking_by_query = query_scores.argsort()[::-1]
    
    best_idx = ranking_by_query[0]
    best_score = query_scores[best_idx]

    if best_score < threshold:
        st.warning("Không tìm thấy sản phẩm nào đủ tương đồng với từ khóa của bạn. Vui lòng thử lại.")
    else:
        # --- B. HIỂN THỊ SẢN PHẨM CHÍNH ---
        st.subheader(f"Sản phẩm phù hợp nhất: {df.loc[best_idx, 'Tên sản phẩm']}")
        
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            image_url = df.loc[best_idx, "Link ảnh"] if "Link ảnh" in df.columns else None
            if image_url:
                st.image(image_url, width=200, caption=df.loc[best_idx, 'Tên sản phẩm'])
            else:
                st.info("Không có hình ảnh.")

        with col_info:
            st.markdown(f"**Tên:** `{df.loc[best_idx, 'Tên sản phẩm']}`")
            st.markdown(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
            st.write(f"**Thương hiệu:** `{df.loc[best_idx, 'Thương hiệu']}`")
            st.markdown(f"**Giá:** `{df.loc[best_idx, 'Giá']}` | **Đánh giá:** `{df.loc[best_idx, 'Điểm đánh giá']}`")
            st.success(f"**Độ tương đồng với Query:** `{best_score:.3f}`")
            
        st.markdown("---")
        
        # --- C. GỢI Ý SẢN PHẨM TƯƠNG TỰ (ITEM-TO-ITEM) ---
        st.subheader("Bạn cũng có thể thích (Gợi ý dựa trên Sản phẩm Chính):")

        recommendations = get_item_recommendations(best_idx, top_k, threshold)
        
        if recommendations:
            
            # Sử dụng st.columns để hiển thị gọn gàng hơn
            rec_cols = st.columns(min(top_k, 5)) # Tối đa 5 cột ngang

            for i, rec in enumerate(recommendations):
                idx = rec["index"]
                
                with rec_cols[i % len(rec_cols)]:
                    # Hiển thị ảnh
                    image_url = df.loc[idx, "Link ảnh"] if "Link ảnh" in df.columns else None
                    if image_url:
                        st.image(image_url, width=120)

                    # Hiển thị thông tin
                    st.markdown(f"**{df.loc[idx, 'Tên sản phẩm']}**")
                    st.caption(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
                    st.caption(f"Giá: {df.loc[idx, 'Giá']}")
                    st.info(f"Tương đồng: `{rec['similarity']:.3f}`")
        else:
            st.warning(f"Không tìm thấy sản phẩm tương tự nào có độ tương đồng lớn hơn {threshold:.2f}.")
