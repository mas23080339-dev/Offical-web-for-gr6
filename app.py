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
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data(csv_path="Gr6.csv"):
    """Tải dữ liệu từ CSV và tiền xử lý."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu tại đường dẫn '{csv_path}'. Vui lòng kiểm tra lại.")
        return pd.DataFrame() 

    # Làm sạch cột Từ khóa và Mô tả
    df["Từ khóa"] = df["Từ khóa"].fillna("").astype(str).str.replace(";", " ")
    df["Mô tả"] = df["Mô tả"].fillna("").astype(str)
    # Loại bỏ khoảng trắng thừa ở tên sản phẩm
    df["Tên sản phẩm"] = df["Tên sản phẩm"].fillna("").astype(str).str.strip() 

    # Gộp tất cả các trường văn bản lại để tính TF-IDF
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

# --------------------------------------------------------------------------------------
# 2) Tính toán TF-IDF và Ma trận Tương đồng giữa các Sản phẩm (Item-to-Item)
# --------------------------------------------------------------------------------------
@st.cache_data
def build_similarity_matrices(df):
    """Tính TF-IDF và ma trận tương đồng giữa các sản phẩm."""
    if df.empty:
        return None, None
        
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["FullText"])
    
    # Ma trận Tương đồng giữa các Sản phẩm (Item-to-Item Similarity Matrix)
    # Đây là mấu chốt để mô hình hoạt động giống như hàm evaluate_verbose trong Colab
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
st.markdown("Chọn sản phẩm chính xác để xem gợi ý Item-to-Item (tương tự như trong Colab).")

if df.empty or vectorizer is None:
    st.stop() 

# SỬ DỤNG SELECTBOX ĐỂ CHỌN CHÍNH XÁC SẢN PHẨM NHƯ EVALUATION TRONG COLAB
product_options = df["Tên sản phẩm"].unique()
selected_product_name = st.selectbox(
    "1. Vui lòng CHỌN SẢN PHẨM để nhận gợi ý:",
    options=product_options,
    index=0 # Chọn sản phẩm đầu tiên làm mặc định
)

# Lấy chỉ mục (index) của sản phẩm được chọn
try:
    # Lấy chỉ mục đầu tiên khớp với tên sản phẩm
    best_idx = df[df["Tên sản phẩm"] == selected_product_name].index[0] 
except IndexError:
    st.error("Lỗi: Không tìm thấy sản phẩm này trong dữ liệu. Vui lòng chọn sản phẩm khác.")
    st.stop()

# Độ tương đồng của sản phẩm với chính nó (luôn là 1.0)
best_score = 1.0 

st.markdown("---")

col_k, col_t = st.columns(2)
with col_k:
    top_k = st.number_input("2. Số lượng gợi ý (Top K):", min_value=1, max_value=20, value=5)
with col_t:
    threshold = st.slider("3. Ngưỡng tương đồng tối thiểu:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)


if selected_product_name:
    
    # --- B. HIỂN THỊ SẢN PHẨM CHÍNH ---
    st.subheader(f"✨ Sản phẩm Chính: {df.loc[best_idx, 'Tên sản phẩm']}")
    
    col_img, col_info = st.columns([1, 3])
    
    with col_img:
        image_url = df.loc[best_idx, "Link ảnh"] if "Link ảnh" in df.columns else None
        if image_url and image_url.strip():
            st.image(image_url, width=200, caption=df.loc[best_idx, 'Tên sản phẩm'])
        else:
            st.info("Không có hình ảnh.")

    with col_info:
        st.markdown(f"**Tên:** `{df.loc[best_idx, 'Tên sản phẩm']}`")
        st.markdown(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
        st.write(f"**Thương hiệu:** `{df.loc[best_idx, 'Thương hiệu']}`")
        st.markdown(f"**Giá:** `{df.loc[best_idx, 'Giá']}` | **Đánh giá:** `{df.loc[best_idx, 'Điểm đánh giá']}`")
        # Hiển thị điểm tương đồng là 1.0 vì đây là sản phẩm chính
        st.success(f"**Điểm Tương đồng (Item-to-Item Basis):** `{best_score:.3f}`") 
        
    st.markdown("---")
    
    # --- C. GỢI Ý SẢN PHẨM TƯƠNG TỰ (ITEM-TO-ITEM) ---
    st.subheader("Bạn cũng có thể thích (Gợi ý dựa trên Sản phẩm Chính):")

    recommendations = get_item_recommendations(best_idx, top_k, threshold)
    
    if recommendations:
        
        # Sử dụng st.columns để hiển thị gọn gàng hơn
        rec_cols = st.columns(min(len(recommendations), 4)) 

        for i, rec in enumerate(recommendations):
            idx = rec["index"]
            
            with rec_cols[i % len(rec_cols)]:
                # Hiển thị ảnh
                image_url = df.loc[idx, "Link ảnh"] if "Link ảnh" in df.columns else None
                if image_url and image_url.strip():
                    st.image(image_url, width=120)
                else:
                    # Thẻ thay thế nếu không có ảnh
                    st.markdown(f"<div style='height:120px; background-color:#333; color:white; padding:10px; border-radius: 5px; display:flex; align-items:center; justify-content:center; text-align:center;'>Ảnh đang cập nhật</div>", unsafe_allow_html=True)
                    

                # Hiển thị thông tin
                st.markdown(f"**{df.loc[idx, 'Tên sản phẩm']}**")
                
                # PHẦN MÔ TẢ TÓM TẮT (100 ký tự đầu tiên)
                description = df.loc[idx, "Mô tả"]
                st.caption(f"{description[:100]}...")
                
                st.caption(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
                st.caption(f"Giá: {df.loc[idx, 'Giá']}")
                st.info(f"Tương đồng: `{rec['similarity']:.3f}`")
    else:
        st.warning(f"Không tìm thấy sản phẩm tương tự nào có độ tương đồng lớn hơn {threshold:.2f} với '{selected_product_name}'.")
