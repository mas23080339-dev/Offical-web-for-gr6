import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Demo CBF Hybrid",
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
    item_similarity_matrix = cosine_similarity(tfidf_matrix) 
    
    return vectorizer, item_similarity_matrix

if not df.empty:
    vectorizer, item_similarity_matrix = build_similarity_matrices(df)
else:
    vectorizer, item_similarity_matrix = None, None
    
# --------------------------------------------------------------------------------------
# 3) Hàm Gợi ý Sản phẩm Tương tự (Sử dụng Item-to-Item Matrix)
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

st.title("Chào mừng đến với cửa hàng của chúng tôi!")
st.markdown("Hãy tìm kiếm sản phẩm của bạn!!.")

if df.empty or vectorizer is None:
    st.stop() 

# Khối điều khiển chung
with st.sidebar:
    st.header("Thiết lập Gợi ý")
    top_k = st.number_input("Số lượng gợi ý (Top K):", min_value=1, max_value=20, value=5)
    threshold = st.slider("Ngưỡng tương đồng tối thiểu:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# --- CHỌN CHẾ ĐỘ HOẠT ĐỘNG ---
mode = st.radio(
    "Vui lòng chọn chế độ hoạt động:",
    ("1. Tìm kiếm bằng Từ khóa (Search Mode)", "2. Chọn Sản phẩm Chính xác (Evaluation Mode)"),
    horizontal=True,
    index=0 # Mặc định là chế độ tìm kiếm
)

# Khởi tạo biến
best_idx = None
best_score = 0.0
is_accurate_mode = False

st.markdown("---")

if mode == "1. Tìm kiếm bằng Từ khóa (Search Mode)":
    # --- CHẾ ĐỘ 1: TÌM SẢN PHẨM BẰNG QUERY ---
    user_query = st.text_input(
        "Nhập từ khóa hoặc mô tả sản phẩm (ví dụ: Áo thun co giãn, ba lô chống nước):",
        key="query_input"
    )
    
    if user_query:
        # Tính toán độ tương đồng giữa Query và TẤT CẢ sản phẩm
        query_vec = vectorizer.transform([user_query])
        query_scores = cosine_similarity(query_vec, vectorizer.transform(df["FullText"]))[0]
        ranking_by_query = query_scores.argsort()[::-1]
        
        best_idx = ranking_by_query[0]
        best_score = query_scores[best_idx]
        
        if best_score < threshold:
            st.warning("Không tìm thấy sản phẩm nào đủ tương đồng với từ khóa của bạn. Vui lòng thử lại.")
            best_idx = None # Reset nếu không tìm thấy
        else:
            st.info(f"Đã tìm thấy sản phẩm chính: {df.loc[best_idx, 'Tên sản phẩm']} (Tương đồng Query: {best_score:.3f})")

elif mode == "2. Chọn Sản phẩm Chính xác (Evaluation Mode)":
    # --- CHẾ ĐỘ 2: CHỌN SẢN PHẨM TỪ DANH SÁCH ---
    product_options = df["Tên sản phẩm"].unique()
    selected_product_name = st.selectbox(
        "Chọn TÊN SẢN PHẨM chính xác để xem gợi ý:",
        options=product_options,
        key="selectbox_input"
    )

    if selected_product_name:
        is_accurate_mode = True
        try:
            best_idx = df[df["Tên sản phẩm"] == selected_product_name].index[0] 
            best_score = 1.0 # Độ tương đồng của sản phẩm với chính nó là 1.0
        except IndexError:
            st.error("Lỗi: Không tìm thấy sản phẩm này trong dữ liệu.")


# --- HIỂN THỊ KẾT QUẢ ---

if best_idx is not None:
    # --- B. HIỂN THỊ SẢN PHẨM CHÍNH (ĐẦU VÀO CỦA MÔ HÌNH GỢI Ý) ---
    st.subheader(f"Sản phẩm Chính ({'Đầu vào Gợi ý' if is_accurate_mode else 'Kết quả Tìm kiếm'}): {df.loc[best_idx, 'Tên sản phẩm']}")
    
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
        
        if is_accurate_mode:
            st.success(f"**Chế độ Evaluation:** Gợi ý dựa trên sản phẩm này.")
        else:
            st.success(f"**Độ tương đồng với Query:** `{best_score:.3f}`")
        
    st.markdown("---")
    
    # --- C. GỢI Ý SẢN PHẨM TƯƠNG TỰ (ITEM-TO-ITEM) ---
    st.subheader("Có thể bạn thích sản phẩm này:")

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
                    st.markdown(f"<div style='height:120px; background-color:#333; color:white; padding:10px; border-radius: 5px; display:flex; align-items:center; justify-content:center; text-align:center; font-size:12px;'>Ảnh đang cập nhật</div>", unsafe_allow_html=True)
                    

                # Hiển thị thông tin
                st.markdown(f"**{df.loc[idx, 'Tên sản phẩm']}**")
                
                # PHẦN MÔ TẢ TÓM TẮT (100 ký tự đầu tiên)
                description = df.loc[idx, "Mô tả"]
                st.caption(f"{description[:100]}...")
                
                st.caption(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
                st.caption(f"Giá: {df.loc[idx, 'Giá']}")
                st.info(f"Tương đồng: `{rec['similarity']:.3f}`")
    else:
        st.warning(f"Không tìm thấy sản phẩm tương tự nào có độ tương đồng lớn hơn {threshold:.2f}.")
