import streamlit as st
import cv2
import numpy as np
import torch
from streamlit_drawable_canvas import st_canvas
import os

# Import từ utils
from utils import ModernCNN, preprocess_image_from_array, predict_top3

# --- CẤU HÌNH CƠ BẢN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa đường dẫn và cấu hình cho 2 model
MODELS_CONFIG = {
    "digits": {
        "name": " Nhận diện Chữ Số (0-9)",
        "path": "models/model_digits_10.pth",
        "num_classes": 10,
        "labels": {i: str(i) for i in range(10)}
    },
    "shapes": {
        "name": " Nhận diện Hình Học (Tròn, CN, Tam Giác)",
        "path": "models/model_shapes_3.pth",
        "num_classes": 3,
        # Lưu ý: Map nhãn này phải khớp với thứ tự lúc bạn train model shapes
        # Ví dụ: 0 là Tròn, 1 là CN, 2 là Tam giác (Kiểm tra lại notebook train của bạn)
        "labels": {0: "Hình Tròn", 1: "Hình Chữ Nhật", 2: "Hình Tam Giác"}
    }
}

st.set_page_config(page_title="Demo Nhận Diện Đa Model", layout="wide")

# --- SIDEBAR: CHỌN CHẾ ĐỘ ---
st.sidebar.title("⚙️ Cấu hình")
st.sidebar.write("Chọn bài toán bạn muốn kiểm thử:")

mode_selection = st.sidebar.radio(
    "Chọn Model:",
    options=["digits", "shapes"],
    format_func=lambda x: MODELS_CONFIG[x]["name"]
)

# Lấy cấu hình hiện tại dựa trên lựa chọn
current_config = MODELS_CONFIG[mode_selection]

# --- HÀM LOAD MODEL (Dynamic) ---
@st.cache_resource
def load_model(model_path, num_classes):
    """Load model dựa trên đường dẫn và số lớp"""
    if not os.path.exists(model_path):
        st.error(f"⚠️ LỖI: Không tìm thấy file model tại: {model_path}")
        return None
    
    try:
        # Khởi tạo model với số lớp tương ứng (10 hoặc 3)
        model = ModernCNN(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải model {model_path}: {e}")
        return None

# Load model ngay khi chọn
st.sidebar.divider()
# st.sidebar.info(f"Đang sử dụng: **{current_config['name']}**")
active_model = load_model(current_config["path"], current_config["num_classes"])

# --- GIAO DIỆN CHÍNH ---
st.title(f"Demo: {current_config['name']}")

col_input, col_process = st.columns([1, 2])
image_to_process = None

# --- CỘT TRÁI: INPUT ---
with col_input:
    st.subheader("1. Đầu vào")
    tab_draw, tab_upload = st.tabs(["🎨 Vẽ tay", "📂 Upload"])

    with tab_draw:
        st.write("Vẽ nét TRẮNG trên nền ĐEN:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{mode_selection}" # Key thay đổi để reset canvas khi đổi model
        )
        if st.button("Dự đoán hình vẽ", type="primary"):
            if canvas_result.image_data is not None:
                raw_img = canvas_result.image_data.astype('uint8')
                image_to_process = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2RGB)

    with tab_upload:
        uploaded_file = st.file_uploader("Chọn ảnh...", type=['png', 'jpg', 'jpeg'], key=f"uploader_{mode_selection}")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            decoded_img = cv2.imdecode(file_bytes, 1)
            st.image(decoded_img, channels="BGR", width=200)
            if st.button("Dự đoán ảnh upload", type="primary"):
                image_to_process = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

# --- CỘT PHẢI: XỬ LÝ & KẾT QUẢ ---
with col_process:
    if image_to_process is not None and active_model is not None:
        st.divider()
        
        # 1. Tiền xử lý
        final_pil, steps = preprocess_image_from_array(image_to_process)
        
        if final_pil is None:
            st.warning("Không tìm thấy đối tượng trong ảnh.")
        else:
            st.subheader("2. Các bước Xử lý (8 Steps)")
            # Hiển thị 8 bước
            items = list(steps.items())
            c1, c2, c3, c4 = st.columns(4)
            c5, c6, c7, c8 = st.columns(4)
            
            c1.image(items[0][1], "1. Gốc", use_container_width=True)
            c2.image(items[1][1], "2. Gray", use_container_width=True)
            c3.image(items[2][1], "3. Blur", use_container_width=True)
            c4.image(items[3][1], "4. Threshold", use_container_width=True)
            
            c5.image(items[4][1], "5. Box", use_container_width=True)
            c6.image(items[5][1], "6. Crop", use_container_width=True)
            c7.image(items[6][1], "7. Resize", use_container_width=True)
            c8.image(items[7][1], "8. Final (28x28)", use_container_width=True)

            st.divider()
            
            # 2. Dự đoán
            st.subheader(f"3. Kết quả ({mode_selection.upper()})")
            
            r1, r2 = st.columns([1, 3])
            with r1:
                st.image(final_pil, width=120, caption="Input Model")
            
            with r2:
                # Gọi hàm predict với model và bộ nhãn hiện tại
                top3 = predict_top3(final_pil, active_model, DEVICE, current_config["labels"])
                
                best = top3[0]
                st.success(f"🏆 DỰ ĐOÁN: **{best['label']}**")
                st.metric("Độ tin cậy", f"{best['conf']:.2f}%")
                
                st.write("Chi tiết Top 3:")
                for item in top3:
                    st.write(f"- {item['label']}: {item['conf']:.2f}%")
                    st.progress(int(item['conf']))
                    
    elif active_model is None:
        st.warning("Vui lòng kiểm tra lại file model trong thư mục models/")