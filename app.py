import streamlit as st
from PIL import Image
import io
import os
import tempfile
from google import genai
from gradio_client import Client, handle_file

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="POD Auto Studio 4K", page_icon="👕", layout="wide")
st.title("👕 POD Auto Clone, Redesign, Upscale 4K & Chuẩn Hóa 4200x4800")

# --- THANH BÊN (SIDEBAR): CÀI ĐẶT API & THÔNG SỐ FILE IN ---
with st.sidebar:
    st.header("🔑 1. Cấu hình API (Miễn phí)")
    gemini_key = st.text_input("Gemini API Key (Đọc Prompt):", type="password", help="Lấy miễn phí tại aistudio.google.com")
    hf_token = st.text_input("HuggingFace Token (GPU AI):", type="password", help="Lấy miễn phí tại huggingface.co/settings/tokens")
    
    st.divider()
    st.header("📐 2. Chuẩn khung in POD")
    CANVAS_W = st.number_input("Chiều rộng chuẩn (px):", value=4200, step=100)
    CANVAS_H = st.number_input("Chiều cao chuẩn (px):", value=4800, step=100)
    margin = st.slider("Lề an toàn (Padding - px):", min_value=0, max_value=500, value=150, step=25, 
                       help="Khoảng cách từ mép thiết kế đến viền khung 4200x4800")
    align_mode = st.radio("Vị trí đặt Art theo chiều dọc:", 
                          ["Căn giữa ngực áo (Cách mép trên ~350px - Chuẩn áo thun)", 
                           "Căn chính giữa tâm khung hình (Center)"])

# --- CÁC HÀM XỬ LÝ LÕI (CORE FUNCTIONS) ---

def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    """Dùng Gemini 2.5 Flash đọc thiết kế gốc và trích xuất Prompt chi tiết"""
    client = genai.Client(api_key=api_key)
    sys_prompt = (
        "Analyze this T-shirt graphic design carefully. Write a detailed image generation prompt "
        "in English to recreate a similar graphic design style, layout, typography placement, and color palette. "
        "IMPORTANT: Describe it as an isolated flat vector/illustration graphic on a solid plain white background, "
        "NO t-shirt mockup, NO wrinkles, crisp edges, print-ready graphic art."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, sys_prompt]
    )
    return response.text

def generate_redesign_flux(prompt: str, hf_tok: str) -> Image.Image:
    """Dùng GPU miễn phí của FLUX.1-schnell trên HuggingFace để vẽ lại thiết kế mới"""
    client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_tok if hf_tok else None)
    result = client.predict(
        prompt=f"{prompt}, isolated t-shirt graphic design, flat 2d vector illustration, solid white background, high contrast, clean edges",
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1216,
        num_inference_steps=4,
        api_name="/infer"
    )
    # Result trả về đường dẫn file ảnh hoặc tuple (filepath, seed)
    img_path = result[0] if isinstance(result, tuple) else result
    return Image.open(img_path).convert("RGBA")

def remove_bg_birefnet(image: Image.Image, hf_tok: str) -> Image.Image:
    """Xóa nền sạch đến từng chi tiết nhỏ bằng model BiRefNet / BRIA RMBG"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Gọi Space BiRefNet (Model tách nền mã nguồn mở tốt nhất hiện tại cho POD)
        client = Client("ZhengPeng7/BiRefNet_demo", hf_token=hf_tok if hf_tok else None)
        result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )
        out_path = result[0] if isinstance(result, (list, tuple)) else result
        return Image.open(out_path).convert("RGBA")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def standardize_pod_canvas(img: Image.Image, target_w=4200, target_h=4800, pad=150, top_chest=True) -> Image.Image:
    """Cắt sát viền thừa, phóng to nét LANCZOS (4K) và căn giữa chuẩn khung 4200x4800"""
    img = img.convert("RGBA")
    
    # 1. Tự động cắt bỏ toàn bộ khoảng trong suốt thừa xung quanh (Auto-trim)
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    # 2. Tính toán tỷ lệ phóng to vừa khít khung an toàn (Upscale + Fit)
    max_w = target_w - (pad * 2)
    max_h = target_h - (pad * 2)
    
    orig_w, orig_h = img.size
    scale_ratio = min(max_w / orig_w, max_h / orig_h)
    
    new_w = int(orig_w * scale_ratio)
    new_h = int(orig_h * scale_ratio)
    
    # Sử dụng bộ lọc nội suy LANCZOS chất lượng cao nhất để giữ độ sắc nét
    resized_art = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Tạo khung Canvas trong suốt 4200x4800 và đặt Art vào vị trí chuẩn
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    
    # Luôn căn chính giữa theo chiều ngang (Horizontal Center)
    pos_x = (target_w - new_w) // 2
    
    # Chiều dọc: Đặt ở ngực áo hoặc chính giữa tâm
    if top_chest:
        pos_y = max(pad, 350)  # Cách mép trên 350px chuẩn in áo thun nếu art thấp, hoặc giữ lề pad
        if pos_y + new_h > target_h - pad:
            pos_y = (target_h - new_h) // 2
    else:
        pos_y = (target_h - new_h) // 2
        
    canvas.paste(resized_art, (pos_x, pos_y), resized_art)
    return canvas

# --- GIAO DIỆN CHÍNH (MAIN UI) ---

col1, col2 = st.columns([1, 1])

if "current_art" not in st.session_state:
    st.session_state.current_art = None
if "extracted_prompt" not in st.session_state:
    st.session_state.extracted_prompt = ""

with col1:
    st.subheader("📥 Bước 1: Tải mẫu gốc & Clone/Redesign")
    uploaded_file = st.file_uploader("Tải ảnh mẫu thiết kế hoặc Mockup lên (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        source_img = Image.open(uploaded_file).convert("RGBA")
        st.image(source_img, caption=f"Ảnh gốc ({source_img.width}x{source_img.height} px)", use_container_width=True)
        
        mode = st.radio("Chọn chế độ xử lý:", [
            "🛠️ Giữ nguyên Art gốc (Chỉ Xóa nền + Upscale + Chuẩn hóa 4200x4800)",
            "✨ Clone & Redesign mẫu mới bằng Prompt tự động (AI)"
        ])
        
        if "Clone & Redesign" in mode:
            if st.button("🔍 1. Tự động đọc Prompt từ ảnh gốc (Gemini AI)"):
                if not gemini_key:
                    st.warning("Vui lòng nhập Gemini API Key ở thanh bên trái!")
                else:
                    with st.spinner("Đang phân tích cấu trúc thiết kế và trích xuất Prompt..."):
                        st.session_state.extracted_prompt = analyze_art_prompt(source_img, gemini_key)
                        
            prompt_text = st.text_area("Chỉnh sửa Prompt để Redesign theo ý bạn:", value=st.session_state.extracted_prompt, height=130)
            
            if st.button("🎨 2. Tạo mẫu Redesign mới (FLUX AI)"):
                if not prompt_text:
                    st.warning("Hãy bấm đọc Prompt hoặc tự nhập Prompt trước!")
                else:
                    with st.spinner("Đang vẽ lại thiết kế trên GPU đám mây (khoảng 10-15 giây)..."):
                        st.session_state.current_art = generate_redesign_flux(prompt_text, hf_token)
                        st.success("Đã tạo xong mẫu Redesign!")
        else:
            st.session_state.current_art = source_img

with col2:
    st.subheader("🚀 Bước 2: Xóa nền, Upscale & Xuất File In 4200x4800")
    
    if st.session_state.current_art is not None:
        st.image(st.session_state.current_art, caption="Thiết kế chuẩn bị đưa vào xử lý in ấn", width=350)
        
        do_remove_bg = st.checkbox("Tự động tách nền sạch (AI BiRefNet)", value=True)
        
        if st.button("⚡ XỬ LÝ TẤT CẢ & XUẤT FILE 4200x4800 PX", type="primary"):
            work_img = st.session_state.current_art
            
            # 1. Xóa nền
            if do_remove_bg:
                with st.spinner("Đang tách nền chi tiết bằng AI..."):
                    try:
                        work_img = remove_bg_birefnet(work_img, hf_token)
                    except Exception as e:
                        st.error(f"Lỗi khi gọi server tách nền: {e}. Vui lòng kiểm tra HuggingFace Token.")
                        st.stop()
            
            # 2. Cắt viền thừa, Phóng to 4K & Căn giữa vào khung 4200x4800
            with st.spinner("Đang cắt viền thừa, làm nét 4K và căn chỉnh vào khung 4200x4800 (300 DPI)..."):
                is_top_chest = "ngực áo" in align_mode
                final_pod_img = standardize_pod_canvas(
                    work_img, 
                    target_w=CANVAS_W, 
                    target_h=CANVAS_H, 
                    pad=margin, 
                    top_chest=is_top_chest
                )
                
                # Lưu vào bộ nhớ đệm với chuẩn 300 DPI
                buf = io.BytesIO()
                final_pod_img.save(buf, format="PNG", dpi=(300, 300))
                byte_im = buf.getvalue()
                
            st.success(f"Hoàn tất! File đạt chuẩn {CANVAS_W}x{CANVAS_H} px, nền trong suốt, 300 DPI.")
            st.image(final_pod_img, caption=f"Bản xem trước (Đã chuẩn hóa {CANVAS_W}x{CANVAS_H})", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải một bức ảnh ở cột bên trái để bắt đầu.")