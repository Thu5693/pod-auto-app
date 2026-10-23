import streamlit as st
from PIL import Image
import io
import os
import tempfile
from google import genai
from gradio_client import Client, handle_file

try:
    from rembg import remove as rembg_remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# --- ĐỔI TIÊU ĐỀ V2.0 ĐỂ NHẬN BIẾT NGAY ĐÃ CHẠY ĐÚNG FILE MỚI ---
st.set_page_config(page_title="POD Auto Studio V2.0", page_icon="🚀", layout="wide")
st.title("🚀 [BẢN V2.0 - ĐÃ FIX LỖI] POD Auto Clone, Redesign & Chuẩn Hóa 4200x4800")

with st.sidebar:
    st.header("🔑 1. Cấu hình API (Miễn phí)")
    
    # Tự động đọc Key từ bộ nhớ bảo mật Secrets (nếu có)
    default_gemini = st.secrets.get("GEMINI_API_KEY", "")
    default_hf = st.secrets.get("HF_TOKEN", "")
    
    if default_gemini and default_hf:
        st.success("✅ Đã tự động kết nối sẵn API Key!")
        gemini_key = default_gemini
        hf_token = default_hf
        with st.expander("Thay đổi API Key khác (Tùy chọn)"):
            gemini_key = st.text_input("Gemini API Key:", value=default_gemini, type="password")
            hf_token = st.text_input("HuggingFace Token:", value=default_hf, type="password")
    else:
        gemini_key = st.text_input("Gemini API Key (Đọc Prompt):", value=default_gemini, type="password")
        hf_token = st.text_input("HuggingFace Token (GPU AI):", value=default_hf, type="password")
    
    st.divider()
    st.header("📐 2. Chuẩn khung in POD")
    CANVAS_W = st.number_input("Chiều rộng chuẩn (px):", value=4200, step=100)
    CANVAS_H = st.number_input("Chiều cao chuẩn (px):", value=4800, step=100)
    margin = st.slider("Lề an toàn (Padding - px):", min_value=0, max_value=500, value=150, step=25)
    align_mode = st.radio("Vị trí đặt Art theo chiều dọc:", 
                          ["Căn giữa ngực áo (Cách mép trên ~350px - Chuẩn áo thun)", 
                           "Căn chính giữa tâm khung hình (Center)"])

def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    client = genai.Client(api_key=api_key.strip())
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
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip()
    client = Client("black-forest-labs/FLUX.1-schnell")
    result = client.predict(
        prompt=f"{prompt}, isolated t-shirt graphic design, flat 2d vector illustration, solid white background, high contrast, clean edges",
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1216,
        num_inference_steps=4,
        api_name="/infer"
    )
    img_path = result[0] if isinstance(result, (list, tuple)) else result
    return Image.open(img_path).convert("RGBA")

@st.cache_resource
def get_rembg_session():
    return new_session("isnet-general-use")

def remove_background_smart(image: Image.Image, hf_tok: str) -> Image.Image:
    # 1. Tách nền trực tiếp trên máy bằng rembg (ISNet)
    if HAS_REMBG:
        session = get_rembg_session()
        return rembg_remove(image, session=session)
        
    # 2. Nếu chưa cài rembg thì gọi qua Cloud (đã bỏ hoàn toàn tham số hf_token)
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    try:
        client = Client("ZhengPeng7/BiRefNet_demo")
        result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )
        out_path = result[-1] if isinstance(result, (list, tuple)) else result
        return Image.open(out_path).convert("RGBA")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def standardize_pod_canvas(img: Image.Image, target_w=4200, target_h=4800, pad=150, top_chest=True) -> Image.Image:
    img = img.convert("RGBA")
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    max_w = target_w - (pad * 2)
    max_h = target_h - (pad * 2)
    orig_w, orig_h = img.size
    scale_ratio = min(max_w / orig_w, max_h / orig_h)
    
    new_w = int(orig_w * scale_ratio)
    new_h = int(orig_h * scale_ratio)
    resized_art = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    pos_x = (target_w - new_w) // 2
    if top_chest:
        pos_y = max(pad, 350)
        if pos_y + new_h > target_h - pad:
            pos_y = (target_h - new_h) // 2
    else:
        pos_y = (target_h - new_h) // 2
        
    canvas.paste(resized_art, (pos_x, pos_y), resized_art)
    return canvas

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
        
        do_remove_bg = st.checkbox("Tự động tách nền sạch (AI V2.0 - Không lỗi Token)", value=True)
        
        if st.button("⚡ XỬ LÝ TẤT CẢ & XUẤT FILE 4200x4800 PX", type="primary"):
            work_img = st.session_state.current_art
            
            if do_remove_bg:
                with st.spinner("Đang tách nền chi tiết (Lần đầu chạy sẽ tải model mất khoảng 20 giây)..."):
                    try:
                        work_img = remove_background_smart(work_img, hf_token)
                    except Exception as e:
                        st.error(f"Chi tiết lỗi tách nền: {e}")
                        st.stop()
            
            with st.spinner("Đang cắt viền thừa, làm nét 4K và căn chỉnh vào khung 4200x4800 (300 DPI)..."):
                is_top_chest = "ngực áo" in align_mode
                final_pod_img = standardize_pod_canvas(
                    work_img, 
                    target_w=CANVAS_W, 
                    target_h=CANVAS_H, 
                    pad=margin, 
                    top_chest=is_top_chest
                )
                
                buf = io.BytesIO()
                final_pod_img.save(buf, format="PNG", dpi=(300, 300))
                byte_im = buf.getvalue()
                
            st.success(f"✅ Hoàn tất! File đạt chuẩn {CANVAS_W}x{CANVAS_H} px, nền trong suốt, 300 DPI.")
            st.image(final_pod_img, caption=f"Bản xem trước ({CANVAS_W}x{CANVAS_H})", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải một bức ảnh ở cột bên trái để bắt đầu.")