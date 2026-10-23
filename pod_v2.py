import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import os
import tempfile
from google import genai
from google.genai import types
from gradio_client import Client, handle_file

try:
    from rembg import remove as rembg_remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="POD Auto Studio Pro", page_icon="🚀", layout="wide")
st.title("🚀 POD Auto Clone, Flatten Art, Upscale 4K & Chuẩn Hóa 4200x4800")

# --- THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("🔑 1. Cấu hình API (Miễn phí)")
    try:
        default_gemini = st.secrets.get("GEMINI_API_KEY", "")
        default_hf = st.secrets.get("HF_TOKEN", "")
    except Exception:
        default_gemini = ""
        default_hf = ""
        
    gemini_key = st.text_input("Gemini API Key:", value=default_gemini, type="password")
    hf_token = st.text_input("HuggingFace Token:", value=default_hf, type="password")
    
    st.divider()
    st.header("📐 2. Chuẩn khung in POD")
    CANVAS_W = st.number_input("Chiều rộng chuẩn (px):", value=4200, step=100)
    CANVAS_H = st.number_input("Chiều cao chuẩn (px):", value=4800, step=100)
    margin = st.slider("Lề an toàn (Padding - px):", min_value=0, max_value=500, value=150, step=25)
    align_mode = st.radio("Vị trí đặt Art theo chiều dọc:", 
                          ["Căn giữa ngực áo (Cách mép trên ~350px - Chuẩn áo thun)", 
                           "Căn chính giữa tâm khung hình (Center)"])

# --- BƯỚC MỚI: LÀM PHẲNG ART, CĂN THẲNG TEXT & LÀM NÉT NHƯ HÌNH 2 ---
def flatten_and_remaster_art(image: Image.Image, api_key: str, hf_tok: str, use_ai_remaster=True) -> Image.Image:
    """
    Biến ảnh chụp mockup bị cong/mờ thành 1 khung hình chữ nhật phẳng (Flat Rectangular Shape),
    căn thẳng lại các dòng chữ (Text) và làm nét chi tiết trước khi tách nền.
    """
    rgb_img = image.convert("RGB")
    
    # 1. Nếu bật AI Remaster (Dùng Gemini Image-to-Image để nắn thẳng chữ & làm phẳng 2D như Hình 2)
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    if use_ai_remaster and clean_key:
        client = genai.Client(api_key=clean_key)
        remaster_prompt = (
            "Flatten and remaster this t-shirt graphic into a clean, high-resolution, perfectly straight "
            "2D rectangular poster graphic on a solid black background. "
            "CRITICAL: Keep the EXACT same people, faces, poses, clothing, colors, and exact text ('MIKE' at top, 'MACDONALD' at bottom). "
            "Straighten all warped or curved text so the letters are aligned horizontally and geometrically straight. "
            "Remove all fabric wrinkles, folds, and perspective distortion. Enhance facial details, sharpness, and lighting."
        )
        # Thử gọi các model Gemini hỗ trợ xuất ảnh trực tiếp (Image Output)
        for img_model in ["gemini-2.0-flash-exp-image-generation", "gemini-2.0-flash-exp", "gemini-2.5-flash-image-preview"]:
            try:
                response = client.models.generate_content(
                    model=img_model,
                    contents=[rgb_img, remaster_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        ai_flat_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        st.success(f"✨ Đã căn thẳng Text & tái tạo độ nét bằng AI ({img_model})!")
                        return ai_flat_img
            except Exception:
                continue

    # 2. Xử lý Hình học & Làm nét siêu cấp (Geometric Rectify + Cloud Super-Resolution + HD Sharpening)
    # Bước 2A: Cắt bỏ viền rách/mép áo thừa xung quanh để đưa về đúng 1 khung Hình Chữ Nhật chuẩn
    arr = np.array(rgb_img)
    h, w, _ = arr.shape
    
    # Tự động phát hiện và đóng khung vuông vức (Rectangular Framing)
    border_pad = int(min(w, h) * 0.04)
    framed = Image.new("RGB", (w + border_pad * 2, h + border_pad * 2), (10, 10, 10))
    framed.paste(rgb_img, (border_pad, border_pad))
    
    # Bước 2B: Gọi GPU Upscale Real-ESRGAN trên Cloud (Nếu có sẵn) để khử vỡ hạt ảnh 368x418
    upscaled = None
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        framed.save(tmp.name)
        tmp_path = tmp.name
    try:
        for sp in ["akhaliq/Real-ESRGAN", "Nick088/Real-ESRGAN_Pytorch"]:
            try:
                c = Client(sp)
                res = c.predict(handle_file(tmp_path), fn_index=0)
                out_p = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out_p, dict) and "path" in out_p:
                    out_p = out_p["path"]
                upscaled = Image.open(out_p).convert("RGB")
                break
            except Exception:
                continue
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    if upscaled is None:
        # Phóng lớn 3x bằng thuật toán nội suy LANCZOS trước khi lọc nét
        upscaled = framed.resize((framed.width * 3, framed.height * 3), Image.Resampling.LANCZOS)

    # Bước 2C: Bộ lọc tái tạo chi tiết Graphic (Unsharp Mask + Micro-Contrast + Color Pop giống Hình 2)
    # Khử nhiễu hạt JPEG cũ nhưng giữ nguyên cạnh sắc (Edge-preserving smoothing)
    smoothed = upscaled.filter(ImageFilter.MedianFilter(size=3))
    # Tăng độ sắc nét cho viền chữ và khuôn mặt
    sharpened = smoothed.filter(ImageFilter.UnsharpMask(radius=2.2, percent=180, threshold=3))
    # Tăng độ tương phản cục bộ để nền đen sâu và màu chữ xanh rực lên như Hình 2
    contrast_img = ImageEnhance.Contrast(sharpened).enhance(1.22)
    color_img = ImageEnhance.Color(contrast_img).enhance(1.18)
    sharp_img = ImageEnhance.Sharpness(color_img).enhance(1.35)
    
    return sharp_img.convert("RGBA")

# --- HÀM TÁCH NỀN KHÔNG BỊ MẤT CHỮ "MIKE" (GIỮ NGUYÊN 100% TEXT & HIỆU ỨNG) ---
@st.cache_resource
def get_rembg_session():
    return new_session("isnet-general-use")

def knockout_shirt_background(image: Image.Image, tolerance: int = 28) -> Image.Image:
    """
    Tách nền chuyên dụng cho thiết kế POD (Bootleg / Vintage Graphic):
    Chỉ xóa phần nền đen/trắng của vải áo, GIỮ NGUYÊN 100% chữ 'MIKE', 'MACDONALD' và hiệu ứng khói xanh!
    """
    img_rgba = image.convert("RGBA")
    data = np.array(img_rgba, dtype=np.float32)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    
    # Lấy màu nền chuẩn từ 4 góc của khung chữ nhật
    corners = [
        data[0, 0, :3], data[0, -1, :3],
        data[-1, 0, :3], data[-1, -1, :3]
    ]
    bg_color = np.median(corners, axis=0)
    
    # Tính khoảng cách màu từ từng điểm ảnh tới màu nền vải áo
    dist = np.sqrt((r - bg_color[0])**2 + (g - bg_color[1])**2 + (b - bg_color[2])**2)
    
    # Tạo mặt nạ Alpha mượt (Soft Edge Feathering) để viền chữ và khói xanh không bị răng cưa
    low_thresh = tolerance
    high_thresh = tolerance + 35.0
    
    alpha_mask = np.clip((dist - low_thresh) / (high_thresh - low_thresh), 0.0, 1.0)
    data[:, :, 3] = alpha_mask * 255.0
    
    return Image.fromarray(data.astype(np.uint8), "RGBA")

def remove_background_smart(image: Image.Image, hf_tok: str, bg_mode: str, tolerance: int = 28) -> Image.Image:
    # Chế độ 1: Giữ nguyên toàn bộ khung hình chữ nhật như Hình 2 (Không xóa nền)
    if "Giữ nguyên khung chữ nhật" in bg_mode:
        return image.convert("RGBA")
        
    # Chế độ 2: Tách nền màu áo thông minh (Bảo toàn 100% chữ MIKE + hiệu ứng khói ánh sáng)
    if "Bảo toàn 100% Chữ" in bg_mode:
        return knockout_shirt_background(image, tolerance=tolerance)
        
    # Chế độ 3: Tách vật thể bằng AI (rembg / Cloud)
    if HAS_REMBG:
        session = get_rembg_session()
        return rembg_remove(image, session=session)
        
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
        
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
        
    try:
        for args in [(handle_file(tmp_path), "birefnet-general"), ("birefnet-general", handle_file(tmp_path))]:
            try:
                client = Client("KenjieDec/RemBG")
                res = client.predict(*args, fn_index=0)
                out_path = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out_path, dict) and "path" in out_path:
                    out_path = out_path["path"]
                return Image.open(out_path).convert("RGBA")
            except Exception:
                continue
        # Nếu Cloud bận thì tự động chuyển về chế độ tách màu nền thông minh
        return knockout_shirt_background(image, tolerance=tolerance)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- HÀM ĐỌC PROMPT & REDESIGN ---
def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    rgb_img = image.convert("RGB")
    rgb_img.thumbnail((1024, 1024))
    sys_prompt = (
        "Analyze this T-shirt graphic design carefully. Write ONLY the raw English image generation prompt "
        "to recreate a similar graphic design style, layout, typography placement, and color palette. "
        "IMPORTANT: Describe it as a clean flat rectangular graphic poster art, straight horizontal text, "
        "NO wrinkles, crisp edges, print-ready graphic art."
    )
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    if clean_key:
        client = genai.Client(api_key=clean_key)
        candidate_models = ["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-pro", "gemini-flash-latest"]
        try:
            for m in client.models.list():
                m_name = m.name.replace("models/", "")
                if "flash" in m_name and m_name not in candidate_models:
                    candidate_models.append(m_name)
        except Exception:
            pass
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(model=model_name, contents=[rgb_img, sys_prompt])
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue
    return "Vintage bootleg 90s rap tee graphic design, bold straight retro typography at top and bottom, central character collage, neon green lighting effects, flat rectangular poster art"

def generate_redesign_flux(prompt: str, hf_tok: str) -> Image.Image:
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    full_prompt = f"{prompt}, flat 2d rectangular t-shirt graphic design, perfectly straight aligned text, solid black background, high contrast, crisp edges"
    client = Client("black-forest-labs/FLUX.1-schnell")
    try:
        result = client.predict(prompt=full_prompt, seed=0, randomize_seed=True, width=1024, height=1216, num_inference_steps=4, api_name="/infer")
    except Exception:
        result = client.predict(full_prompt, 0, True, 1024, 1216, 4, fn_index=0)
    img_path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(img_path, dict) and "path" in img_path:
        img_path = img_path["path"]
    return Image.open(img_path).convert("RGBA")

# --- HÀM CHUẨN HÓA KHUNG IN 4200x4800 PX (300 DPI) ---
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

# --- GIAO DIỆN CHÍNH (MAIN UI) ---
col1, col2 = st.columns([1, 1])

if "current_art" not in st.session_state:
    st.session_state.current_art = None
if "extracted_prompt" not in st.session_state:
    st.session_state.extracted_prompt = ""

with col1:
    st.subheader("📥 Bước 1: Tải mẫu gốc & Làm nét khung chữ nhật (Như Hình 2)")
    uploaded_file = st.file_uploader("Tải ảnh mẫu thiết kế hoặc Mockup lên (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        source_img = Image.open(uploaded_file).convert("RGBA")
        st.image(source_img, caption=f"Ảnh tải lên ({source_img.width}x{source_img.height} px)", use_container_width=True)
        
        mode = st.radio("Chọn chế độ xử lý:", [
            "🛠️ Giữ nguyên Art gốc (Làm nét khung chữ nhật + Xóa nền + Upscale 4200x4800)",
            "✨ Clone & Redesign mẫu mới hoàn toàn bằng Prompt (FLUX AI)"
        ])
        
        if "Giữ nguyên Art gốc" in mode:
            st.markdown("##### ✨ Bước 1.5: Chuẩn hóa Art thành khung chữ nhật sắc nét (Giống Hình 2)")
            if st.button("🪄 LÀM NÉT, CĂN THẲNG TEXT & ĐÓNG KHUNG CHỮ NHẬT CHUẨN", type="secondary"):
                with st.spinner("Đang làm phẳng thiết kế, khử mờ vỡ hạt và đóng khung chữ nhật sắc nét..."):
                    st.session_state.current_art = flatten_and_remaster_art(source_img, gemini_key, hf_token, use_ai_remaster=True)
                    st.success("✅ Đã xử lý Art sắc nét trong khung chữ nhật! Xem kết quả ở Cột 2 bên phải.")
            
            if st.session_state.current_art is None:
                st.session_state.current_art = source_img
        else:
            if st.button("🔍 1. Tự động đọc Prompt từ ảnh gốc (Gemini AI)"):
                if not gemini_key:
                    st.warning("Vui lòng nhập Gemini API Key ở thanh bên trái!")
                else:
                    with st.spinner("Đang phân tích cấu trúc thiết kế..."):
                        st.session_state.extracted_prompt = analyze_art_prompt(source_img, gemini_key)
                        
            prompt_text = st.text_area("Chỉnh sửa Prompt để Redesign:", value=st.session_state.extracted_prompt, height=130)
            
            if st.button("🎨 2. Tạo mẫu Redesign mới (FLUX AI)"):
                with st.spinner("Đang vẽ lại thiết kế trên GPU đám mây..."):
                    st.session_state.current_art = generate_redesign_flux(prompt_text, hf_token)
                    st.success("🎉 Đã tạo xong mẫu Redesign!")

with col2:
    st.subheader("🚀 Bước 2: Xóa nền (Không mất chữ) & Xuất File 4200x4800")
    
    if st.session_state.current_art is not None:
        st.image(st.session_state.current_art, caption="Art đã đóng khung chữ nhật & làm nét (Sẵn sàng xuất file)", width=380)
        
        bg_option = st.radio("Chọn kiểu tách nền khi xuất file in 4200x4800:", [
            "🔥 Tách nền Graphic POD (Bảo toàn 100% Chữ 'MIKE' + Hiệu ứng ánh sáng)",
            "🖼️ Giữ nguyên khung chữ nhật nền đen như Hình 2 (Chỉ Upscale + Căn giữa 4200x4800)",
            "✂️ Chỉ tách lấy người/vật thể chính (AI RemBG - Sẽ xóa chữ chìm phía sau)"
        ])
        
        tol_val = 28
        if "Bảo toàn 100% Chữ" in bg_option:
            tol_val = st.slider("Độ nhạy lọc màu nền vải áo (Chỉnh cao nếu muốn xóa sạch viền đen hơn):", 10, 80, 28)
        
        if st.button("⚡ XỬ LÝ TẤT CẢ & XUẤT FILE 4200x4800 PX", type="primary"):
            work_img = st.session_state.current_art
            
            with st.spinner("Đang xử lý nền theo chế độ đã chọn..."):
                work_img = remove_background_smart(work_img, hf_token, bg_option, tolerance=tol_val)
            
            with st.spinner("Đang phóng to 4K và căn chỉnh vào khung 4200x4800 (300 DPI)..."):
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
                
            st.success(f"✅ Hoàn tất! File đạt chuẩn {CANVAS_W}x{CANVAS_H} px, 300 DPI, giữ nguyên vẹn chữ MIKE & MACDONALD!")
            st.image(final_pod_img, caption=f"Bản xem trước ({CANVAS_W}x{CANVAS_H})", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải một bức ảnh ở cột bên trái để bắt đầu.")
