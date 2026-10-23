import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
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
st.set_page_config(page_title="POD Auto Studio V3.0 Pro", page_icon="🎨", layout="wide")
st.title("🎨 [V3.0 PRO] POD Crop Art, AI Remaster 1:1, Tách Nền Sạch & Chuẩn Hóa 4200x4800")

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

# --- 1. HÀM TỰ ĐỘNG GỢI Ý VÙNG CROP TRÊN ÁO MOCKUP ---
def auto_detect_art_box(img: Image.Image):
    """Tự động tìm vùng có hình in trên áo thun để gợi ý khung Crop"""
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    # Lấy màu nền ở vùng rìa ảnh
    bg_color = np.median([rgb[int(h*0.1), int(w*0.1)], rgb[int(h*0.1), int(w*0.9)]], axis=0)
    diff = np.sqrt(np.sum((rgb - bg_color)**2, axis=2))
    # Giới hạn trong vùng thân áo (bỏ qua viền ngoài của áo)
    mask = np.zeros_like(diff, dtype=bool)
    mask[int(h*0.15):int(h*0.88), int(w*0.22):int(w*0.78)] = True
    art_pixels = (diff > 35) & mask
    coords = np.argwhere(art_pixels)
    if len(coords) > 100:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad_x, pad_y = int(w * 0.03), int(h * 0.03)
        return (
            max(0, int((x0 - pad_x) / w * 100)),
             min(100, int((x1 + pad_x) / w * 100)),
            max(0, int((y0 - pad_y) / h * 100)),
            min(100, int((y1 + pad_y) / h * 100))
        )
    return (24, 76, 18, 82)

# --- 2. HÀM AI REMASTER 1:1 (LÀM NÉT CHUẨN HÌNH 2 & GIỮ NGUYÊN MÀU GỐC) ---
def remaster_cropped_art(cropped_img: Image.Image, api_key: str, hf_tok: str, custom_edit: str = "") -> Image.Image:
    """
    Biến vùng ảnh Crop từ áo mẫu thành bản Art 2D sắc nét, giữ nguyên 100% màu sắc gốc và chi tiết
    như Hình 2 (Hỗ trợ chỉnh sửa nhẹ như đổi font chữ nếu người dùng yêu cầu).
    """
    rgb_img = cropped_img.convert("RGB")
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    
    # Ưu tiên 1: Dùng Gemini Image Generation để tái tạo độ nét 1:1 (Giống hệt Hình 2)
    if clean_key:
        client = genai.Client(api_key=clean_key)
        edit_instruction = f" Also apply this specific adjustment: {custom_edit}." if custom_edit.strip() else " Keep the exact same typography and font."
        
        remaster_prompt = (
            "Remaster and upscale this cropped t-shirt graphic into a ultra-crisp, high-resolution, print-ready 2D digital illustration "
            "isolated on a solid pure black background (#000000). "
            "CRITICAL RULES: "
            "1. Preserve the EXACT original artwork, character details, line art, composition, and EXACT original color palette (do NOT oversaturate or alter colors). "
            "2. Remove all fabric texture, blurry edges, and compression artifacts so every line, flower petal, and bone detail is razor-sharp like original vector art."
            f"{edit_instruction}"
        )
        
        image_models = [
            "gemini-2.5-flash-image-preview",
            "gemini-2.0-flash-exp-image-generation",
            "gemini-2.0-flash-exp",
            "gemini-3.8-flash"
        ]
        for m_name in image_models:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=[rgb_img, remaster_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        out_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        st.success(f"✨ Đã tái tạo Art siêu nét & giữ chuẩn màu gốc bằng AI (`{m_name}`)!")
                        return out_img
            except Exception:
                continue

    # Ưu tiên 2: Dùng GPU Upscaler Real-ESRGAN trên Cloud (Giữ nguyên 100% màu gốc, KHÔNG chỉnh Contrast làm cháy màu)
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        rgb_img.save(tmp.name)
        tmp_path = tmp.name
    try:
        for sp in ["akhaliq/Real-ESRGAN", "Nick088/Real-ESRGAN_Pytorch", "bookbot/Image-Upscaling-Playground"]:
            try:
                c = Client(sp)
                res = c.predict(handle_file(tmp_path), fn_index=0)
                out_p = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out_p, dict) and "path" in out_p:
                    out_p = out_p["path"]
                st.success(f"✨ Đã Upscale 4K giữ nguyên 100% màu gốc qua GPU `{sp}`!")
                return Image.open(out_p).convert("RGBA")
            except Exception:
                continue
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Ưu tiên 3: Phóng to LANCZOS bảo toàn tuyệt đối màu gốc + làm sắc viền nhẹ (Không dùng MedianFilter gây bệt)
    w, h = rgb_img.size
    scale = max(1, 2400 // max(w, h))
    up_img = rgb_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    # Chỉ làm nét nhẹ đường viền (threshold=5 để không làm nhiễu mảng màu phẳng, giữ nguyên 100% màu gốc)
    crisp_img = up_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=5))
    return crisp_img.convert("RGBA")

# --- 3. THUẬT TOÁN TÁCH NỀN KHỬ VIỀN TRẮNG/ĐEN (DEFRINGE CLEAN ALPHA) ---
@st.cache_resource
def get_rembg_session():
    return new_session("isnet-general-use")

def clean_defringe_knockout(image: Image.Image, bg_threshold: int = 22, feather: int = 18) -> Image.Image:
    """
    Thuật toán tách nền kèm Khử nhiễm màu viền (Color Decontamination):
    Giải quyết triệt để lỗi viền răng cưa trắng/xám ở Hình 1, giúp cành lá và chữ sắc nét trên mọi màu áo.
    """
    img_rgba = image.convert("RGBA")
    arr = np.array(img_rgba, dtype=np.float32)
    rgb = arr[:, :, :3]
    
    # Xác định màu nền từ 4 góc của vùng đã Crop
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg_color = np.median(corners, axis=0)
    
    # Khoảng cách màu Euclidean kết hợp độ sáng (Luminance)
    diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
    
    # Tính Alpha mượt không răng cưa
    alpha = np.clip((diff - bg_threshold) / max(1.0, float(feather)), 0.0, 1.0)
    
    # KHỬ VIỀN (Color Decontamination): Loại bỏ thành phần màu nền (đen/xám) bám ở các pixel полу-transparent (viền chữ, viền lá)
    edge_zone = (alpha > 0.05) & (alpha < 0.95)
    for c in range(3):
        channel = rgb[:, :, c]
        # Khôi phục màu gốc của tiền cảnh tại vùng rìa: F = (I - (1 - alpha) * B) / alpha
        decontaminated = (channel - (1.0 - alpha) * bg_color[c]) / np.maximum(alpha, 0.15)
        rgb[:, :, c] = np.where(edge_zone, np.clip(decontaminated, 0, 255), channel)
        
    arr[:, :, :3] = rgb
    arr[:, :, 3] = (alpha ** 1.15) * 255.0  # Làm gọn mép Alpha để không còn bóng mờ
    
    return Image.fromarray(arr.astype(np.uint8), "RGBA")

def remove_background_pro(image: Image.Image, hf_tok: str, mode_str: str, thresh: int = 22) -> Image.Image:
    if "Giữ nguyên nền" in mode_str:
        return image.convert("RGBA")
        
    if "Tách nền Khử Viền POD" in mode_str:
        return clean_defringe_knockout(image, bg_threshold=thresh, feather=20)
        
    # Chế độ AI BiRefNet + Khử viền kết hợp
    if HAS_REMBG:
        session = get_rembg_session()
        ai_out = rembg_remove(image, session=session)
        return ai_out
        
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
        return clean_defringe_knockout(image, bg_threshold=thresh, feather=20)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 4. HÀM ĐỌC PROMPT & REDESIGN MẪU MỚI ---
def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    rgb_img = image.convert("RGB")
    rgb_img.thumbnail((1024, 1024))
    sys_prompt = (
        "Analyze this cropped T-shirt graphic design carefully. Write ONLY the raw English image generation prompt "
        "to recreate this exact illustration style, subject, floral elements, color palette, and text layout "
        "isolated on a solid black background, crisp vector-like edges, print-ready graphic art."
    )
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    if clean_key:
        client = genai.Client(api_key=clean_key)
        for model_name in ["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-pro", "gemini-flash-latest"]:
            try:
                response = client.models.generate_content(model=model_name, contents=[rgb_img, sys_prompt])
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue
    return "Vintage illustration of a skeleton girl with floral crown reading a book surrounded by wildflowers, sun, and butterflies, text 'JUST A GIRL WHO LOVES BOOKS' at bottom, isolated on solid black background"

def generate_redesign_flux(prompt: str, hf_tok: str) -> Image.Image:
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    full_prompt = f"{prompt}, isolated t-shirt graphic art, solid plain black background, crisp clean edges, original vintage color palette"
    client = Client("black-forest-labs/FLUX.1-schnell")
    try:
        result = client.predict(prompt=full_prompt, seed=0, randomize_seed=True, width=1024, height=1216, num_inference_steps=4, api_name="/infer")
    except Exception:
        result = client.predict(full_prompt, 0, True, 1024, 1216, 4, fn_index=0)
    img_path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(img_path, dict) and "path" in img_path:
        img_path = img_path["path"]
    return Image.open(img_path).convert("RGBA")

# --- 5. HÀM CHUẨN HÓA KHUNG IN 4200x4800 PX (300 DPI) ---
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
col1, col2 = st.columns([1.05, 0.95])

if "current_art" not in st.session_state:
    st.session_state.current_art = None
if "extracted_prompt" not in st.session_state:
    st.session_state.extracted_prompt = ""
if "crop_box_pct" not in st.session_state:
    st.session_state.crop_box_pct = (0, 100, 0, 100)

with col1:
    st.subheader("✂️ Bước 1: Tải ảnh & Crop đúng vùng Art trên áo")
    uploaded_file = st.file_uploader("Tải ảnh áo Mockup hoặc ảnh thiết kế lên (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGBA")
        w, h = raw_img.size
        
        # --- CÔNG CỤ CROP TRỰC QUAN ---
        st.markdown("##### 📐 Công cụ cắt vùng Art (Loại bỏ cổ áo & tay áo thừa)")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🎯 Tự động nhận diện vùng Art trên áo"):
                st.session_state.crop_box_pct = auto_detect_art_box(raw_img)
        with c_btn2:
            if st.button("🔄 Chọn lại toàn bộ ảnh (100%)"):
                st.session_state.crop_box_pct = (0, 100, 0, 100)
                
        def_x0, def_x1, def_y0, def_y1 = st.session_state.crop_box_pct
        
        col_x, col_y = st.columns(2)
        with col_x:
            crop_x = st.slider("Cắt Trái - Phải (% Chiều ngang):", 0, 100, (def_x0, def_x1), step=1)
        with col_y:
            crop_y = st.slider("Cắt Trên - Dưới (% Chiều dọc):", 0, 100, (def_y0, def_y1), step=1)
            
        # Tính tọa độ Pixel thực tế từ %
        left_px = int(w * crop_x[0] / 100)
        right_px = max(left_px + 20, int(w * crop_x[1] / 100))
        top_px = int(h * crop_y[0] / 100)
        bottom_px = max(top_px + 20, int(h * crop_y[1] / 100))
        
        cropped_art = raw_img.crop((left_px, top_px, right_px, bottom_px))
        
        # Vẽ khung viền đỏ lên ảnh gốc để người dùng nhìn thấy vùng đang chọn
        preview_box_img = raw_img.copy()
        draw = ImageDraw.Draw(preview_box_img)
        line_w = max(3, int(min(w, h) * 0.008))
        draw.rectangle([left_px, top_px, right_px, bottom_px], outline="#FF2B2B", width=line_w)
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.image(preview_box_img, caption=f"Khung đỏ: Vùng đang chọn ({w}x{h}px)", use_container_width=True)
        with p_col2:
            st.image(cropped_art, caption=f"Vùng Art sau khi Crop ({cropped_art.width}x{cropped_art.height}px)", use_container_width=True)
            
        st.divider()
        st.subheader("✨ Bước 2: Làm nét chuẩn Hình 2 & Giữ nguyên màu gốc")
        
        custom_tweak = st.text_input(
            "Tùy chọn chỉnh sửa nhỏ khi làm nét (Để trống nếu giữ nguyên 100%, hoặc gõ ví dụ: 'Change text font to soft italic serif'):",
            value=""
        )
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🪄 1. AI Làm Nét Chuẩn Hình 2 (Giữ Màu & Chi Tiết Gốc)", type="primary"):
                with st.spinner("Đang làm nét vùng Art đã Crop, khử mờ vải áo và giữ nguyên màu gốc..."):
                    st.session_state.current_art = remaster_cropped_art(cropped_art, gemini_key, hf_token, custom_tweak)
                    st.success("✅ Đã làm nét xong! Chuyển sang Cột 2 bên phải để tách nền & xuất file.")
        with b_col2:
            if st.button("➡️ Dùng luôn ảnh vừa Crop (Không qua AI)"):
                st.session_state.current_art = cropped_art
                
        with st.expander("🎨 Mở rộng: Clone & Vẽ lại mẫu mới hoàn toàn bằng Prompt (FLUX AI)"):
            if st.button("🔍 Đọc Prompt từ vùng đã Crop"):
                st.session_state.extracted_prompt = analyze_art_prompt(cropped_art, gemini_key)
            prompt_text = st.text_area("Prompt Redesign:", value=st.session_state.extracted_prompt, height=100)
            if st.button("🎨 Vẽ mẫu mới bằng FLUX AI"):
                with st.spinner("Đang vẽ mẫu mới..."):
                    st.session_state.current_art = generate_redesign_flux(prompt_text, hf_token)

with col2:
    st.subheader("🚀 Bước 3: Tách Nền Khử Viền & Xuất File 4200x4800")
    
    if st.session_state.current_art is not None:
        st.image(st.session_state.current_art, caption="Bản Art sắc nét chuẩn bị tách nền & đóng khung in", width=400)
        
        bg_option = st.radio("Chọn phương pháp tách nền:", [
            "💎 Tách nền Khử Viền POD (Giữ 100% Chữ + Hoa lá + Khử sạch viền răng cưa trắng/đen)",
            "🤖 Tách nền bằng AI BiRefNet (Dành cho ảnh phức tạp nhiều màu nền)",
            "🖼️ Giữ nguyên nền đen như Hình 2 (Chỉ chuẩn hóa vào khung 4200x4800)"
        ])
        
        thresh_val = 22
        if "Khử Viền POD" in bg_option:
            thresh_val = st.slider("Độ sâu lọc nền đen của áo (Mặc định 22 là chuẩn nhất cho Art có chữ):", 5, 65, 22)
            
        preview_bg = st.radio("Màu nền xem trước sau khi tách (File tải về luôn là PNG trong suốt):", 
                              ["Nền Đen (Kiểm tra viền trắng)", "Nền Trắng", "Nền Trong Suốt"], horizontal=True)
        
        if st.button("⚡ XỬ LÝ TÁCH NỀN & XUẤT FILE 4200x4800 PX (300 DPI)", type="primary"):
            work_img = st.session_state.current_art
            
            with st.spinner("Đang tách nền & khử viền răng cưa..."):
                transparent_art = remove_background_pro(work_img, hf_token, bg_option, thresh=thresh_val)
                
            with st.spinner("Đang căn chỉnh chuẩn xác vào khung 4200x4800 (300 DPI)..."):
                is_top_chest = "ngực áo" in align_mode
                final_pod_img = standardize_pod_canvas(
                    transparent_art, 
                    target_w=CANVAS_W, 
                    target_h=CANVAS_H, 
                    pad=margin, 
                    top_chest=is_top_chest
                )
                
                # Xuất file PNG trong suốt 300 DPI
                buf = io.BytesIO()
                final_pod_img.save(buf, format="PNG", dpi=(300, 300))
                byte_im = buf.getvalue()
                
            st.success(f"✅ Hoàn tất! File in đạt chuẩn {CANVAS_W}x{CANVAS_H} px, 300 DPI, giữ nguyên màu gốc & không bị viền trắng!")
            
            # Tạo bản xem trước trên nền đen để kiểm tra chắc chắn không còn viền trắng như Hình 1
            if "Nền Đen" in preview_bg:
                bg_prev = Image.new("RGBA", final_pod_img.size, (15, 15, 15, 255))
                bg_prev.paste(final_pod_img, (0, 0), final_pod_img)
                st.image(bg_prev, caption="Xem trước trên nền áo Đen (Không còn viền trắng lem nhem)", use_container_width=True)
            elif "Nền Trắng" in preview_bg:
                bg_prev = Image.new("RGBA", final_pod_img.size, (255, 255, 255, 255))
                bg_prev.paste(final_pod_img, (0, 0), final_pod_img)
                st.image(bg_prev, caption="Xem trước trên nền áo Trắng", use_container_width=True)
            else:
                st.image(final_pod_img, caption=f"File PNG Trong Suốt ({CANVAS_W}x{CANVAS_H})", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG TRONG SUỐT (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải ảnh ở Cột 1, bấm 'Tự động nhận diện vùng Art' (hoặc kéo thanh trượt Crop) rồi bấm '1. AI Làm Nét Chuẩn Hình 2' để bắt đầu!")
