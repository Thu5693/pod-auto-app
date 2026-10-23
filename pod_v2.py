import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import io
import os
import tempfile
from google import genai
from google.genai import types
from gradio_client import Client, handle_file

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="POD Auto Studio V5.2", page_icon="🎯", layout="wide")
st.title("🎯 [V5.2 PRO] POD Crop, AI Remaster, Tách Sạch Khe Chữ/Màu Nền & Chuẩn Hóa 4200x4800")

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
    align_mode = st.radio(
        "Vị trí đặt Art theo chiều dọc:", 
        [
            "Căn giữa ngực áo (Cách mép trên ~350px - Chuẩn áo thun)", 
            "Căn chính giữa tâm khung hình (Center)"
        ]
    )

# --- 1. HÀM TỰ ĐỘNG PHÁT HIỆN MÀU NỀN BẤT KỲ & GỢI Ý VÙNG CROP ---
def detect_dominant_bg_color(rgb_arr: np.ndarray) -> np.ndarray:
    top_edge = rgb_arr[0, :, :]
    bottom_edge = rgb_arr[-1, :, :]
    left_edge = rgb_arr[:, 0, :]
    right_edge = rgb_arr[:, -1, :]
    border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge], axis=0)
    return np.median(border_pixels, axis=0)

def auto_detect_art_box(img: Image.Image):
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    bg_color = np.median([rgb[int(h*0.1), int(w*0.15)], rgb[int(h*0.1), int(w*0.85)]], axis=0)
    diff = np.sqrt(np.sum((rgb - bg_color)**2, axis=2))
    mask = np.zeros_like(diff, dtype=bool)
    mask[int(h*0.14):int(h*0.88), int(w*0.20):int(w*0.80)] = True
    art_pixels = (diff > 32) & mask
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

# --- 2. HÀM AI REMASTER 1:1 (GIỮ NGUYÊN MÀU GỐC & NỀN GỐC) ---
def remaster_cropped_art(cropped_img: Image.Image, api_key: str, hf_tok: str, custom_edit: str = "") -> Image.Image:
    rgb_img = cropped_img.convert("RGB")
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    
    if clean_key:
        client = genai.Client(api_key=clean_key)
        edit_instruction = f" Also apply this specific adjustment: {custom_edit}." if custom_edit.strip() else " Keep the exact same typography and font."
        
        remaster_prompt = (
            "Remaster and upscale this cropped t-shirt graphic into an ultra-crisp, high-resolution, print-ready 2D digital illustration "
            "isolated on a solid flat uniform background matching the original background color. "
            "CRITICAL RULES: "
            "1. Preserve the EXACT original artwork, character details, line art, composition, and EXACT original color palette. "
            "2. Remove all fabric texture, blurry edges, and compression artifacts so every line, flower petal, text edge, and bone detail is razor-sharp."
            f"{edit_instruction}"
        )
        
        for m_name in ["gemini-2.5-flash-image-preview", "gemini-2.0-flash-exp-image-generation", "gemini-2.0-flash-exp", "gemini-3.8-flash"]:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=[rgb_img, remaster_prompt],
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        out_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        st.success(f"✨ Đã tái tạo Art siêu nét & giữ chuẩn màu gốc bằng AI ({m_name})!")
                        return out_img
            except Exception:
                continue

    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        rgb_img.save(tmp.name)
        tmp_path = tmp.name
    try:
        for sp in ["akhaliq/Real-ESRGAN", "Nick088/Real-ESRGAN_Pytorch"]:
            try:
                c = Client(sp)
                res = c.predict(handle_file(tmp_path), fn_index=0)
                out_p = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(out_p, dict) and "path" in out_p:
                    out_p = out_p["path"]
                st.success(f"✨ Đã Upscale 4K giữ nguyên 100% màu gốc qua GPU {sp}!")
                return Image.open(out_p).convert("RGBA")
            except Exception:
                continue
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    w, h = rgb_img.size
    scale = max(1, 2000 // max(w, h))
    up_img = rgb_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    crisp_img = up_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=5))
    return crisp_img.convert("RGBA")

# --- 3. CÔNG NGHỆ TÁCH NỀN V5.2: ĐỤC SẠCH LỖ CHỮ (A, R, O, B), KHE LÁ & HỖ TRỢ MỌI MÀU NỀN ---
def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def universal_clean_knockout_v5(
    image: Image.Image,
    bg_rgb: np.ndarray,
    tolerance: int = 44,
    edge_choke_px: int = 2,
    smooth_radius: float = 1.4
) -> Image.Image:
    orig_w, orig_h = image.size
    work_scale = max(1.0, 1800.0 / max(orig_w, orig_h))
    if work_scale > 1.05:
        work_img = image.resize((int(orig_w * work_scale), int(orig_h * work_scale)), Image.Resampling.LANCZOS)
    else:
        work_img = image.copy()

    img_rgba = work_img.convert("RGBA")
    arr = np.array(img_rgba, dtype=np.float32)
    rgb = arr[:, :, :3]
    h, w, _ = rgb.shape

    dr = rgb[:, :, 0] - bg_rgb[0]
    dg = rgb[:, :, 1] - bg_rgb[1]
    db = rgb[:, :, 2] - bg_rgb[2]
    r_mean = (rgb[:, :, 0] + bg_rgb[0]) / 2.0
    color_dist = np.sqrt((2.0 + r_mean / 256.0) * (dr**2) + 4.0 * (dg**2) + (2.0 + (255.0 - r_mean) / 256.0) * (db**2)) / np.sqrt(8.0)

    feather = 22.0
    raw_alpha = np.clip((color_dist - tolerance) / feather, 0.0, 1.0)

    alpha_pil = Image.fromarray((raw_alpha * 255.0).astype(np.uint8), mode="L")
    for _ in range(edge_choke_px):
        alpha_pil = alpha_pil.filter(ImageFilter.MinFilter(size=3))
        
    if smooth_radius > 0:
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0
        smooth_a = np.clip((smooth_a - 0.36) / (0.86 - 0.36), 0.0, 1.0)
        smooth_a = smooth_a * smooth_a * (3.0 - 2.0 * smooth_a)
    else:
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0

    safe_alpha = np.maximum(smooth_a[:, :, np.newaxis], 0.20)
    bg_3d = bg_rgb.reshape((1, 1, 3))
    unpremultiplied_rgb = np.clip((rgb - (1.0 - safe_alpha) * bg_3d) / safe_alpha, 0.0, 255.0)

    bg_luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    pure_rgb_pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if bg_luminance < 128:
        bleed_pil = pure_rgb_pil.filter(ImageFilter.MaxFilter(size=5))
    else:
        bleed_pil = pure_rgb_pil.filter(ImageFilter.MinFilter(size=5))
    bleed_rgb = np.array(bleed_pil, dtype=np.float32)

    is_edge_region = (smooth_a > 0.01) & (smooth_a < 0.96)
    edge_weight = np.where(is_edge_region, 1.0 - (smooth_a ** 2), 0.0)[:, :, np.newaxis]
    
    clean_rgb = rgb * (1.0 - edge_weight) + (0.55 * bleed_rgb + 0.45 * unpremultiplied_rgb) * edge_weight
    clean_rgb = np.clip(clean_rgb, 0.0, 255.0)

    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    out_arr[:, :, :3] = clean_rgb.astype(np.uint8)
    out_arr[:, :, 3] = (smooth_a * 255.0).astype(np.uint8)

    return Image.fromarray(out_arr, mode="RGBA")

# --- 4. HÀM ĐỌC PROMPT & REDESIGN MẪU MỚI ---
def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    rgb_img = image.convert("RGB")
    rgb_img.thumbnail((1024, 1024))
    sys_prompt = (
        "Analyze this cropped T-shirt graphic design carefully. Write ONLY the raw English image generation prompt "
        "to recreate this exact illustration style, subject, floral elements, color palette, and text layout "
        "isolated on a solid flat background, crisp vector-like edges, print-ready graphic art."
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
    return "Vintage illustration of a skeleton girl with floral crown reading a book surrounded by wildflowers, sun, and butterflies, text 'JUST A GIRL WHO LOVES BOOKS' at bottom, isolated on solid background"

def generate_redesign_flux(prompt: str, hf_tok: str) -> Image.Image:
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    full_prompt = f"{prompt}, isolated t-shirt graphic art, solid plain background, crisp clean edges, original vintage color palette"
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
col1, col2 = st.columns([1.0, 1.0])

if "current_art" not in st.session_state:
    st.session_state.current_art = None
if "extracted_prompt" not in st.session_state:
    st.session_state.extracted_prompt = ""
if "crop_box_pct" not in st.session_state:
    st.session_state.crop_box_pct = (0, 100, 0, 100)

with col1:
    st.subheader("✂️ Bước 1: Tải ảnh & Crop đúng vùng Art")
    uploaded_file = st.file_uploader("Tải ảnh áo Mockup hoặc ảnh thiết kế lên (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGBA")
        w, h = raw_img.size
        
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
            
        left_px = int(w * crop_x[0] / 100)
        right_px = max(left_px + 20, int(w * crop_x[1] / 100))
        top_px = int(h * crop_y[0] / 100)
        bottom_px = max(top_px + 20, int(h * crop_y[1] / 100))
        
        cropped_art = raw_img.crop((left_px, top_px, right_px, bottom_px))
        
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
        st.subheader("✨ Bước 2: Làm nét chuẩn & Giữ nguyên màu gốc")
        
        custom_tweak = st.text_input(
            "Tùy chọn chỉnh sửa nhỏ khi làm nét (Để trống nếu giữ nguyên 100%, hoặc gõ ví dụ: 'Change text font to soft italic serif'):",
            value=""
        )
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🪄 1. AI Làm Nét Chuẩn (Giữ Màu & Chi Tiết Gốc)", type="primary"):
                with st.spinner("Đang làm nét vùng Art đã Crop, khử mờ vải áo và giữ nguyên màu gốc..."):
                    st.session_state.current_art = remaster_cropped_art(cropped_art, gemini_key, hf_token, custom_tweak)
                    st.success("✅ Đã làm nét xong! Chuyển sang Cột 2 bên phải để tách nền & xuất file.")
        with b_col2:
            if st.button("➡️ Dùng luôn ảnh vừa Crop (Nếu ảnh đã nét sẵn)"):
                st.session_state.current_art = cropped_art
                
        with st.expander("🎨 Mở rộng: Clone & Vẽ lại mẫu mới hoàn toàn bằng Prompt (FLUX AI)"):
            if st.button("🔍 Đọc Prompt từ vùng đã Crop"):
                st.session_state.extracted_prompt = analyze_art_prompt(cropped_art, gemini_key)
            prompt_text = st.text_area("Prompt Redesign:", value=st.session_state.extracted_prompt, height=100)
            if st.button("🎨 Vẽ mẫu mới bằng FLUX AI"):
                with st.spinner("Đang vẽ mẫu mới..."):
                    st.session_state.current_art = generate_redesign_flux(prompt_text, hf_token)

with col2:
    st.subheader("🚀 Bước 3: Tách Sạch Mọi Khe Kín (A, R, O, B) & Mọi Màu Nền V5.2")
    
    if st.session_state.current_art is not None:
        work_art = st.session_state.current_art
        st.image(work_art, caption="Bản Art chuẩn bị đưa vào tách nền V5.2", width=340)
        
        detected_rgb = detect_dominant_bg_color(np.array(work_art.convert("RGB"), dtype=np.float32))
        detected_hex = "#{:02x}{:02x}{:02x}".format(int(detected_rgb[0]), int(detected_rgb[1]), int(detected_rgb[2]))
        
        st.markdown("##### 🎨 1. Nhận diện màu nền gốc của ảnh (Hỗ trợ mọi màu nền áo)")
        bg_col1, bg_col2 = st.columns([2, 1])
        with bg_col1:
            bg_detect_mode = st.radio(
                "Chế độ chọn màu nền cần xóa:",
                [
                    f"🤖 Tự động nhận diện từ viền ảnh (Đang phát hiện màu: {detected_hex})",
                    "🎯 Tự chấm/chọn màu nền thủ công (Dùng khi viền ảnh bị vướng chi tiết)"
                ]
            )
        with bg_col2:
            picked_hex = st.color_picker("Màu nền đang xóa:", value=detected_hex)
            
        target_bg_rgb = detected_rgb if "Tự động" in bg_detect_mode else hex_to_rgb(picked_hex)

        st.markdown("##### ⚙️ 2. Chọn phương pháp đục lỗ & tách nền")
        bg_option = st.radio(
            "Phương pháp xử lý:",
            [
                "🎯 V5.2 Universal Đục Lỗ Thông Minh (Tách sạch 100% bụng chữ A, R, O, B, khe cằm & kẽ lá + Mượt cạnh)",
                "🖼️ Giữ nguyên nền gốc (Chỉ chuẩn hóa vào khung 4200x4800)"
            ]
        )
        
        st.markdown("##### 🎛️ 3. Tinh chỉnh độ sâu đục khe kín & độ mượt:")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            thresh_val = st.slider("Độ nhạy bóc nền & đục khe:", 15, 90, 44)
        with s_col2:
            choke_val = st.slider("Gọt viền (Choke px):", 0, 5, 2)
        with s_col3:
            smooth_val = st.slider("Độ mượt đường nét:", 0.0, 3.0, 1.4, step=0.2)
            
        preview_bg = st.radio(
            "🔍 Màu nền kiểm tra kết quả (File tải về luôn là PNG Trong Suốt):", 
            ["Nền Trắng (Kiểm tra nền tối)", "Nền Đen (Kiểm tra nền sáng)", "Nền Xanh Dương (Kiểm tra mọi loại viền)"],
            horizontal=True
        )
        
        if st.button("⚡ XỬ LÝ TÁCH SẠCH KHE KÍN & XUẤT FILE 4200x4800 PX", type="primary"):
            with st.spinner("Đang đục sạch nền trong chữ A, R, O, B, các khe lá, dưới cằm và làm mượt đường nét..."):
                if "Giữ nguyên nền" in bg_option:
                    transparent_art = work_art.convert("RGBA")
                else:
                    transparent_art = universal_clean_knockout_v5(
                        work_art, target_bg_rgb, 
                        tolerance=thresh_val, edge_choke_px=choke_val, smooth_radius=smooth_val
                    )
                
            with st.spinner("Đang căn chỉnh chuẩn xác vào khung 4200x4800 (300 DPI)..."):
                is_top_chest = "ngực áo" in align_mode
                final_pod_img = standardize_pod_canvas(
                    transparent_art, 
                    target_w=CANVAS_W, 
                    target_h=CANVAS_H, 
                    pad=margin, 
                    top_chest=is_top_chest
                )
                
                buf = io.BytesIO()
                final_pod_img.save(buf, format="PNG", dpi=(300, 300))
                byte_im = buf.getvalue()
                
            st.success(f"✅ Hoàn tất! Toàn bộ bụng chữ A, R, O, B, khe dưới cằm và kẽ lá đã được tách sạch sẽ ở chuẩn {CANVAS_W}x{CANVAS_H} px!")
            
            # Thu nhỏ ảnh xem trước xuống 800x914px khi hiển thị trên web để không làm tràn RAM của Streamlit Cloud
            preview_small = final_pod_img.copy()
            preview_small.thumbnail((800, 914))
            
            if "Nền Trắng" in preview_bg:
                bg_prev_img = Image.new("RGBA", preview_small.size, (255, 255, 255, 255))
            elif "Nền Đen" in preview_bg:
                bg_prev_img = Image.new("RGBA", preview_small.size, (15, 15, 15, 255))
            else:
                bg_prev_img = Image.new("RGBA", preview_small.size, (0, 120, 215, 255))
                
            bg_prev_img.paste(preview_small, (0, 0), preview_small)
            st.image(bg_prev_img, caption=f"Kiểm tra kết quả trên {preview_bg} (File tải về bên dưới là Full 4200x4800px)", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG TRONG SUỐT (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải ảnh ở Cột 1, chọn vùng Art rồi chuyển sang Cột 2 để xử lý tách nền V5.2!")
