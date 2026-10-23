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
st.set_page_config(page_title="POD Auto Studio V5.3", page_icon="🎯", layout="wide")
st.title("🎯 [V5.3 PRO] POD Crop, Làm Nét Đồng Nhất Nền (4200x4800) & Tách Nền Sạch Khe Kín")

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

# --- 1. CÁC HÀM TIỆN ÍCH MÀU SẮC & TỰ ĐỘNG GỢI Ý VÙNG CROP ---
def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def rgb_to_hex(rgb_arr) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]))

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

# --- 2. HÀM ĐỒNG NHẤT MÀU NỀN & LÀM NÉT CHUẨN (GIỮ NGUYÊN 100% ART GỐC) ---
def unify_background_solid_color(
    image: Image.Image, 
    target_bg_rgb: np.ndarray, 
    bg_tolerance: int = 38
) -> Image.Image:
    """
    Chuyển toàn bộ phần nền vải áo (có thể bị loang màu, nhăn hoặc khác màu)
    về ĐỒNG NHẤT 1 MÀU DUY NHẤT (target_bg_rgb), đồng thời giữ nguyên 100% chi tiết và màu sắc của Art.
    """
    rgb_arr = np.array(image.convert("RGB"), dtype=np.float32)
    orig_bg_rgb = detect_dominant_bg_color(rgb_arr)
    
    # Tính khoảng cách màu từ từng điểm ảnh tới màu nền gốc của áo
    dr = rgb_arr[:, :, 0] - orig_bg_rgb[0]
    dg = rgb_arr[:, :, 1] - orig_bg_rgb[1]
    db = rgb_arr[:, :, 2] - orig_bg_rgb[2]
    r_mean = (rgb_arr[:, :, 0] + orig_bg_rgb[0]) / 2.0
    color_dist = np.sqrt((2.0 + r_mean / 256.0) * (dr**2) + 4.0 * (dg**2) + (2.0 + (255.0 - r_mean) / 256.0) * (db**2)) / np.sqrt(8.0)
    
    # Tạo mặt nạ phân tách Nền (0.0) và Art (1.0) với vùng chuyển tiếp mượt
    feather = 18.0
    art_mask = np.clip((color_dist - bg_tolerance) / feather, 0.0, 1.0)
    
    # Làm mịn mặt nạ để viền tiếp giáp giữa Art và màu nền mới mượt mà
    mask_pil = Image.fromarray((art_mask * 255.0).astype(np.uint8), mode="L")
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    smooth_mask = (np.array(mask_pil, dtype=np.float32) / 255.0)[:, :, np.newaxis]
    
    # Khử màu nền cũ bám ở rìa Art trước khi đặt lên màu nền mới
    safe_mask = np.maximum(smooth_mask, 0.22)
    orig_bg_3d = orig_bg_rgb.reshape((1, 1, 3))
    target_bg_3d = target_bg_rgb.reshape((1, 1, 3))
    
    decontaminated_art = np.clip((rgb_arr - (1.0 - safe_mask) * orig_bg_3d) / safe_mask, 0.0, 255.0)
    is_edge = (smooth_mask > 0.02) & (smooth_mask < 0.95)
    clean_fg = np.where(is_edge, decontaminated_art, rgb_arr)
    
    # Hòa trộn Art sạch lên tấm nền đồng nhất 100% (target_bg_3d)
    unified_rgb = clean_fg * smooth_mask + target_bg_3d * (1.0 - smooth_mask)
    return Image.fromarray(np.clip(unified_rgb, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

def remaster_and_unify_art(
    cropped_img: Image.Image, 
    api_key: str, 
    hf_tok: str, 
    target_bg_rgb: np.ndarray,
    bg_tolerance: int = 38,
    custom_edit: str = ""
) -> Image.Image:
    """
    Làm nét 4K giữ nguyên Art gốc + Đưa nền về đồng nhất 1 màu người dùng chọn.
    """
    rgb_img = cropped_img.convert("RGB")
    target_hex = rgb_to_hex(target_bg_rgb)
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    
    enhanced_img = None
    
    # 1. Thử làm nét & đồng nhất nền qua Gemini Image Generation
    if clean_key:
        client = genai.Client(api_key=clean_key)
        edit_instruction = f" Also apply this specific adjustment: {custom_edit}." if custom_edit.strip() else " Keep the exact same typography and font."
        
        remaster_prompt = (
            f"Remaster and upscale this cropped t-shirt graphic into an ultra-crisp, high-resolution, print-ready 2D digital illustration "
            f"isolated on a completely flat, solid uniform background of exact hex color {target_hex}. "
            "CRITICAL RULES: "
            "1. Preserve the EXACT original artwork, character details, line art, effects, composition, and EXACT original color palette of the design. "
            f"2. Make the background 100% solid uniform {target_hex} with zero fabric texture, zero wrinkles, and razor-sharp clean edges."
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
                        enhanced_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        st.success(f"✨ Đã tái tạo Art siêu nét & đồng nhất nền bằng AI ({m_name})!")
                        break
                if enhanced_img is not None:
                    break
            except Exception:
                continue

    # 2. Nếu chưa qua Gemini Image, gọi GPU Upscaler Real-ESRGAN để tăng nét 4K giữ nguyên 100% chi tiết gốc
    if enhanced_img is None:
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
                    enhanced_img = Image.open(out_p).convert("RGBA")
                    st.success(f"✨ Đã Upscale 4K giữ nguyên 100% màu & chi tiết gốc qua GPU {sp}!")
                    break
                except Exception:
                    continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 3. Dự phòng LANCZOS Ultra Crisp nếu server Cloud bận
    if enhanced_img is None:
        w, h = rgb_img.size
        scale = max(1, 2000 // max(w, h))
        up_img = rgb_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        enhanced_img = up_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=4)).convert("RGBA")

    # 4. Ép phẳng nền về ĐÚNG 1 MÀU ĐỒNG NHẤT TUYỆT ĐỐI (target_bg_rgb) để không còn vết vải loang hay lệch mã màu
    final_step2_img = unify_background_solid_color(enhanced_img, target_bg_rgb, bg_tolerance=bg_tolerance)
    return final_step2_img

# --- 3. HÀM CHUẨN HÓA KHUNG 4200x4800 CÓ MÀU NỀN ĐỒNG NHẤT (DÀNH CHO BƯỚC 2) ---
def standardize_solid_bg_canvas(
    img: Image.Image, 
    bg_rgb: np.ndarray, 
    target_w=4200, 
    target_h=4800, 
    pad=150, 
    top_chest=True,
    tol=35
) -> Image.Image:
    """
    Đặt bức ảnh Bước 2 vào đúng chuẩn khung in 4200x4800px trên tấm nền đồng nhất 100% (Chưa tách nền trong suốt).
    """
    rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
    # Tìm khung giới hạn thực sự của Art trên nền màu đồng nhất để căn giữa chuẩn xác
    diff = np.sqrt(np.sum((rgb_arr - bg_rgb.reshape((1, 1, 3))) ** 2, axis=2))
    coords = np.argwhere(diff > tol)
    if len(coords) > 50:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    else:
        cropped = img

    max_w = target_w - (pad * 2)
    max_h = target_h - (pad * 2)
    orig_w, orig_h = cropped.size
    scale_ratio = min(max_w / orig_w, max_h / orig_h)
    
    new_w = int(orig_w * scale_ratio)
    new_h = int(orig_h * scale_ratio)
    resized_art = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    bg_tuple = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)
    canvas = Image.new("RGBA", (target_w, target_h), bg_tuple)
    
    pos_x = (target_w - new_w) // 2
    if top_chest:
        pos_y = max(pad, 350)
        if pos_y + new_h > target_h - pad:
            pos_y = (target_h - new_h) // 2
    else:
        pos_y = (target_h - new_h) // 2
        
    canvas.paste(resized_art, (pos_x, pos_y))
    return canvas

# --- 4. CÔNG NGHỆ TÁCH NỀN TRONG SUỐT V5.3 (DÀNH CHO BƯỚC 3) ---
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

# --- 5. HÀM ĐỌC PROMPT & REDESIGN MẪU MỚI ---
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

# --- GIAO DIỆN CHÍNH (MAIN UI) ---
col1, col2 = st.columns([1.05, 0.95])

if "current_art" not in st.session_state:
    st.session_state.current_art = None
if "step2_solid_4200_bytes" not in st.session_state:
    st.session_state.step2_solid_4200_bytes = None
if "step2_preview_img" not in st.session_state:
    st.session_state.step2_preview_img = None
if "step2_bg_hex" not in st.session_state:
    st.session_state.step2_bg_hex = "#000000"
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
        st.subheader("✨ Bước 2: Làm nét, Đồng nhất màu nền & Xuất ảnh 4200x4800 (Chưa tách nền)")
        
        # Phát hiện màu nền gốc của vùng vừa Crop
        orig_crop_bg = detect_dominant_bg_color(np.array(cropped_art.convert("RGB"), dtype=np.float32))
        orig_crop_hex = rgb_to_hex(orig_crop_bg)
        
        st.markdown("##### 🎨 Tùy chọn màu nền đồng nhất cho Bước 2:")
        bg_choice_col1, bg_choice_col2 = st.columns([2.2, 1.0])
        with bg_choice_col1:
            step2_bg_option = st.radio(
                "Chọn màu nền đồng nhất muốn xuất ở Bước 2:",
                [
                    "⬛ Đen tuyền (#000000)",
                    "⬜ Trắng tinh (#FFFFFF)",
                    "🟦 Xanh dương (#0078D7)",
                    f"🪄 Giữ màu nền gốc nhưng làm phẳng đồng nhất ({orig_crop_hex})",
                    "🌈 Tự chọn màu bất kỳ (Chọn ở ô màu bên cạnh)"
                ]
            )
        with bg_choice_col2:
            if "Đen tuyền" in step2_bg_option:
                chosen_step2_hex = "#000000"
            elif "Trắng tinh" in step2_bg_option:
                chosen_step2_hex = "#FFFFFF"
            elif "Xanh dương" in step2_bg_option:
                chosen_step2_hex = "#0078D7"
            elif "Giữ màu nền gốc" in step2_bg_option:
                chosen_step2_hex = orig_crop_hex
            else:
                chosen_step2_hex = "#00FF00"
                
            picked_step2_hex = st.color_picker("Bảng màu nền Bước 2:", value=chosen_step2_hex)
            if "Tự chọn" in step2_bg_option:
                chosen_step2_hex = picked_step2_hex
                
        step2_bg_rgb = hex_to_rgb(chosen_step2_hex)
        
        step2_tol = st.slider(
            "Độ nhạy nhận diện nền áo cũ để thay sang nền đồng nhất:", 
            15, 80, 38, 
            help="Giúp làm phẳng hoàn toàn phần nền vải nhăn về đúng 1 mã màu đồng nhất mà không ảnh hưởng đến Art"
        )
        
        custom_tweak = st.text_input(
            "Tùy chọn chỉnh sửa nhỏ khi làm nét (Để trống nếu giữ nguyên 100%, hoặc gõ ví dụ: 'Change text font to soft italic serif'):",
            value=""
        )
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🪄 1. AI Làm Nét + Đồng Nhất Nền & Xuất 4200x4800", type="primary"):
                with st.spinner(f"Đang làm nét, đồng nhất nền về màu {chosen_step2_hex} và chuẩn hóa 4200x4800px..."):
                    remastered = remaster_and_unify_art(
                        cropped_art, gemini_key, hf_token, 
                        target_bg_rgb=step2_bg_rgb, bg_tolerance=step2_tol, custom_edit=custom_tweak
                    )
                    st.session_state.current_art = remastered
                    st.session_state.step2_bg_hex = chosen_step2_hex
                    
                    # Chuẩn hóa luôn bản 4200x4800px CÓ NỀN ĐỒNG NHẤT cho Bước 2
                    is_top_chest = "ngực áo" in align_mode
                    solid_4200 = standardize_solid_bg_canvas(
                        remastered, step2_bg_rgb, 
                        target_w=CANVAS_W, target_h=CANVAS_H, pad=margin, top_chest=is_top_chest, tol=step2_tol
                    )
                    buf_s2 = io.BytesIO()
                    solid_4200.save(buf_s2, format="PNG", dpi=(300, 300))
                    st.session_state.step2_solid_4200_bytes = buf_s2.getvalue()
                    
                    prev_s2 = solid_4200.copy()
                    prev_s2.thumbnail((700, 800))
                    st.session_state.step2_preview_img = prev_s2
                    
        with b_col2:
            if st.button("➡️ 2. Đồng Nhất Nền & Xuất 4200x4800 Nhanh (Không qua AI)"):
                with st.spinner(f"Đang đồng nhất nền về màu {chosen_step2_hex} và xuất khung 4200x4800px..."):
                    unified_fast = unify_background_solid_color(cropped_art, step2_bg_rgb, bg_tolerance=step2_tol)
                    st.session_state.current_art = unified_fast
                    st.session_state.step2_bg_hex = chosen_step2_hex
                    
                    is_top_chest = "ngực áo" in align_mode
                    solid_4200 = standardize_solid_bg_canvas(
                        unified_fast, step2_bg_rgb, 
                        target_w=CANVAS_W, target_h=CANVAS_H, pad=margin, top_chest=is_top_chest, tol=step2_tol
                    )
                    buf_s2 = io.BytesIO()
                    solid_4200.save(buf_s2, format="PNG", dpi=(300, 300))
                    st.session_state.step2_solid_4200_bytes = buf_s2.getvalue()
                    
                    prev_s2 = solid_4200.copy()
                    prev_s2.thumbnail((700, 800))
                    st.session_state.step2_preview_img = prev_s2

        # HIỂN THỊ KẾT QUẢ & NÚT TẢI FILE 4200x4800 CHƯA TÁCH NỀN NGAY TẠI BƯỚC 2
        if st.session_state.step2_solid_4200_bytes is not None and st.session_state.step2_preview_img is
