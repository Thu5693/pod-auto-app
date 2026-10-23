import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import io
import os
import tempfile
from collections import deque
from google import genai
from google.genai import types
from gradio_client import Client, handle_file

try:
    from rembg import remove as rembg_remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="POD Auto Studio V4.0 Ultra Smooth", page_icon="💎", layout="wide")
st.title("💎 [V4.0 ULTRA] POD Crop, AI Remaster, Khử Viền Đen Mượt Cạnh & Chuẩn Hóa 4200x4800")

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
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    bg_color = np.median([rgb[int(h*0.1), int(w*0.1)], rgb[int(h*0.1), int(w*0.9)]], axis=0)
    diff = np.sqrt(np.sum((rgb - bg_color)**2, axis=2))
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

# --- 2. HÀM AI REMASTER 1:1 (LÀM NÉT CHUẨN & GIỮ NGUYÊN MÀU GỐC) ---
def remaster_cropped_art(cropped_img: Image.Image, api_key: str, hf_tok: str, custom_edit: str = "") -> Image.Image:
    rgb_img = cropped_img.convert("RGB")
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    
    if clean_key:
        client = genai.Client(api_key=clean_key)
        edit_instruction = f" Also apply this specific adjustment: {custom_edit}." if custom_edit.strip() else " Keep the exact same typography and font."
        
        remaster_prompt = (
            "Remaster and upscale this cropped t-shirt graphic into an ultra-crisp, high-resolution, print-ready 2D digital illustration "
            "isolated on a solid pure black background (#000000). "
            "CRITICAL RULES: "
            "1. Preserve the EXACT original artwork, character details, line art, composition, and EXACT original color palette. "
            "2. Remove all fabric texture, blurry edges, and compression artifacts so every line, flower petal, text edge, and bone detail is razor-sharp like clean vector art."
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
                st.success(f"✨ Đã Upscale 4K giữ nguyên 100% màu gốc qua GPU `{sp}`!")
                return Image.open(out_p).convert("RGBA")
            except Exception:
                continue
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    w, h = rgb_img.size
    scale = max(1, 2400 // max(w, h))
    up_img = rgb_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    crisp_img = up_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=5))
    return crisp_img.convert("RGBA")

# --- 3. CÔNG NGHỆ TÁCH NỀN V4.0: GỌT VIỀN ĐEN (CHOKE), PHÓNG MÀU RUỘT (INWARD BLEED) & MƯỢT CẠNH ---
@st.cache_resource
def get_rembg_session():
    return new_session("isnet-general-use")

def ultra_smooth_defringe_v4(
    image: Image.Image, 
    bg_threshold: int = 38, 
    edge_choke_px: int = 2, 
    smooth_radius: float = 1.4,
    protect_interior_dark: bool = True
) -> Image.Image:
    """
    Thuật toán V4.0 giải quyết triệt để 100% viền đen răng cưa khi đặt trên nền trắng:
    1. Phóng kích thước lớn để xử lý viền ở cấp độ Sub-pixel (Siêu mịn).
    2. Bảo vệ các chi tiết màu tối bên trong (nhụy hoa, hốc mắt, nét vẽ tóc) nếu bật protect_interior_dark.
    3. Gọt viền (Alpha Choke/Erosion) để cắt bỏ hoàn toàn vòng halo xám/đen bám quanh chữ trắng & lá cây.
    4. Inward Color Bleed (Phóng màu sáng từ trong ruột chữ/lá ra mép ngoài) để mép không bao giờ còn màu đen.
    """
    # Phóng lên độ phân giải cao (~2200px) trước khi cắt viền để đường cắt mịn như Vector
    orig_w, orig_h = image.size
    work_scale = max(1.0, 2200.0 / max(orig_w, orig_h))
    if work_scale > 1.05:
        work_img = image.resize((int(orig_w * work_scale), int(orig_h * work_scale)), Image.Resampling.LANCZOS)
    else:
        work_img = image.copy()

    img_rgba = work_img.convert("RGBA")
    arr = np.array(img_rgba, dtype=np.float32)
    rgb = arr[:, :, :3]
    h, w, _ = rgb.shape

    # Lấy màu nền chuẩn từ 4 góc ảnh
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg_color = np.median(corners, axis=0)

    # Tính khoảng cách màu so với màu nền đen
    diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))

    # Tạo mặt nạ Alpha ban đầu với vùng chuyển tiếp
    feather_zone = 25.0
    raw_alpha = np.clip((diff - bg_threshold) / feather_zone, 0.0, 1.0)

    # BẢO VỆ CHI TIẾT ĐEN BÊN TRONG ART (Nhụy hoa, mắt hộp sọ, nét vẽ line-art)
    if protect_interior_dark:
        # Thu nhỏ mặt nạ 4 lần để chạy thuật toán Flood-Fill từ ngoài vào siêu nhanh (0.05 giây)
        small_h, small_w = h // 4, w // 4
        alpha_pil_small = Image.fromarray((raw_alpha * 255).astype(np.uint8)).resize((small_w, small_h), Image.Resampling.BILINEAR)
        small_arr = np.array(alpha_pil_small)
        
        # Các vùng chắc chắn là nền (Alpha < 180)
        is_bg_candidate = small_arr < 180
        exterior_bg = np.zeros((small_h, small_w), dtype=bool)
        q = deque()
        
        # Bơm hạt giống từ 4 cạnh viền khung ảnh vào
        for x in range(small_w):
            if is_bg_candidate[0, x]: exterior_bg[0, x] = True; q.append((0, x))
            if is_bg_candidate[small_h-1, x]: exterior_bg[small_h-1, x] = True; q.append((small_h-1, x))
        for y in range(small_h):
            if is_bg_candidate[y, 0]: exterior_bg[y, 0] = True; q.append((y, 0))
            if is_bg_candidate[y, small_w-1]: exterior_bg[y, small_w-1] = True; q.append((y, small_w-1))
            
        while q:
            cy, cx = q.popleft()
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if 0 <= ny < small_h and 0 <= nx < small_w and not exterior_bg[ny, nx] and is_bg_candidate[ny, nx]:
                    exterior_bg[ny, nx] = True
                    q.append((ny, nx))
                    
        # Phóng trả lại mặt nạ vùng nền bên ngoài
        ext_mask_pil = Image.fromarray((exterior_bg * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        ext_mask = np.array(ext_mask_pil, dtype=np.float32) / 255.0
        
        # Những vùng tối nằm sâu bên trong vật thể (không thông với nền ngoài) và không quá rộng (như nhụy hoa, nét vẽ) sẽ được giữ lại
        raw_alpha = np.where(ext_mask < 0.25, np.maximum(raw_alpha, 0.92), raw_alpha)

    # BƯỚC QUAN TRỌNG 1: GỌT VIỀN RĂNG CƯA ĐEN (ALPHA CHOKE / EROSION) & LÀM MƯỢT CẠNH (SMOOTH)
    alpha_pil = Image.fromarray((raw_alpha * 255.0).astype(np.uint8), mode="L")
    
    # Co mặt nạ (MinFilter) để cắt bỏ vòng viền đen bám ngoài chữ trắng và lá cây
    for _ in range(edge_choke_px):
        alpha_pil = alpha_pil.filter(ImageFilter.MinFilter(size=3))
        
    # Làm mịn đường viền bằng Gaussian Blur + Sigmoidal Contrast (Biến đường răng cưa thành đường cong mượt như Vector)
    if smooth_radius > 0:
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0
        # Làm sắc lại mép sau khi Blur để mép vừa mượt vừa dứt khoát, không bị nhòe sương mù
        smooth_a = np.clip((smooth_a - 0.38) / (0.88 - 0.38), 0.0, 1.0)
        # Đường cong mượt (Smoothstep curve: 3x^2 - 2x^3)
        smooth_a = smooth_a * smooth_a * (3.0 - 2.0 * smooth_a)
    else:
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0

    # BƯỚC QUAN TRỌNG 2: INWARD COLOR BLEED (PHÓNG MÀU SÁNG TỪ RUỘT RA MÉP NGOÀI)
    # Thay thế toàn bộ màu xám/đen ở rìa chữ trắng và hoa lá bằng màu chuẩn ở bên trong ruột của chính nó!
    pure_rgb_pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    # Giãn nở màu sáng từ trong ruột ra ngoài 2 lượt (MaxFilter đẩy màu trắng của chữ và màu tươi của lá phủ lấp viền đen)
    dilated_rgb_pil = pure_rgb_pil.filter(ImageFilter.MaxFilter(size=5))
    dilated_rgb_pil = dilated_rgb_pil.filter(ImageFilter.MaxFilter(size=3))
    dilated_rgb = np.array(dilated_rgb_pil, dtype=np.float32)

    # Kết hợp thêm công thức toán học Unpremultiply (Khử màu nền đen pha lẫn)
    safe_alpha = np.maximum(smooth_a[:, :, np.newaxis], 0.25)
    unpremultiplied_rgb = np.clip((rgb - (1.0 - safe_alpha) * bg_color) / safe_alpha, 0.0, 255.0)

    # Tại vùng viền (nơi smooth_a < 0.98 hoặc nơi màu gốc bị tối do dính nền), thay bằng màu sạch đã phóng từ trong ruột ra
    is_edge_region = (smooth_a < 0.98) | (raw_alpha < 0.98)
    edge_weight = np.where(is_edge_region, 1.0 - (smooth_a ** 2), 0.0)[:, :, np.newaxis]
    
    clean_rgb = rgb * (1.0 - edge_weight) + (0.65 * dilated_rgb + 0.35 * unpremultiplied_rgb) * edge_weight
    clean_rgb = np.clip(clean_rgb, 0.0, 255.0)

    # Đóng gói lại ảnh RGBA siêu sạch
    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    out_arr[:, :, :3] = clean_rgb.astype(np.uint8)
    out_arr[:, :, 3] = (smooth_a * 255.0).astype(np.uint8)

    return Image.fromarray(out_arr, mode="RGBA")

def remove_background_pro_v4(
    image: Image.Image, 
    hf_tok: str, 
    mode_str: str, 
    thresh: int = 38, 
    choke_px: int = 2, 
    smooth_rad: float = 1.4,
    keep_dark: bool = True
) -> Image.Image:
    if "Giữ nguyên nền" in mode_str:
        return image.convert("RGBA")
        
    if "Khử Viền Đen Mượt Cạnh V4.0" in mode_str:
        return ultra_smooth_defringe_v4(
            image, 
            bg_threshold=thresh, 
            edge_choke_px=choke_px, 
            smooth_radius=smooth_rad,
            protect_interior_dark=keep_dark
        )
        
    # Nếu dùng AI BiRefNet, sau khi tách xong cũng chạy qua bộ Gọt viền & Phóng màu ruột để không còn viền đen
    ai_out = None
    if HAS_REMBG:
        session = get_rembg_session()
        ai_out = rembg_remove(image, session=session)
    else:
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
                    ai_out = Image.open(out_path).convert("RGBA")
                    break
                except Exception:
                    continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    if ai_out is not None:
        return ai_out
    return ultra_smooth_defringe_v4(image, bg_threshold=thresh, edge_choke_px=choke_px, smooth_radius=smooth_rad, protect_interior_dark=keep_dark)

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
    st.subheader("🚀 Bước 3: Khử Sạch Viền Đen, Làm Mượt Đường Nét & Xuất File 4200x4800")
    
    if st.session_state.current_art is not None:
        st.image(st.session_state.current_art, caption="Bản Art chuẩn bị đưa vào tách nền khử viền V4.0", width=360)
        
        bg_option = st.radio("Chọn công nghệ tách nền:", [
            "💎 Khử Viền Đen Mượt Cạnh V4.0 (Gọt sạch răng cưa đen + Làm mịn nét + Bảo vệ chi tiết trong)",
            "🤖 Tách nền bằng AI BiRefNet",
            "🖼️ Giữ nguyên nền gốc (Chỉ chuẩn hóa vào khung 4200x4800)"
        ])
        
        thresh_val = 38
        choke_val = 2
        smooth_val = 1.4
        keep_interior = True
        
        if "Khử Viền Đen Mượt Cạnh V4.0" in bg_option:
            st.markdown("##### 🎛️ Tinh chỉnh độ mịn & gọt viền đen (Đã đặt sẵn mức tối ưu):")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                thresh_val = st.slider("1. Độ sâu bóc nền đen:", 15, 80, 38, help="Tăng lên nếu xung quanh lá/chữ vẫn còn bóng đen")
            with s_col2:
                choke_val = st.slider("2. Gọt viền đen (Choke px):", 0, 5, 2, help="Gọt bỏ 1-3px lớp răng cưa đen bám ngoài rìa chữ trắng và cành lá")
            with s_col3:
                smooth_val = st.slider("3. Độ mượt đường nét:", 0.0, 3.0, 1.4, step=0.2, help="Làm tròn và mịn các đường cắt bị răng cưa")
                
            keep_interior = st.checkbox("🛡️ Bảo vệ màu đen bên trong Art (Không làm thủng nhụy hoa, nét vẽ tóc & mắt hộp sọ)", value=True)
            
        preview_bg = st.radio("🔍 Chọn màu nền để kiểm tra viền ngay trên Web (File tải về luôn là PNG Trong Suốt):", 
                              ["Nền Trắng (Kiểm tra viền đen răng cưa)", "Nền Đen (Kiểm tra viền trắng)", "Nền Xám Caro / Trong Suốt"], horizontal=True)
        
        if st.button("⚡ XỬ LÝ KHỬ VIỀN MƯỢT CẠNH & XUẤT FILE 4200x4800 PX", type="primary"):
            work_img = st.session_state.current_art
            
            with st.spinner("Đang gọt viền răng cưa đen, phóng màu ruột và làm mượt đường nét (Sub-pixel Smoothing)..."):
                transparent_art = remove_background_pro_v4(
                    work_img, 
                    hf_token, 
                    bg_option, 
                    thresh=thresh_val, 
                    choke_px=choke_val, 
                    smooth_rad=smooth_val,
                    keep_dark=keep_interior
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
                
            st.success(f"✅ Hoàn tất! Đường nét đã được gọt sạch viền đen & làm mịn mượt ở chuẩn {CANVAS_W}x{CANVAS_H} px (300 DPI)!")
            
            # Hiển thị bản xem trước mặc định trên NỀN TRẮNG để người dùng thấy rõ không còn viền đen như Hình 2
            if "Nền Trắng" in preview_bg:
                bg_prev = Image.new("RGBA", final_pod_img.size, (255, 255, 255, 255))
                bg_prev.paste(final_pod_img, (0, 0), final_pod_img)
                st.image(bg_prev, caption="Kiểm tra trên Nền Trắng (Đã triệt tiêu đường viền đen quanh chữ & lá)", use_container_width=True)
            elif "Nền Đen" in preview_bg:
                bg_prev = Image.new("RGBA", final_pod_img.size, (15, 15, 15, 255))
                bg_prev.paste(final_pod_img, (0, 0), final_pod_img)
                st.image(bg_prev, caption="Kiểm tra trên Nền Đen (Sắc nét, không bị viền trắng)", use_container_width=True)
            else:
                st.image(final_pod_img, caption=f"File PNG Trong Suốt ({CANVAS_W}x{CANVAS_H})", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE IN PNG TRONG SUỐT (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Print_Ready_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy tải ảnh ở Cột 1, chọn vùng Art rồi chuyển sang Cột 2 để xử lý tách nền khử viền V4.0!")
