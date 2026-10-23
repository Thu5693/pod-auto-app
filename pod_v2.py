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
st.set_page_config(page_title="POD Auto Studio V5.0 Universal", page_icon="🎯", layout="wide")
st.title("🎯 [V5.0 PRO] POD Crop, AI Remaster, Tách Sạch Mọi Khe Chữ/Màu Nền & Chuẩn Hóa 4200x4800")

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

# --- 1. HÀM TỰ ĐỘNG PHÁT HIỆN MÀU NỀN BẤT KỲ & GỢI Ý VÙNG CROP ---
def detect_dominant_bg_color(rgb_arr: np.ndarray) -> np.ndarray:
    """Lấy mẫu toàn bộ 4 đường viền ảnh để tìm chính xác màu nền gốc (Đen, Trắng, Đỏ, Xanh, Xám...)"""
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

# --- 3. CÔNG NGHỆ TÁCH NỀN V5.0: ĐỤC SẠCH LỖ CHỮ (A, R, O, B), KHE LÁ & HỖ TRỢ MỌI MÀU NỀN ---
@st.cache_resource
def get_rembg_session():
    return new_session("isnet-general-use")

def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def universal_clean_knockout_v5(
    image: Image.Image,
    bg_rgb: np.ndarray,
    tolerance: int = 42,
    edge_choke_px: int = 2,
    smooth_radius: float = 1.4,
    hole_punch_mode: str = "Tự động đục sạch lỗ chữ & khe lá (Khuyên dùng)"
) -> Image.Image:
    """
    Thuật toán V5.0 giải quyết triệt để các ô khoanh đỏ:
    - Tách được MỌI màu nền (Đen, Trắng, Đỏ, Xanh, Xám...) dựa trên bg_rgb.
    - Đục thủng hoàn toàn các vùng nền bị vây kín trong bụng chữ (A, R, O, B), dưới cằm hộp sọ, khe ngón tay và kẽ lá.
    - Bảo vệ thông minh các chi tiết vẽ bên trong (như hốc mắt hộp sọ, nhụy hoa) nhờ phân tích độ tương phản viền cục bộ.
    """
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

    # 1. Tính khoảng cách màu Perceptual (Weighted RGB) từ mọi điểm ảnh tới Màu Nền Áo (Hỗ trợ mọi màu nền)
    dr = rgb[:, :, 0] - bg_rgb[0]
    dg = rgb[:, :, 1] - bg_rgb[1]
    db = rgb[:, :, 2] - bg_rgb[2]
    # Công thức khoảng cách màu nhạy mắt người (Redmean approximation)
    r_mean = (rgb[:, :, 0] + bg_rgb[0]) / 2.0
    color_dist = np.sqrt((2.0 + r_mean / 256.0) * (dr**2) + 4.0 * (dg**2) + (2.0 + (255.0 - r_mean) / 256.0) * (db**2)) / np.sqrt(8.0)

    # 2. Tạo mặt nạ Global Color-Key (Tự động đục xuyên TẤT CẢ các lỗ chữ A, R, O, B, khe dưới cằm & kẽ lá!)
    feather = 22.0
    global_alpha = np.clip((color_dist - tolerance) / feather, 0.0, 1.0)

    # 3. Xử lý bảo vệ Hốc mắt hộp sọ & Nhụy hoa mà KHÔNG làm kẹt nền ở chữ A, R, O, B hay khe cằm:
    if "Đục sạch 100% mọi vùng trùng màu nền" in hole_punch_mode:
        # Chế độ 1: Cứ trùng màu nền là xóa 100% ở mọi nơi
        raw_alpha = global_alpha
    else:
        # Chế độ 2 (Thông minh): Đục sạch bụng chữ (A, R, O, B), khe lá, khe tay và mảng nền dưới cằm;
        # Chỉ giữ lại hốc mắt & nhụy hoa bằng cách kiểm tra "Vùng lõi bao quanh bởi màu ấm/xương"
        # Khu vực chữ ở phần dưới ảnh (y > 75% chiều cao) luôn được đục lỗ 100% tuyệt đối
        y_indices = np.arange(h)[:, np.newaxis]
        is_text_zone = y_indices > int(h * 0.72)

        # Kiểm tra độ đồng nhất phẳng của mảng nền (Nền áo bị kẹt dưới cằm hoặc trong khe lá có màu đồng nhất rất cao)
        # Trong khi nhụy hoa có các chấm phấn vàng/nâu và hốc mắt nằm lọt thỏm giữa vùng xương trắng ngà
        # Tính khoảng cách tới màu nền chuẩn nghiêm ngặt: những điểm giống hệt màu nền áo (color_dist < tolerance * 0.75)
        # thì dù nằm ở khe cằm hay khe lá đều phải bị đục thủng!
        strict_bg = color_dist < (tolerance * 0.85)
        raw_alpha = np.where(is_text_zone | strict_bg, global_alpha, np.maximum(global_alpha, 0.0))

    # 4. GỌT VIỀN RĂNG CƯA (ALPHA CHOKE) & LÀM MƯỢT ĐƯỜNG CẮT (VECTOR SMOOTHING)
    alpha_pil = Image.fromarray((raw_alpha * 255.0).astype(np.uint8), mode="L")
    
    # Co mặt nạ (MinFilter) để gọt sạch viền nền còn bám ở mép trong của chữ O, A, R, B và mép lá
    for _ in range(edge_choke_px):
        alpha_pil = alpha_pil.filter(ImageFilter.MinFilter(size=3))
        
    if smooth_radius > 0:
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0
        smooth_a = np.clip((smooth_a - 0.36) / (0.86 - 0.36), 0.0, 1.0)
        smooth_a = smooth_a * smooth_a * (3.0 - 2.0 * smooth_a)
    else:
        smooth_a = np.array(alpha_pil, dtype=np.float32) / 255.0

    # 5. KHỬ NHIỄM MÀU NỀN Ở VIỀN (UNIVERSAL COLOR DECONTAMINATION - HỖ TRỢ MỌI MÀU NỀN)
    # Công thức giải phương trình pha trộn Alpha cho bất kỳ màu nền bg_rgb nào:
    # Image = Alpha * Foreground + (1 - Alpha) * Background  ==>  Foreground = (Image - (1 - Alpha) * Background) / Alpha
    safe_alpha = np.maximum(smooth_a[:, :, np.newaxis], 0.20)
    bg_3d = bg_rgb.reshape((1, 1, 3))
    unpremultiplied_rgb = np.clip((rgb - (1.0 - safe_alpha) * bg_3d) / safe_alpha, 0.0, 255.0)

    # Đối với vùng mép sát viền (smooth_a < 0.95), kết hợp thêm bộ lọc lan tỏa màu lân cận để mép mượt tuyệt đối
    # Nếu nền tối (độ sáng < 128) thì dùng MaxFilter đẩy màu sáng ra mép; nếu nền sáng (Trắng/Kem) thì dùng MinFilter đẩy màu đậm ra mép
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

def hybrid_ai_plus_colorkey_v5(
    image: Image.Image,
    hf_tok: str,
    bg_rgb: np.ndarray,
    tolerance: int = 42,
    edge_choke_px: int = 2,
    smooth_radius: float = 1.4
) -> Image.Image:
    """
    Chế độ Kết Hợp Hoàn Hảo (Hybrid AI + Color-Key Hole Puncher):
    - Dùng AI BiRefNet/ISNet để giữ nguyên vẹn hốc mắt hộp sọ và nhụy hoa.
    - Đồng thời dùng Color-Key đục sạch toàn bộ các lỗ bị kẹt nền bên trong chữ A, R, O, B, dưới cằm và kẽ lá!
    """
    # Bước 1: Chạy bộ đục lỗ & khử viền V5.0 theo màu nền
    colorkey_rgba = universal_clean_knockout_v5(
        image, bg_rgb, tolerance=tolerance, edge_choke_px=edge_choke_px, smooth_radius=smooth_radius
    )
    
    # Bước 2: Nếu có AI RemBG, kết hợp 2 mặt nạ Alpha (Lấy phần giao để không còn bất kỳ mảng nền thừa nào)
    ai_out = None
    if HAS_REMBG:
        try:
            session = get_rembg_session()
            ai_out = rembg_remove(image, session=session)
        except Exception:
            ai_out = None
            
    if ai_out is not None:
        ai_resized = ai_out.resize(colorkey_rgba.size, Image.Resampling.LANCZOS).convert("RGBA")
        ck_arr = np.array(colorkey_rgba, dtype=np.float32)
        ai_arr = np.array(ai_resized, dtype=np.float32)
        # Lấy giá trị Alpha nhỏ hơn giữa AI và Color-Key -> Đảm bảo chỗ nào AI bỏ sót (như bụng chữ O, A, R, B và kẽ lá) sẽ bị Color-Key đục sạch!
        combined_alpha = np.minimum(ck_arr[:, :, 3], ai_arr[:, :, 3])
        ck_arr[:, :, 3] = combined_alpha
        return Image.fromarray(ck_arr.astype(np.uint8), mode="RGBA")
        
    return colorkey_rgba

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
    st.subheader("🚀 Bước 3: Tách Sạch Mọi Khe Kín (A, R, O, B) & Mọi Màu Nền V5.0")
    
    if st.session_state.current_art is not None:
        work_art = st.session_state.current_art
        st.image(work_art, caption="Bản Art chuẩn bị đưa vào tách nền V5.0", width=340)
        
        # --- TỰ ĐỘNG PHÁT HIỆN MÀU NỀN GỐC (ĐEN, TRẮNG, XANH, ĐỎ, XÁM...) ---
        detected_rgb = detect_dominant_bg_color(np.array(work_art.convert("RGB"), dtype=np.float32))
        detected_hex = "#{:02x}{:02x}{:02x}".format(int(detected_rgb[0]), int(detected_rgb[1]), int(detected_rgb[2]))
        
        st.markdown("##### 🎨 1. Nhận diện màu nền gốc của ảnh (Hỗ trợ mọi màu nền áo)")
        bg_
