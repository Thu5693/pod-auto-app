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
st.set_page_config(page_title="POD Auto Studio V5.6", page_icon="🎯", layout="wide")
st.title("🎯 [V5.6 PRO] POD Crop, Phục Hồi Góc Chữ, Chuẩn Hóa Warp (Arc Upper) & Tách Nền 4200x4800")

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

# --- 2. BỘ XỬ LÝ HÌNH HỌC: XÓA RÁC MÉP MOCKUP, MỞ RỘNG GÓC CHỮ & NẮN WARP ARC UPPER ---
def clean_mockup_border_artifacts(img: Image.Image, bg_rgb: np.ndarray, strip_pct: float = 0.035) -> Image.Image:
    """
    Loại bỏ các vệt rác lọt vào sát mép ảnh mockup (như lọn tóc người mẫu ở sát mép trái chữ M
    hoặc nếp gấp áo trắng sát mép phải chữ D) bằng cách lọc dải biên ngoài cùng.
    """
    if strip_pct <= 0:
        return img
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    margin_w = int(w * strip_pct)
    
    # Tại vùng dải biên trái/phải (3.5% chiều rộng), những điểm có tông màu da/tóc nâu vàng hoặc vệt xám mờ
    # không mang màu xanh lá/trắng sáng của viền chữ sẽ được tô đồng nhất về màu nền
    for x_range in [range(0, margin_w), range(w - margin_w, w)]:
        sub = arr[:, x_range, :]
        r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        # Giữ lại những điểm ảnh thuộc chữ (màu xanh lá rực G > R + 15 hoặc viền trắng sáng R,G,B > 200)
        is_text_pixel = (g > r + 15) | ((r > 195) & (g > 195) & (b > 195))
        for c in range(3):
            sub[:, :, c] = np.where(is_text_pixel, sub[:, :, c], bg_rgb[c])
        arr[:, x_range, :] = sub
        
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

def apply_text_warp_correction(
    img: Image.Image, 
    bg_rgb: np.ndarray, 
    bottom_arc_upper_strength: float = 0.22,
    top_straighten_strength: float = 0.08
) -> Image.Image:
    """
    Biến đổi lưới tọa độ (Mesh Displacement Warp) chuyên dụng cho POD:
    - Khắc phục hiện tượng chữ dưới ('MACDONALD') bị võng bụng áo, đẩy phần giữa vòm lên chuẩn 'Warp: Arc Upper'
      và nắn phẳng đường chân chữ.
    - Bù đắp độ vòm nhẹ ở góc trên để chữ 'MIKE' cân đối hoàn hảo trong khung chữ nhật.
    """
    if abs(bottom_arc_upper_strength) < 0.01 and abs(top_straighten_strength) < 0.01:
        return img

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    
    # Thêm khoảng đệm trên/dưới và trái/phải để khi kéo vòm không bị lẹm mất chữ
    pad_y = int(h * 0.05)
    pad_x = int(w * 0.03)
    padded = np.zeros((h + pad_y * 2, w + pad_x * 2, 3), dtype=np.float32)
    padded[:, :, :] = bg_rgb.reshape((1, 1, 3))
    padded[pad_y:pad_y + h, pad_x:pad_x + w, :] = arr
    
    ph, pw, _ = padded.shape
    out_arr = np.zeros_like(padded)
    out_arr[:, :, :] = bg_rgb.reshape((1, 1, 3))
    
    # Tính hệ số parabol theo chiều ngang: norm_x chạy từ -1 (mép trái) -> 0 (chính giữa) -> +1 (mép phải)
    x_coords = np.arange(pw, dtype=np.float32)
    norm_x = (x_coords - (pw / 2.0)) / (pw / 2.0)
    # Đường cong vòm (1 - x^2): đạt cực đại = 1 tại tâm (x=0) và = 0 tại 2 mép
    arch_profile = np.clip(1.0 - (norm_x ** 2), 0.0, 1.0)
    
    y_coords = np.arange(ph, dtype=np.float32)[:, np.newaxis]
    
    # 1. Vùng chữ dưới (từ 65% đến 100% chiều cao ảnh):
    # Trên áo mẫu, phần giữa chữ MACDONALD bị sụp xuống dưới. Để đẩy tâm chữ vòm lên chuẩn Arc Upper,
    # tại tâm (arch_profile=1), ta lấy điểm ảnh từ phía dưới kéo dịch lên trên (tức là cộng thêm delta_y vào tọa độ nguồn).
    bottom_zone_weight = np.clip((y_coords - ph * 0.62) / (ph * 0.32), 0.0, 1.0)
    # Hiệu ứng Arc Upper: kéo đỉnh vòm phần chữ dưới lên cao hơn ở chính giữa
    max_shift_bottom = ph * 0.085 * bottom_arc_upper_strength
    shift_bottom = bottom_zone_weight * (arch_profile[np.newaxis, :] * max_shift_bottom)
    
    # 2. Vùng chữ trên (từ 0% đến 32% chiều cao ảnh - chữ MIKE):
    # Cân bằng lại đường ngang của chữ MIKE nếu hai góc bị sụp
    top_zone_weight = np.clip((ph * 0.32 - y_coords) / (ph * 0.32), 0.0, 1.0)
    max_shift_top = ph * 0.05 * top_straighten_strength
    shift_top = -top_zone_weight * ((1.0 - arch_profile[np.newaxis, :]) * max_shift_top)
    
    total_src_y = y_coords + shift_bottom + shift_top
    
    # Nội suy tuyến tính (Bilinear Vertical Interpolation) để chữ sau khi uốn vòm mượt tuyệt đối không bị răng cưa
    y0 = np.floor(total_src_y).astype(np.int32)
    y1 = y0 + 1
    wy = (total_src_y - y0)[:, :, np.newaxis]
    
    y0_clip = np.clip(y0, 0, ph - 1)
    y1_clip = np.clip(y1, 0, ph - 1)
    x_idx = np.arange(pw, dtype=np.int32)[np.newaxis, :]
    
    warped = (1.0 - wy) * padded[y0_clip, x_idx] + wy * padded[y1_clip, x_idx]
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), mode="RGB")

# --- 3. HÀM ĐỒNG NHẤT MÀU NỀN, PHỤC HỒI CHỮ BỊ CẮT GÓC & LÀM NÉT ---
def unify_background_solid_color(
    image: Image.Image, 
    target_bg_rgb: np.ndarray, 
    bg_tolerance: int = 38
) -> Image.Image:
    rgb_arr = np.array(image.convert("RGB"), dtype=np.float32)
    orig_bg_rgb = detect_dominant_bg_color(rgb_arr)
    
    dr = rgb_arr[:, :, 0] - orig_bg_rgb[0]
    dg = rgb_arr[:, :, 1] - orig_bg_rgb[1]
    db = rgb_arr[:, :, 2] - orig_bg_rgb[2]
    r_mean = (rgb_arr[:, :, 0] + orig_bg_rgb[0]) / 2.0
    color_dist = np.sqrt((2.0 + r_mean / 256.0) * (dr**2) + 4.0 * (dg**2) + (2.0 + (255.0 - r_mean) / 256.0) * (db**2)) / np.sqrt(8.0)
    
    feather = 18.0
    art_mask = np.clip((color_dist - bg_tolerance) / feather, 0.0, 1.0)
    
    mask_pil = Image.fromarray((art_mask * 255.0).astype(np.uint8), mode="L")
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    smooth_mask = (np.array(mask_pil, dtype=np.float32) / 255.0)[:, :, np.newaxis]
    
    safe_mask = np.maximum(smooth_mask, 0.22)
    orig_bg_3d = orig_bg_rgb.reshape((1, 1, 3))
    target_bg_3d = target_bg_rgb.reshape((1, 1, 3))
    
    decontaminated_art = np.clip((rgb_arr - (1.0 - safe_mask) * orig_bg_3d) / safe_mask, 0.0, 255.0)
    is_edge = (smooth_mask > 0.02) & (smooth_mask < 0.95)
    clean_fg = np.where(is_edge, decontaminated_art, rgb_arr)
    
    unified_rgb = clean_fg * smooth_mask + target_bg_3d * (1.0 - smooth_mask)
    return Image.fromarray(np.clip(unified_rgb, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

def remaster_and_unify_art(
    cropped_img: Image.Image, 
    api_key: str, 
    hf_tok: str, 
    target_bg_rgb: np.ndarray,
    bg_tolerance: int = 38,
    arc_strength: float = 0.22,
    clean_border: bool = True,
    custom_edit: str = ""
) -> Image.Image:
    rgb_img = cropped_img.convert("RGB")
    orig_bg = detect_dominant_bg_color(np.array(rgb_img, dtype=np.float32))
    
    # Bước A: Xóa lọn tóc/rác mockup ở sát mép trái chữ M và mép phải chữ D
    if clean_border:
        rgb_img = clean_mockup_border_artifacts(rgb_img, orig_bg, strip_pct=0.04)
        
    # Bước B: Áp dụng nắn khung hình học Warp (Arc Upper cho chữ dưới & cân đối chữ trên)
    rgb_img = apply_text_warp_correction(rgb_img, orig_bg, bottom_arc_upper_strength=arc_strength, top_straighten_strength=0.1)

    target_hex = rgb_to_hex(target_bg_rgb)
    clean_key = api_key.strip().replace('"', '').replace("'", "") if api_key else ""
    enhanced_img = None
    
    # Bước C: Tự động tìm đúng Model tạo/chỉnh sửa ảnh đang mở trên tài khoản Gemini của người dùng
    if clean_key:
        client = genai.Client(api_key=clean_key)
        edit_instruction = f" Also apply this specific adjustment: {custom_edit}." if custom_edit.strip() else ""
        
        remaster_prompt = (
            f"Remaster, complete, and upscale this t-shirt graphic into an ultra-crisp, high-resolution, print-ready 2D graphic "
            f"isolated on a completely flat, solid uniform background of exact hex color {target_hex}. "
            "CRITICAL INSTRUCTIONS: "
            "1. COMPLETE CLIPPED LETTERS: If any letter at the corners/edges is slightly cut off or clipped by the crop (such as the top-left corner of the letter 'M' in 'MIKE' or the edge of 'D' in 'MACDONALD'), reconstruct and complete the missing stroke/serif seamlessly so every letter is 100% whole and intact. "
            "2. TEXT WARP & ALIGNMENT: Ensure the top text ('MIKE') is straight and bold, and the bottom text ('MACDONALD') has a clean, symmetrical 'Arc Upper' warp (flat horizontal bottom baseline, arched top contour). "
            "3. REMOVE ARTIFACTS: Remove any stray hair or mockup fabric at the left/right edges. Preserve the exact faces, clothing, colors, and lighting effects."
            f"{edit_instruction}"
        )
        
        # Tự động quét các model hỗ trợ sinh ảnh trong tài khoản
        candidate_img_models = ["gemini-2.5-flash-image-preview", "gemini-2.0-flash-exp-image-generation", "gemini-3.8-flash"]
        try:
            for m in client.models.list():
                m_name = m.name.replace("models/", "")
                if ("image" in m_name or "flash" in m_name) and m_name not in candidate_img_models:
                    candidate_img_models.insert(0, m_name)
        except Exception:
            pass

        for m_name in candidate_img_models:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=[rgb_img, remaster_prompt],
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        enhanced_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        st.success(f"✨ Đã phục hồi góc chữ bị cắt & căn chuẩn Warp bằng AI ({m_name})!")
                        break
                if enhanced_img is not None:
                    break
            except Exception:
                continue

    # Bước D: Nếu tài khoản Gemini chỉ hỗ trợ đọc chữ (không mở quyền xuất ảnh), dùng GPU Real-ESRGAN làm nét 4K bức ảnh đã được nắn Warp
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
                    st.success(f"✨ Đã nắn chuẩn Warp Arc Upper, xóa rác mép & Upscale 4K qua GPU {sp}!")
                    break
                except Exception:
                    continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if enhanced_img is None:
        w, h = rgb_img.size
        scale = max(1, 2000 // max(w, h))
        up_img = rgb_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        enhanced_img = up_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=4)).convert("RGBA")

    final_step2_img = unify_background_solid_color(enhanced_img, target_bg_rgb, bg_tolerance=bg_tolerance)
    return final_step2_img

# --- 4. HÀM CHUẨN HÓA KHUNG 4200x4800 CÓ MÀU NỀN ĐỒNG NHẤT ---
def standardize_solid_bg_canvas(
    img: Image.Image, 
    bg_rgb: np.ndarray, 
    target_w=4200, 
    target_h=4800, 
    pad=150, 
    top_chest=True,
    tol=35
) -> Image.Image:
    rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
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

# --- 5. CÔNG NGHỆ TÁCH NỀN TRONG SUỐT V5.6 ---
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

# --- 6. HÀM ĐỌC PROMPT & REDESIGN MẪU MỚI ---
def analyze_art_prompt(image: Image.Image, api_key: str) -> str:
    rgb_img = image.convert("RGB")
    rgb_img.thumbnail((1024, 1024))
    sys_prompt = (
        "Analyze this cropped T-shirt graphic design carefully. Write ONLY the raw English image generation prompt "
        "to recreate this exact illustration style, subject, typography warp (such as top straight bold text and bottom Arc Upper warped text), "
        "color palette, and layout isolated on a solid flat background, crisp vector-like edges, print-ready graphic art."
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
    return "Vintage bootleg rap tee graphic design, complete unclipped bold typography at top and Arc Upper warped typography at bottom, central character collage, neon green lighting effects, isolated on solid background"

def generate_redesign_flux(prompt: str, hf_tok: str) -> Image.Image:
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok.strip().replace('"', '').replace("'", "")
    full_prompt = f"{prompt}, isolated t-shirt graphic art, complete unclipped letters with generous margin, solid plain background, crisp clean edges"
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
        st.subheader("✨ Bước 2: Phục Hồi Góc Chữ, Căn Chuẩn Warp (Arc Upper) & Đồng Nhất Nền")
        
        orig_crop_bg = detect_dominant_bg_color(np.array(cropped_art.convert("RGB"), dtype=np.float32))
        orig_crop_hex = rgb_to_hex(orig_crop_bg)
        
        st.markdown("##### 🎨 1. Tùy chọn màu nền đồng nhất cho Bước 2:")
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
        
        st.markdown("##### 📐 2. Căn chỉnh Text Warp (Arc Upper) & Xóa rác mép Mockup:")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            arc_val = st.slider(
                "Độ cong vòm chữ dưới (Warp - Arc Upper):", 
                -0.30, 0.60, 0.24, step=0.02,
                help="Khắc phục chữ dưới (như MACDONALD) bị võng bụng áo, đẩy vòm chữ cong lên chuẩn Arc Upper và làm phẳng đáy chữ"
            )
        with w_col2:
            step2_tol = st.slider(
                "Độ nhạy nhận diện nền áo cũ:", 
                15, 80, 38, 
                help="Làm phẳng phần nền vải nhăn về đúng 1 mã màu đồng nhất"
            )
            
        clean_mockup_edge = st.checkbox(
            "🧹 Tự động lọc bỏ tóc/vệt vải thừa của Mockup dính ở sát mép trái/phải (Ví dụ: lọn tóc cạnh chữ M)", 
            value=True
        )
        
        custom_tweak = st.text_input(
            "Tùy chọn chỉnh sửa thêm cho AI (Để trống nếu giữ nguyên, hoặc gõ yêu cầu cụ thể):",
            value=""
        )
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🪄 1. AI Phục Hồi Góc Chữ + Căn Chuẩn Warp & Xuất 4200x4800", type="primary"):
                with st.spinner(f"Đang phục hồi góc chữ, nắn chuẩn Warp Arc Upper, đồng nhất nền {chosen_step2_hex} và xuất 4200x4800px..."):
                    remastered = remaster_and_unify_art(
                        cropped_art, gemini_key, hf_token, 
                        target_bg_rgb=step2_bg_rgb, bg_tolerance=step2_tol, 
                        arc_strength=arc_val, clean_border=clean_mockup_edge, custom_edit=custom_tweak
                    )
                    st.session_state.current_art = remastered
                    st.session_state.step2_bg_hex = chosen_step2_hex
                    
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
            if st.button("➡️ 2. Căn Chuẩn Warp + Đồng Nhất Nền Nhanh (0.5 giây)"):
                with st.spinner(f"Đang nắn Warp Arc Upper, xóa rác mép và xuất khung 4200x4800px..."):
                    work_fast = cropped_art.convert("RGB")
                    if clean_mockup_edge:
                        work_fast = clean_mockup_border_artifacts(work_fast, orig_crop_bg, strip_pct=0.04)
                    work_fast = apply_text_warp_correction(work_fast, orig_crop_bg, bottom_arc_upper_strength=arc_val, top_straighten_strength=0.1)
                    unified_fast = unify_background_solid_color(work_fast, step2_bg_rgb, bg_tolerance=step2_tol)
                    
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

        if st.session_state.step2_solid_4200_bytes is not None and st.session_state.step2_preview_img is not None:
            st.success(f"✅ Hoàn tất Bước 2! Đã căn chuẩn Warp, xử lý góc chữ và đưa về nền đồng nhất ({st.session_state.step2_bg_hex}) chuẩn {CANVAS_W}x{CANVAS_H} px!")
            st.image(
                st.session_state.step2_preview_img, 
                caption=f"Kết quả Bước 2: Chưa tách nền - Nền đồng nhất {st.session_state.step2_bg_hex} ({CANVAS_W}x{CANVAS_H} px)", 
                use_container_width=True
            )
            st.download_button(
                label=f"📥 TẢI FILE BƯỚC 2 - CÓ NỀN ĐỒNG NHẤT {st.session_state.step2_bg_hex} ({CANVAS_W}x{CANVAS_H} - 300 DPI)",
                data=st.session_state.step2_solid_4200_bytes,
                file_name=f"POD_Step2_SolidBG_{st.session_state.step2_bg_hex.lstrip('#')}_4200x4800.png",
                mime="image/png"
            )
                
        with st.expander("🎨 Mở rộng: Clone & Vẽ lại mẫu mới hoàn toàn bằng Prompt (FLUX AI)"):
            if st.button("🔍 Đọc Prompt từ vùng đã Crop"):
                st.session_state.extracted_prompt = analyze_art_prompt(cropped_art, gemini_key)
            prompt_text = st.text_area("Prompt Redesign:", value=st.session_state.extracted_prompt, height=100)
            if st.button("🎨 Vẽ mẫu mới bằng FLUX AI"):
                with st.spinner("Đang vẽ mẫu mới..."):
                    st.session_state.current_art = generate_redesign_flux(prompt_text, hf_token)
                    st.success("🎉 Đã tạo xong mẫu Redesign bằng FLUX AI! Hãy chuyển sang Bước 3 bên phải để xử lý tiếp.")

with col2:
    st.subheader("🚀 Bước 3: Tách Nền Trong Suốt Sạch Khe Kín (A, R, O, B) & Xuất 4200x4800")
    
    if st.session_state.current_art is not None:
        work_art = st.session_state.current_art
        st.image(work_art, caption="Bản Art chuyển sang để tách nền trong suốt", width=320)
        
        detected_rgb = hex_to_rgb(st.session_state.step2_bg_hex)
        detected_hex = st.session_state.step2_bg_hex
        
        st.markdown("##### 🎨 1. Màu nền cần tách sang trong suốt")
        bg_col1, bg_col2 = st.columns([2, 1])
        with bg_col1:
            bg_detect_mode = st.radio(
                "Chế độ nhận diện màu nền Bước 3:",
                [
                    f"🤖 Tự động nhận diện nền đồng nhất từ Bước 2 ({detected_hex})",
                    "🎯 Tự chấm/chọn màu nền thủ công"
                ]
            )
        with bg_col2:
            picked_hex = st.color_picker("Màu nền sẽ xóa:", value=detected_hex)
            
        target_bg_rgb = detected_rgb if "Tự động" in bg_detect_mode else hex_to_rgb(picked_hex)

        st.markdown("##### 🎛️ 2. Tinh chỉnh độ sâu đục khe kín (A, R, O, B) & độ mượt đường nét:")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            thresh_val = st.slider("Độ nhạy bóc nền & đục khe:", 15, 90, 42)
        with s_col2:
            choke_val = st.slider("Gọt viền (Choke px):", 0, 5, 2)
        with s_col3:
            smooth_val = st.slider("Độ mượt đường nét:", 0.0, 3.0, 1.4, step=0.2)
            
        preview_bg = st.radio(
            "🔍 Màu nền kiểm tra kết quả Bước 3 (File tải về luôn là PNG Trong Suốt):", 
            ["Nền Trắng (Kiểm tra nền tối)", "Nền Đen (Kiểm tra nền sáng)", "Nền Xanh Dương (Kiểm tra mọi loại viền)"],
            horizontal=True
        )
        
        if st.button("⚡ XỬ LÝ TÁCH NỀN TRONG SUỐT & XUẤT FILE 4200x4800 PX", type="primary"):
            with st.spinner("Đang tách nền trong suốt, đục sạch bụng chữ A, R, O, B, các khe lá và làm mượt đường nét..."):
                transparent_art = universal_clean_knockout_v5(
                    work_art, target_bg_rgb, 
                    tolerance=thresh_val, edge_choke_px=choke_val, smooth_radius=smooth_val
                )
                
            with st.spinner("Đang căn chỉnh chuẩn xác vào khung trong suốt 4200x4800 (300 DPI)..."):
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
                
            st.success(f"✅ Hoàn tất Bước 3! File PNG Trong Suốt đạt chuẩn {CANVAS_W}x{CANVAS_H} px (300 DPI)!")
            
            preview_small = final_pod_img.copy()
            preview_small.thumbnail((800, 914))
            
            if "Nền Trắng" in preview_bg:
                bg_prev_img = Image.new("RGBA", preview_small.size, (255, 255, 255, 255))
            elif "Nền Đen" in preview_bg:
                bg_prev_img = Image.new("RGBA", preview_small.size, (15, 15, 15, 255))
            else:
                bg_prev_img = Image.new("RGBA", preview_small.size, (0, 120, 215, 255))
                
            bg_prev_img.paste(preview_small, (0, 0), preview_small)
            st.image(bg_prev_img, caption=f"Kiểm tra kết quả Bước 3 trên {preview_bg} (File tải về bên dưới là PNG Trong Suốt 4200x4800px)", use_container_width=True)
            
            st.download_button(
                label="📥 TẢI FILE BƯỚC 2/3 - PNG TRONG SUỐT ĐÃ TÁCH NỀN (4200x4800 - 300 DPI)",
                data=byte_im,
                file_name="POD_Step3_Transparent_4200x4800.png",
                mime="image/png"
            )
    else:
        st.info("👈 Hãy thực hiện Bước 1 & Bước 2 ở Cột bên trái trước, ảnh đã đồng nhất nền sẽ tự động chuyển sang Bước 3 tại đây!")
