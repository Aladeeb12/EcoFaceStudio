import os
import mediapipe as mp

# كود لإجبار ميديا بايب على معرفة مكان ملفاته رغم المسار العربي
mp_path = os.path.join(os.path.dirname(mp.__file__), 'modules')
os.environ['MEDIAPIPE_BINARY_GRAPH_PATH'] = mp_path

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os
import math
import io

# --- 1. إعدادات الموديلات (الطريقة الأضمن للإصدار 0.10.11) ---
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation # تصحيح الاسم هنا
mp_drawing = mp.solutions.drawing_utils

# --- 2. إعدادات الصفحة والتصميم (Nature UI) ---
st.set_page_config(page_title="Eco AI Face Studio", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1b3022 0%, #2c5d33 100%);
        color: #f0f7f4;
    }
    section[data-testid="stSidebar"] {
        background-color: #242b23 !important;
    }
    h1, h2, h3 { color: #d4e4bc !important; }
    .stButton>button {
        border-radius: 30px !important;
        border: 2px solid #d4e4bc !important;
        background-color: transparent !important;
        color: #d4e4bc !important;
    }
    </style>
    """, unsafe_allow_html=True)

if 'run_live' not in st.session_state:
    st.session_state.run_live = False

# --- 3. تحميل الموديلات مع التخزين المؤقت ---
@st.cache_resource
def get_models():
    # استخدام المتغيرات التي عرفناها في الأعلى
    mesh_mod = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        min_detection_confidence=0.6, 
        min_tracking_confidence=0.6
    )
    seg_mod = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    return mesh_mod, seg_mod

mesh_mod, seg_mod = get_models()

# --- 4. وظائف المعالجة ---

def draw_clown_nose(img, lms, size_factor, offset_x, offset_y):
    h, w, _ = img.shape
    cx, cy = int(lms[4].x * w) + offset_x, int(lms[4].y * h) + offset_y
    radius = int(abs(lms[263].x - lms[33].x) * w * 0.15 * size_factor)
    cv2.circle(img, (cx, cy), radius, (0, 0, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (cx - radius//3, cy - radius//3), radius//4, (255, 255, 255), -1, cv2.LINE_AA)
    return img

def enlarge_eyes(img, lms, intensity):
    h, w, _ = img.shape
    eye_centers = [468, 473]
    out = img.copy()
    radius = int(w * 0.05)
    strength = intensity * 0.3 
    for idx in eye_centers:
        cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w, cx + radius), min(h, cy + radius)
        if x2 <= x1 or y2 <= y1: continue
        roi = img[y1:y2, x1:x2].copy()
        rows, cols = roi.shape[:2]
        mid_x, mid_y = cols / 2, rows / 2
        map_x, map_y = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                dx, dy = j - mid_x, i - mid_y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < radius:
                    factor = math.pow(distance / radius, strength) if distance > 0 else 1
                    map_x[i, j], map_y[i, j] = mid_x + dx * factor, mid_y + dy * factor
                else:
                    map_x[i, j], map_y[i, j] = j, i
        eye_zoomed = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = eye_zoomed
    return out

def apply_makeup_realistic(img, lms, points, color, alpha):
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(lms[p].x * w), int(lms[p].y * h)] for p in points])
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha_mask = (mask / 255.0) * alpha
    out = img.copy()
    for c in range(3):
        out[:, :, c] = (img[:, :, c] * (1 - alpha_mask) + color[c] * alpha_mask).astype(np.uint8)
    return out

def overlay_acc(bg, overlay, x, y, w, h, angle=0):
    try:
        if overlay.shape[2] < 4: return bg # تخطي إذا لم تكن الصورة PNG شفافة
        overlay = cv2.resize(overlay, (int(w), int(h)))
        # ... (بقية كود الدمج كما هو)
        y1, y2 = max(0, y), min(bg.shape[0], y + h)
        x1, x2 = max(0, x), min(bg.shape[1], x + w)
        img_part = overlay[0:y2-y1, 0:x2-x1]
        alpha_s = img_part[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (alpha_s * img_part[:, :, c] + (1.0 - alpha_s) * bg[y1:y2, x1:x2, c])
    except: pass
    return bg

# --- 5. واجهة المستخدم الجانبية ---
st.sidebar.markdown("<h1 style='text-align: center;'>🍃 Eco Studio</h1>", unsafe_allow_html=True)

def accessory_controls(label):
    with st.sidebar.expander(f"🌱 ضبط: {label}"):
        sc = st.slider(f"الحجم", 0.1, 5.0, 1.0, key=f"s_{label}")
        rot = st.slider(f"الدوران (°)", -180, 180, 0, key=f"r_{label}")
        off_x = st.slider(f"تحريك X", -500, 500, 0, key=f"x_{label}")
        off_y = st.slider(f"تحريك Y", -500, 500, 0, key=f"y_{label}")
    return sc, rot, off_x, off_y

st.sidebar.subheader("🌻 زينة الوجه")
f_clown_nose = st.sidebar.checkbox("أنف المهرج")
c_clown = accessory_controls("الأنف") if f_clown_nose else (1.0, 0, 0, 0)

f_glasses = st.sidebar.checkbox("النظارات الشمسية")
c_glasses = accessory_controls("النظارة") if f_glasses else (1.0, 0, 0, 0)

# (بقية الخيارات تتبع نفس النمط...)
f_mustache = st.sidebar.checkbox("الشارب")
c_mustache = accessory_controls("الشارب") if f_mustache else (1.0, 0, 0, 0)

st.sidebar.subheader("✨ لمسات الجمال")
f_lipstick = st.sidebar.checkbox("أحمر الشفاه")
f_big_eyes = st.sidebar.checkbox("توسيع العيون")
f_smooth = st.sidebar.checkbox("تنعيم البشرة")
intensity = st.sidebar.slider("قوة التأثير", 0.0, 1.0, 0.5)

f_blur = st.sidebar.checkbox("ضباب الغابة (العزل)")

# --- 6. محرك المعالجة الشامل ---
def process_frame(img_bgr):
    h, w, _ = img_bgr.shape
    out_bgr = img_bgr.copy()
    
    # 1. العزل (Background Blur)
    if f_blur:
        res_seg = seg_mod.process(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        if res_seg.segmentation_mask is not None:
            mask = res_seg.segmentation_mask > 0.1
            bg_blur = cv2.GaussianBlur(out_bgr, (65, 65), 0)
            out_bgr = np.where(np.stack((mask,)*3, axis=-1), out_bgr, bg_blur)

    # 2. تتبع الوجه
    res_mesh = mesh_mod.process(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    if res_mesh.multi_face_landmarks:
        lms = res_mesh.multi_face_landmarks[0].landmark

        if f_smooth: out_bgr = cv2.bilateralFilter(out_bgr, 9, int(75*intensity), int(75*intensity))
        
        if f_lipstick:
            lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            out_bgr = apply_makeup_realistic(out_bgr, lms, lips, (40, 30, 160), intensity)
            
        if f_big_eyes: out_bgr = enlarge_eyes(out_bgr, lms, intensity)
        
        if f_clown_nose: out_bgr = draw_clown_nose(out_bgr, lms, c_clown[0], c_clown[2], c_clown[3])

        # الإكسسوارات (تأكد من وجود المجلد assets والصور بداخله)
        acc_cfg = {
            "نظارة": (f_glasses, "assets/glasses.png", 168, 1.5, 0, c_glasses), 
            "شارب": (f_mustache, "assets/mustache.png", 2, 0.9, 15, c_mustache)
        }
        for name, (active, path, idx, bs, yo, ctrl) in acc_cfg.items():
            if active and os.path.exists(path):
                asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if asset is not None:
                    aw = int(abs(lms[263].x - lms[33].x) * w * bs * ctrl[0])
                    ah = int(aw * asset.shape[0] / asset.shape[1])
                    px = int(lms[idx].x*w) - aw//2 + ctrl[2]
                    py = int(lms[idx].y*h) - ah//2 + int(yo) + ctrl[3]
                    out_bgr = overlay_acc(out_bgr, asset, px, py, aw, ah, ctrl[1])

    return out_bgr

# --- 7. العرض الرئيسي ---
st.markdown("<h1 style='text-align: center;'>🌿 AI NATURE STUDIO</h1>", unsafe_allow_html=True)

col_main1, col_main2, col_main3 = st.columns([1, 2, 1])
with col_main2:
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        if st.button("🌲 ابدأ الكاميرا"): st.session_state.run_live = True
    with sub_col2:
        if st.button("🍂 توقف"): st.session_state.run_live = False

st.markdown("---")

if st.session_state.run_live:
    st_frame = st.empty()
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state.run_live:
        ret, frame = cap.read()
        if not ret: break
        processed = process_frame(cv2.flip(frame, 1))
        st_frame.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)
    cap.release()
else:
    file = st.file_uploader("📂 اختر صورة من جهازك", type=['jpg','png','jpeg'])
    if file:
        img_in = Image.open(file)
        orig_bgr = cv2.cvtColor(np.array(img_in), cv2.COLOR_RGB2BGR)
        res_bgr = process_frame(orig_bgr)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1: st.image(img_in, caption="الأصل", use_column_width=True)
        with col2: st.image(res_rgb, caption="بعد التعديل", use_column_width=True)
        
        buf = io.BytesIO()
        Image.fromarray(res_rgb).save(buf, format="PNG")
        st.download_button("💾 احفظ التحفة الفنية", buf.getvalue(), "nature_result.png", "image/png")