import os
import sys
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import io
import time
import mediapipe as mp

# --- 1. حل مشكلة المسارات والبيئة (متوافق مع GitHub) ---
# سطر حل المسار العربي سيعمل محلياً ولن يضر عند الرفع
try:
    import mediapipe as mp
    mp_path = os.path.join(os.path.dirname(mp.__file__), 'modules')
    os.environ['MEDIAPIPE_BINARY_GRAPH_PATH'] = mp_path
except:
    pass

# دالة ذكية لإيجاد الصور في مجلد assets
def get_asset_path(filename):
    base_path = os.path.dirname(__file__)
    return os.path.join(base_path, "assets", filename)

# --- 2. إعدادات الموديلات ---
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# --- 3. إعدادات الصفحة ---
st.set_page_config(page_title="Eco AI Face Studio", layout="wide", page_icon="🌿")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1b3022 0%, #2c5d33 100%);
        color: #f0f7f4;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e261f !important;
        border-right: 1px solid #3d4a3e;
    }
    h1, h2, h3 { color: #d4e4bc !important; }
    </style>
    """, unsafe_allow_html=True)

if 'run_live' not in st.session_state:
    st.session_state.run_live = False

@st.cache_resource
def get_models():
    mesh_mod = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seg_mod = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    return mesh_mod, seg_mod

mesh_mod, seg_mod = get_models()

# --- 4. وظائف المعالجة (نفس منطقك الرائع) ---
def enlarge_eyes(img, lms, intensity):
    if intensity == 0: return img
    h, w, _ = img.shape
    eye_centers = [468, 473]
    out = img.copy()
    radius = int(w * 0.04)
    strength = intensity * 0.4 
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
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radius:
                    factor = math.pow(dist / radius, strength) if dist > 0 else 1
                    map_x[i, j], map_y[i, j] = mid_x + dx * factor, mid_y + dy * factor
                else: map_x[i, j], map_y[i, j] = j, i
        out[y1:y2, x1:x2] = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
    return out

def apply_makeup_realistic(img, lms, points, color, alpha):
    if alpha == 0: return img
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
        if overlay.shape[2] < 4: return bg
        overlay = cv2.resize(overlay, (int(w), int(h)))
        if angle != 0:
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            overlay = cv2.warpAffine(overlay, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        y1, y2 = max(0, y), min(bg.shape[0], y + h)
        x1, x2 = max(0, x), min(bg.shape[1], x + w)
        overlay_part = overlay[0:y2-y1, 0:x2-x1]
        alpha_s = overlay_part[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            bg[y1:y2, x1:x2, c] = (alpha_s * overlay_part[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])
    except: pass
    return bg

# --- 5. واجهة المستخدم ---
st.sidebar.markdown("<h1 style='text-align: center;'>🍃 Eco Studio</h1>", unsafe_allow_html=True)
cam_id = st.sidebar.selectbox("مصدر الكاميرا", options=[0, 1, 2], index=0)

def accessory_controls(label):
    with st.sidebar.expander(f"🌱 ضبط: {label}"):
        sc = st.slider(f"الحجم", 0.1, 3.0, 1.0, key=f"s_{label}")
        rot = st.slider(f"الدوران", -180, 180, 0, key=f"r_{label}")
        off_x = st.slider(f"إزاحة X", -200, 200, 0, key=f"x_{label}")
        off_y = st.slider(f"إزاحة Y", -200, 200, 0, key=f"y_{label}")
    return sc, rot, off_x, off_y

f_smooth = st.sidebar.toggle("تنعيم البشرة ✨")
f_big_eyes = st.sidebar.toggle("عيون واسعة 👀")
f_lipstick = st.sidebar.toggle("أحمر شفاه 💄")
intensity = st.sidebar.slider("قوة التأثير", 0.0, 1.0, 0.4)

f_glasses = st.sidebar.toggle("نظارة شمسية 😎")
c_glasses = accessory_controls("النظارة") if f_glasses else (1.0, 0, 0, 0)

f_mustache = st.sidebar.toggle("شارب كلاسيكي 👨🏻")
c_mustache = accessory_controls("الشارب") if f_mustache else (1.0, 0, 0, 0)
f_blur = st.sidebar.toggle("عزل الخلفية (بوكيه) 🌳")

# --- 6. محرك المعالجة ---
def process_frame(img_bgr):
    h, w, _ = img_bgr.shape
    out_bgr = img_bgr.copy()
    
    if f_blur:
        res_seg = seg_mod.process(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        if res_seg.segmentation_mask is not None:
            mask = res_seg.segmentation_mask > 0.2
            bg_blur = cv2.GaussianBlur(out_bgr, (55, 55), 0)
            out_bgr = np.where(np.stack((mask,)*3, axis=-1), out_bgr, bg_blur)

    res_mesh = mesh_mod.process(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    if res_mesh.multi_face_landmarks:
        lms = res_mesh.multi_face_landmarks[0].landmark
        if f_smooth: out_bgr = cv2.bilateralFilter(out_bgr, 7, int(50*intensity), int(50*intensity))
        if f_lipstick:
            lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            out_bgr = apply_makeup_realistic(out_bgr, lms, lips, (60, 40, 180), intensity)
        if f_big_eyes: out_bgr = enlarge_eyes(out_bgr, lms, intensity)

        # تكوين الإكسسوارات بمسارات ذكية
        acc_cfg = {
            "نظارة": (f_glasses, get_asset_path("glasses.png"), 168, 1.6, 0, c_glasses), 
            "شارب": (f_mustache, get_asset_path("mustache.png"), 2, 1.0, 20, c_mustache)
        }
        for name, (active, path, idx, bs, yo, ctrl) in acc_cfg.items():
            if active and os.path.exists(path):
                asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if asset is not None:
                    aw = int(abs(lms[263].x - lms[33].x) * w * bs * ctrl[0])
                    ah = int(aw * asset.shape[0] / asset.shape[1])
                    px = int(lms[idx].x*w) - aw//2 + ctrl[2]
                    py = int(lms[idx].y*h) - ah//2 + yo + ctrl[3]
                    out_bgr = overlay_acc(out_bgr, asset, px, py, aw, ah, ctrl[1])
    return out_bgr

# --- 7. العرض الرئيسي ---
st.markdown("<h1 style='text-align: center;'>🌿 AI NATURE STUDIO</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🌲 ابدأ الكاميرا"): st.session_state.run_live = True
    with sc2:
        if st.button("🍂 توقف"): st.session_state.run_live = False

if st.session_state.run_live:
    st_frame = st.empty()
    cap = cv2.VideoCapture(cam_id)
    while cap.isOpened() and st.session_state.run_live:
        ret, frame = cap.read()
        if not ret: break
        processed = process_frame(cv2.flip(frame, 1))
        st_frame.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    cap.release()
else:
    file = st.file_uploader("📂 اختر صورة لمعالجتها", type=['jpg','png','jpeg'])
    if file:
        img_in = Image.open(file)
        res_bgr = process_frame(cv2.cvtColor(np.array(img_in), cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)