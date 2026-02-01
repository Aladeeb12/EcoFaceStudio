import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- إعدادات MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

# --- إعدادات السيرفر للكاميرا (لضمان العمل أونلاين) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- دالة جلب الصور ---
def get_asset(name):
    path = os.path.join(os.path.dirname(__file__), "assets", name)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None

# --- التنسيق الجمالي (CSS) ---
st.set_page_config(page_title="Eco AI Studio", layout="wide")
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1b3022 0%, #2c5d33 100%); color: #f0f7f4; }
    h1 { color: #d4e4bc !important; text-align: center; }
    sidebar .sidebar-content { background-color: #1e261f; }
    </style>
    """, unsafe_allow_html=True)

# --- الواجهة الجانبية ---
st.sidebar.title("🎭 لوحة التحكم")
f_mask = st.sidebar.toggle("إظهار الماسك (Mesh) 🕸️")
f_glasses = st.sidebar.toggle("نظارة شمسية 😎")
f_mustache = st.sidebar.toggle("شارب كلاسيكي 👨🏻")
intensity = st.sidebar.slider("قوة تأثير التجميل", 0.0, 1.0, 0.5)

# تحميل الإكسسوارات
img_glasses = get_asset("glasses.png")
img_mustache = get_asset("mustache.png")

# --- محرك المعالجة الحية ---
class FaceProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for lms in results.multi_face_landmarks:
                landmarks = lms.landmark
                
                # 1. رسم الماسك (الشبكة)
                if f_mask:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img, lms, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                # 2. إضافة النظارة
                if f_glasses and img_glasses is not None:
                    # حساب عرض العينين لضبط حجم النظارة
                    eye_width = int(abs(landmarks[263].x - landmarks[33].x) * w * 1.5)
                    eye_height = int(eye_width * img_glasses.shape[0] / img_glasses.shape[1])
                    # تحديد موقع النظارة (نقطة منتصف الأنف 168)
                    x = int(landmarks[168].x * w) - eye_width // 2
                    y = int(landmarks[168].y * h) - eye_height // 2
                    img = self.overlay_image(img, img_glasses, x, y, eye_width, eye_height)

                # 3. إضافة الشارب
                if f_mustache and img_mustache is not None:
                    m_width = int(abs(landmarks[205].x - landmarks[425].x) * w * 1.2)
                    m_height = int(m_width * img_mustache.shape[0] / img_mustache.shape[1])
                    x = int(landmarks[2].x * w) - m_width // 2
                    y = int(landmarks[2].y * h) # نقطة تحت الأنف
                    img = self.overlay_image(img, img_mustache, x, y, m_width, m_height)

        return img

    def overlay_image(self, bg, overlay, x, y, w, h):
        try:
            overlay_res = cv2.resize(overlay, (w, h))
            y1, y2 = max(0, y), min(bg.shape[0], y + h)
            x1, x2 = max(0, x), min(bg.shape[1], x + w)
            overlay_part = overlay_res[0:y2-y1, 0:x2-x1]
            alpha = overlay_part[:, :, 3] / 255.0
            for c in range(3):
                bg[y1:y2, x1:x2, c] = (alpha * overlay_part[:, :, c] + (1 - alpha) * bg[y1:y2, x1:x2, c])
        except: pass
        return bg

# --- تشغيل التطبيق ---
st.title("🌿 Eco AI Nature Studio")
st.write("قم بتفعيل الفلاتر من اليسار ثم ابدأ الكاميرا")

webrtc_streamer(
    key="eco-filter",
    video_transformer_factory=FaceProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)