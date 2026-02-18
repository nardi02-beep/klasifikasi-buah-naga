"""
🔴 AUTO-CAPTURE V4 - STREAMLIT WEBRTC VERSION
OPTIMIZED for RED Dragon Fruit Detection with Real-Time Camera

Key Features:
1. ✅ Real-time video processing (like cv2.VideoCapture)
2. ✅ Enhanced RED color detection (dark, bright, pink)
3. ✅ Brightness normalization
4. ✅ Auto-capture with validation
5. ✅ Streamlit UI for easy deployment
"""

import streamlit as st
import cv2
import numpy as np
import joblib
from scipy.stats import skew
from collections import deque
import time
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Dragon Fruit Scanner V4",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 1. LOAD MODEL ----------
MODEL_PATH = "knn_buah_naga_optimized.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        
        if hasattr(model, 'classes_'):
            classes = model.classes_
        elif hasattr(model, 'named_steps'):
            knn = model.named_steps.get('knn')
            classes = knn.classes_ if knn and hasattr(knn, 'classes_') else []
        else:
            classes = []
        
        return model, classes
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, []

model, classes = load_model()

# ---------- 2. CONFIGURATION (OPTIMIZED FOR REAL FRUIT) ----------
TARGET_SIZE = (800, 800)
ROI_SIZE = 350

# 🔴 OPTIMIZED FOR RED DRAGON FRUIT
MIN_DRAGON_FRUIT_AREA = 12.0        # 12% (turun dari 20%) - akomodasi refleksi
MIN_VARIANCE = 150                   # Very lenient - glossy surface OK
MIN_EDGE_DENSITY = 0.015             # Very lenient - smooth surface OK
MAX_VARIANCE = 2000                  # Reject screen/display (texture terlalu tinggi)

BASE_CAPTURE_DIR = "auto_captures"
os.makedirs(BASE_CAPTURE_DIR, exist_ok=True)

# Session state for video processor
if 'prediction_buffer' not in st.session_state:
    st.session_state.prediction_buffer = deque(maxlen=7)
if 'capture_history' not in st.session_state:
    st.session_state.capture_history = deque(maxlen=10)
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 65.0
if 'capture_cooldown' not in st.session_state:
    st.session_state.capture_cooldown = 3.0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_captures' not in st.session_state:
    st.session_state.total_captures = 0

# ---------- 3. HELPER FUNCTIONS ----------
def bgr_to_hsi(bgr_image):
    """BGR to HSI conversion"""
    b = bgr_image[:, :, 0].astype(np.float32) / 255.0
    g = bgr_image[:, :, 1].astype(np.float32) / 255.0
    r = bgr_image[:, :, 2].astype(np.float32) / 255.0

    i = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    denominator = r + g + b + 1e-6
    s = 1 - (3.0 * min_rgb) / denominator
    s = np.clip(s, 0, 1)

    numerator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    valid = numerator > 1e-6
    h = np.zeros_like(numerator)
    cos_theta = np.where(valid, ((r - g) + (r - b)) / (2.0 * numerator + 1e-6), 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    h = np.degrees(np.arccos(cos_theta))
    h[b > g] = 360 - h[b > g]
    h = np.clip(h, 0, 360)
    return h, s, i

def extract_color_features(bgr_img):
    """Extract 18 color features"""
    img = cv2.resize(bgr_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    features = []
    def get_stats(channel):
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        skew_val = skew(channel.flatten()) if std_val > 1e-6 else 0.0
        return mean_val, std_val, skew_val
    
    b, g, r = cv2.split(img)
    for channel in [r, g, b]:
        features.extend(get_stats(channel))
    
    h, s, i_channel = bgr_to_hsi(img)
    for channel in [h, s, i_channel]:
        features.extend(get_stats(channel))
    
    return np.array(features)

# ---------- 4. PREDICTION ----------
def predict_maturity(roi):
    """Predict with smoothing using session state"""
    try:
        features = extract_color_features(roi)
        label = model.predict([features])[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([features])[0]
            confidence = np.max(proba) * 100
            
            if hasattr(model, 'classes_'):
                classes = model.classes_
            elif hasattr(model, 'named_steps'):
                knn = model.named_steps.get('knn')
                classes = knn.classes_ if knn else []
            else:
                classes = []
            
            class_probs = {cls: prob * 100 for cls, prob in zip(classes, proba)}
        else:
            confidence = 100.0
            class_probs = {label: 100.0}
        
        st.session_state.prediction_buffer.append((label, confidence))
        
        labels = [pred[0] for pred in st.session_state.prediction_buffer]
        from collections import Counter
        label_counts = Counter(labels)
        smoothed_label = label_counts.most_common(1)[0][0]
        
        confidences = [pred[1] for pred in st.session_state.prediction_buffer if pred[0] == smoothed_label]
        smoothed_confidence = np.mean(confidences) if confidences else confidence
        
        return smoothed_label, smoothed_confidence, features, class_probs
    except Exception as e:
        return "ERROR", 0.0, None, {}

# ---------- 5. 🔴 ENHANCED RED DRAGON FRUIT COLOR DETECTION ----------
def is_dragon_fruit_color(roi):
    """
    🔴 OPTIMIZED untuk buah naga merah ASLI
    
    Covers:
    - Dark red (mature): H=0-10, 170-180, S=20-100, V=30-90
    - Bright red (ripe): H=0-15, 165-180, S=30-100, V=70-255
    - Pink (immature): H=340-360, S=20-80, V=60-200
    
    Returns: (is_dragon_fruit, color_type, percentage, debug_info)
    """
    # Brightness normalization untuk consistent detection
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l = cv2.equalizeHist(l)  # Normalize luminance
    lab_normalized = cv2.merge([l, a, b_ch])
    roi_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi_normalized, cv2.COLOR_BGR2HSV)
    
    # 🔴 RED COLOR RANGES (EXPANDED!)
    
    # Dark Red (mature dragon fruit) - merah gelap
    # H: 0-10 & 170-180 (red wrap), S: 20-100 (akomodasi glossy), V: 30-90 (dark)
    mask_dark_red1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([10, 100, 90]))
    mask_dark_red2 = cv2.inRange(hsv, np.array([170, 20, 30]), np.array([180, 100, 90]))
    mask_dark_red = cv2.bitwise_or(mask_dark_red1, mask_dark_red2)
    
    # Bright Red (ripe dragon fruit) - merah cerah
    # H: 0-15 & 165-180, S: 30-100, V: 70-255 (bright)
    mask_bright_red1 = cv2.inRange(hsv, np.array([0, 30, 70]), np.array([15, 100, 255]))
    mask_bright_red2 = cv2.inRange(hsv, np.array([165, 30, 70]), np.array([180, 100, 255]))
    mask_bright_red = cv2.bitwise_or(mask_bright_red1, mask_bright_red2)
    
    # Pink/Light Red (less mature) - merah muda
    # H: 0-5 & 175-180, S: 15-60, V: 80-255
    mask_pink1 = cv2.inRange(hsv, np.array([0, 15, 80]), np.array([5, 60, 255]))
    mask_pink2 = cv2.inRange(hsv, np.array([175, 15, 80]), np.array([180, 60, 255]))
    mask_pink = cv2.bitwise_or(mask_pink1, mask_pink2)
    
    # Calculate percentages
    total_pixels = roi.shape[0] * roi.shape[1]
    dark_red_percent = (np.sum(mask_dark_red > 0) / total_pixels) * 100
    bright_red_percent = (np.sum(mask_bright_red > 0) / total_pixels) * 100
    pink_percent = (np.sum(mask_pink > 0) / total_pixels) * 100
    
    # Combine all red masks
    mask_all_red = cv2.bitwise_or(mask_dark_red, mask_bright_red)
    mask_all_red = cv2.bitwise_or(mask_all_red, mask_pink)
    total_red_percent = (np.sum(mask_all_red > 0) / total_pixels) * 100
    
    # Also check for yellow/green (untuk buah naga kuning/putih)
    mask_yellow = cv2.inRange(hsv, np.array([20, 30, 60]), np.array([40, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([40, 20, 40]), np.array([80, 180, 220]))
    
    yellow_percent = (np.sum(mask_yellow > 0) / total_pixels) * 100
    green_percent = (np.sum(mask_green > 0) / total_pixels) * 100
    
    # Total dragon fruit color (red + yellow + green)
    mask_dragon_fruit = cv2.bitwise_or(mask_all_red, mask_yellow)
    mask_dragon_fruit = cv2.bitwise_or(mask_dragon_fruit, mask_green)
    dragon_fruit_percent = (np.sum(mask_dragon_fruit > 0) / total_pixels) * 100
    
    # Determine color type
    color_percentages = {
        'Dark Red': dark_red_percent,
        'Bright Red': bright_red_percent,
        'Pink': pink_percent,
        'Yellow': yellow_percent,
        'Green': green_percent
    }
    dominant_color = max(color_percentages, key=color_percentages.get)
    
    # 🎯 VALIDATION: Minimal 12% (turun dari 20%)
    is_dragon_fruit = dragon_fruit_percent >= MIN_DRAGON_FRUIT_AREA
    
    debug_info = {
        'dark_red': dark_red_percent,
        'bright_red': bright_red_percent,
        'pink': pink_percent,
        'yellow': yellow_percent,
        'green': green_percent,
        'total_red': total_red_percent,
        'total': dragon_fruit_percent
    }
    
    return is_dragon_fruit, dominant_color, dragon_fruit_percent, debug_info

# ---------- 6. OBJECT DETECTION (VERY LENIENT + REJECT SCREEN) ----------
def is_object_present(roi):
    """
    Object detection dengan:
    - Very lenient threshold (buah asli OK)
    - Reject screen/display (variance terlalu tinggi)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Check variance range
    has_variance_ok = MIN_VARIANCE < variance < MAX_VARIANCE
    has_edges_ok = edge_density > MIN_EDGE_DENSITY
    
    # Reject if variance TOO HIGH (likely screen/display)
    is_screen = variance > MAX_VARIANCE
    
    is_valid_object = has_variance_ok and has_edges_ok and not is_screen
    
    return is_valid_object, variance, edge_density, is_screen

# ---------- 7. VISUAL HELPERS ----------
def get_color_by_status(color_valid, confidence, is_screen):
    """ROI box color using session state"""
    confidence_threshold = st.session_state.confidence_threshold
    
    if is_screen:
        return (0, 0, 255)        # Red - screen detected
    elif not color_valid:
        return (100, 100, 100)    # Gray - not dragon fruit
    elif confidence >= confidence_threshold:
        return (0, 255, 0)        # Green - ready
    else:
        return (0, 255, 255)      # Yellow - low confidence

# ---------- 8. VIDEO TRANSFORMER CLASS ----------
class DragonFruitDetector(VideoTransformerBase):
    """Real-time video processor for Dragon Fruit Detection"""
    
    def __init__(self):
        self.prev_frame = None
        self.frame_count = 0
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        st.session_state.frame_count += 1
        h, w = img.shape[:2]
        
        # ROI
        size = ROI_SIZE
        x1, y1 = (w - size) // 2, (h - size) // 2
        x2, y2 = x1 + size, y1 + size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Layer 1: Object detection
        obj_present, variance, edge_density, is_screen = is_object_present(roi)
        obj_info = (obj_present, variance, edge_density, is_screen)
        
        # Default values
        label, confidence, class_probs = "WAITING", 0.0, {}
        color_valid = False
        color_info = {'dark_red': 0, 'bright_red': 0, 'pink': 0, 'yellow': 0, 
                     'green': 0, 'total_red': 0, 'total': 0}
        
        # Layer 2: Color validation
        if obj_present and not is_screen:
            color_valid, dominant_color, color_percent, debug_info = is_dragon_fruit_color(roi)
            color_info = debug_info
            color_info['dominant'] = dominant_color
            
            # Layer 3: Prediction (process every frame for real-time)
            if color_valid:
                label, confidence, _, class_probs = predict_maturity(roi)
                
                # Auto-capture with session state
                captured, status = auto_capture_streamlit(
                    roi, label, confidence, class_probs, 
                    color_valid, color_percent, is_screen
                )
                
                if captured:
                    # Flash effect
                    flash = np.ones_like(img) * 255
                    img = cv2.addWeighted(img, 0.6, flash, 0.4, 0)
        else:
            if is_screen:
                label = "SCREEN DETECTED"
            else:
                label = "NO OBJECT"
        
        # Draw ROI box
        box_color = get_color_by_status(color_valid, confidence, is_screen)
        confidence_threshold = st.session_state.confidence_threshold
        thickness = 4 if (color_valid and confidence >= confidence_threshold and not is_screen) else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(img, (center_x, center_y), box_color, cv2.MARKER_CROSS, 30, 3)
        
        # Status text above ROI
        if is_screen:
            status_text = "SCREEN/DISPLAY - REJECTED"
            status_color = (0, 0, 255)
        elif not obj_present:
            status_text = "WAITING FOR OBJECT"
            status_color = (100, 100, 100)
        elif not color_valid:
            status_text = f"NOT DRAGON FRUIT ({color_info['total']:.1f}%)"
            status_color = (0, 100, 255)
        elif confidence < confidence_threshold:
            status_text = f"LOW CONF ({confidence:.1f}%)"
            status_color = (0, 255, 255)
        else:
            status_text = "READY!"
            status_color = (0, 255, 0)
        
        # Draw status text
        text_y = max(y1 - 10, 30)
        cv2.putText(img, status_text, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw compact info overlay
        if color_valid or obj_present:
            self.draw_compact_info(img, label, confidence, class_probs, 
                                  color_valid, color_info, obj_info)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def draw_compact_info(self, frame, label, confidence, class_probs, 
                         color_valid, color_info, obj_info):
        """Compact info overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent black box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        y = 30
        
        # Object Detection
        obj_present, variance, edge_density, is_screen = obj_info
        status = "SCREEN" if is_screen else ("OK" if obj_present else "FAIL")
        color = (0, 0, 255) if (is_screen or not obj_present) else (0, 255, 0)
        cv2.putText(frame, f"Object: {status} (Var:{variance:.0f})", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 30
        
        # Color Detection
        status = "OK" if color_valid else "FAIL"
        color = (0, 255, 0) if color_valid else (0, 0, 255)
        cv2.putText(frame, f"Color: {status} ({color_info['total']:.1f}%)", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 25
        cv2.putText(frame, f"  R:{color_info['total_red']:.0f}% P:{color_info['pink']:.0f}%", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += 30
        
        # Prediction
        conf_threshold = st.session_state.confidence_threshold
        status = "OK" if confidence >= conf_threshold else "LOW"
        color = (0, 255, 0) if confidence >= conf_threshold else (0, 165, 255)
        cv2.putText(frame, f"Predict: {label} ({confidence:.1f}%)", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 30
        
        # Result
        all_pass = obj_present and not is_screen and color_valid and confidence >= conf_threshold
        result = "READY TO CAPTURE" if all_pass else "Validating..."
        result_color = (0, 255, 0) if all_pass else (100, 100, 100)
        cv2.putText(frame, result, 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 2)

# ---------- 9. AUTO-CAPTURE FOR STREAMLIT ----------
def auto_capture_streamlit(roi, label, confidence, class_probs, color_valid, color_percent, is_screen):
    """Auto capture with Streamlit session state"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_capture_time
    confidence_threshold = st.session_state.confidence_threshold
    capture_cooldown = st.session_state.capture_cooldown
    
    # Reject screen/display
    if is_screen:
        return False, "Screen detected"
    
    # Color validation
    if not color_valid:
        return False, f"Color: {color_percent:.1f}%"
    
    # Confidence check
    if confidence < confidence_threshold:
        return False, f"Low confidence: {confidence:.1f}%"
    
    # Cooldown
    if time_since_last < capture_cooldown:
        return False, f"Cooldown: {capture_cooldown - time_since_last:.1f}s"
    
    # CAPTURE!
    class_dir = os.path.join(BASE_CAPTURE_DIR, label)
    os.makedirs(class_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{confidence:.0f}_{timestamp}.jpg"
    filepath = os.path.join(class_dir, filename)
    
    cv2.imwrite(filepath, roi)
    st.session_state.last_capture_time = current_time
    st.session_state.total_captures += 1
    
    capture_info = {
        'time': datetime.now().strftime("%H:%M:%S"),
        'label': label,
        'confidence': confidence,
        'filename': filename,
        'filepath': filepath
    }
    st.session_state.capture_history.append(capture_info)
    
    return True, "✅ Captured!"

# ---------- 10. STREAMLIT UI ----------
def main():
    st.title("🔴 Dragon Fruit Scanner V4 - Streamlit WebRTC")
    st.markdown("Real-time detection untuk buah naga merah dengan auto-capture")
    
    if model is None:
        st.error("❌ Model tidak ditemukan! Pastikan file `knn_buah_naga_optimized.pkl` ada di folder yang sama.")
        st.stop()
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Model Info")
        st.info(f"📦 Model: {MODEL_PATH}")
        if len(classes) > 0:
            st.success(f"✅ Classes: {', '.join(map(str, classes))}")
        
        st.markdown("---")
        
        st.subheader("Detection Parameters")
        
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50.0,
            max_value=95.0,
            value=st.session_state.confidence_threshold,
            step=5.0,
            help="Minimum confidence untuk auto-capture"
        )
        
        st.session_state.capture_cooldown = st.slider(
            "Capture Cooldown (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.capture_cooldown,
            step=0.5,
            help="Jeda waktu antar capture"
        )
        
        st.markdown("---")
        
        st.subheader("Detection Info")
        st.text(f"Min Color Area: {MIN_DRAGON_FRUIT_AREA}%")
        st.text(f"Variance Range: {MIN_VARIANCE}-{MAX_VARIANCE}")
        st.text(f"Edge Density: >{MIN_EDGE_DENSITY}")
        
        st.markdown("---")
        
        st.subheader("📊 Statistics")
        st.metric("Total Frames", st.session_state.frame_count)
        st.metric("Total Captures", st.session_state.total_captures)
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.frame_count = 0
            st.session_state.total_captures = 0
            st.session_state.capture_history.clear()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # WebRTC Configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="dragon-fruit-detection",
            video_transformer_factory=DragonFruitDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        st.markdown("""
        **📝 Instructions:**
        1. Klik **START** untuk memulai kamera
        2. Arahkan kamera ke buah naga merah
        3. Tunggu hingga deteksi valid (kotak hijau)
        4. Auto-capture akan berjalan otomatis
        5. Hasil capture akan muncul di panel kanan
        """)
    
    with col2:
        st.subheader("📸 Capture History")
        
        if len(st.session_state.capture_history) > 0:
            for i, cap in enumerate(reversed(list(st.session_state.capture_history))):
                with st.expander(f"#{i+1} - {cap['time']} | {cap['label']} ({cap['confidence']:.1f}%)"):
                    st.text(f"Time: {cap['time']}")
                    st.text(f"Label: {cap['label']}")
                    st.text(f"Confidence: {cap['confidence']:.1f}%")
                    st.text(f"File: {cap['filename']}")
                    
                    # Show captured image
                    if os.path.exists(cap['filepath']):
                        img = cv2.imread(cap['filepath'])
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
        else:
            st.info("Belum ada capture. Arahkan kamera ke buah naga merah.")
        
        if st.button("🗑️ Clear History"):
            st.session_state.capture_history.clear()
            st.rerun()
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    ### 🔍 Detection Layers:
    1. **Object Detection**: Variance & edge density validation
    2. **Color Detection**: Red color range (dark, bright, pink)
    3. **KNN Prediction**: Maturity classification
    
    **Auto-Capture Conditions:**
    - ✅ Valid object detected (not screen/display)
    - ✅ Dragon fruit color ≥ {:.1f}%
    - ✅ Confidence ≥ {:.1f}%
    - ✅ Cooldown passed
    """.format(MIN_DRAGON_FRUIT_AREA, st.session_state.confidence_threshold))

if __name__ == "__main__":
    main()
