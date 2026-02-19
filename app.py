"""
🔴 Dragon Fruit Scanner - STREAMLIT CAMERA INPUT VERSION
Simplified version without WebRTC for better cloud compatibility

Key Features:
1. ✅ Camera input (foto langsung dari kamera)
2. ✅ Upload file support
3. ✅ Enhanced RED color detection
4. ✅ Auto-save captures
5. ✅ No WebRTC dependency
"""

import streamlit as st
import cv2
import numpy as np
import joblib
from scipy.stats import skew
import time
import os
from datetime import datetime
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Dragon Fruit Scanner",
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

# ---------- 2. CONFIGURATION ----------
TARGET_SIZE = (800, 800)
MIN_DRAGON_FRUIT_AREA = 12.0
MIN_VARIANCE = 100               # Lower threshold - real fruits OK
MIN_EDGE_DENSITY = 0.010         # More lenient
MAX_VARIANCE = 8000              # Much higher - reject only obvious screens

BASE_CAPTURE_DIR = "auto_captures"
os.makedirs(BASE_CAPTURE_DIR, exist_ok=True)

# Session state
if 'capture_history' not in st.session_state:
    st.session_state.capture_history = []
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 65.0
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'enable_screen_rejection' not in st.session_state:
    st.session_state.enable_screen_rejection = False

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
    """Predict maturity with confidence"""
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
        
        return label, confidence, class_probs
    except Exception as e:
        return "ERROR", 0.0, {}

# ---------- 5. COLOR DETECTION ----------
def is_dragon_fruit_color(roi):
    """Enhanced red dragon fruit color detection"""
    # Brightness normalization
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab_normalized = cv2.merge([l, a, b_ch])
    roi_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(roi_normalized, cv2.COLOR_BGR2HSV)
    
    # Red color ranges
    mask_dark_red1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([10, 100, 90]))
    mask_dark_red2 = cv2.inRange(hsv, np.array([170, 20, 30]), np.array([180, 100, 90]))
    mask_dark_red = cv2.bitwise_or(mask_dark_red1, mask_dark_red2)
    
    mask_bright_red1 = cv2.inRange(hsv, np.array([0, 30, 70]), np.array([15, 100, 255]))
    mask_bright_red2 = cv2.inRange(hsv, np.array([165, 30, 70]), np.array([180, 100, 255]))
    mask_bright_red = cv2.bitwise_or(mask_bright_red1, mask_bright_red2)
    
    mask_pink1 = cv2.inRange(hsv, np.array([0, 15, 80]), np.array([5, 60, 255]))
    mask_pink2 = cv2.inRange(hsv, np.array([175, 15, 80]), np.array([180, 60, 255]))
    mask_pink = cv2.bitwise_or(mask_pink1, mask_pink2)
    
    # Calculate percentages
    total_pixels = roi.shape[0] * roi.shape[1]
    dark_red_percent = (np.sum(mask_dark_red > 0) / total_pixels) * 100
    bright_red_percent = (np.sum(mask_bright_red > 0) / total_pixels) * 100
    pink_percent = (np.sum(mask_pink > 0) / total_pixels) * 100
    
    mask_all_red = cv2.bitwise_or(mask_dark_red, mask_bright_red)
    mask_all_red = cv2.bitwise_or(mask_all_red, mask_pink)
    total_red_percent = (np.sum(mask_all_red > 0) / total_pixels) * 100
    
    # Yellow/green for other dragon fruit types
    mask_yellow = cv2.inRange(hsv, np.array([20, 30, 60]), np.array([40, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([40, 20, 40]), np.array([80, 180, 220]))
    
    yellow_percent = (np.sum(mask_yellow > 0) / total_pixels) * 100
    green_percent = (np.sum(mask_green > 0) / total_pixels) * 100
    
    mask_dragon_fruit = cv2.bitwise_or(mask_all_red, mask_yellow)
    mask_dragon_fruit = cv2.bitwise_or(mask_dragon_fruit, mask_green)
    dragon_fruit_percent = (np.sum(mask_dragon_fruit > 0) / total_pixels) * 100
    
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
    
    return is_dragon_fruit, dragon_fruit_percent, debug_info

# ---------- 6. OBJECT DETECTION ----------
def is_object_present(roi):
    """Object detection with optional screen rejection"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Check if it's a screen (only if enabled)
    is_screen = False
    if st.session_state.enable_screen_rejection:
        is_screen = variance > MAX_VARIANCE
    
    # More lenient validation - focus on minimum requirements
    has_variance_ok = variance > MIN_VARIANCE
    has_edges_ok = edge_density > MIN_EDGE_DENSITY
    
    is_valid_object = has_variance_ok and has_edges_ok and not is_screen
    
    return is_valid_object, variance, edge_density, is_screen

# ---------- 7. IMAGE PROCESSING ----------
def process_image(image):
    """Process uploaded/captured image"""
    # Convert PIL to OpenCV
    if isinstance(image, Image.Image):
        img = np.array(image)
        if img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    h, w = img.shape[:2]
    
    # Crop to center square
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    roi = img[y1:y1+size, x1:x1+size]
    
    # Validate
    obj_present, variance, edge_density, is_screen = is_object_present(roi)
    color_valid, color_percent, color_info = is_dragon_fruit_color(roi)
    
    result = {
        'roi': roi,
        'obj_present': obj_present,
        'variance': variance,
        'edge_density': edge_density,
        'is_screen': is_screen,
        'color_valid': color_valid,
        'color_percent': color_percent,
        'color_info': color_info,
        'annotated': None,
        'label': None,
        'confidence': 0.0,
        'class_probs': {}
    }
    
    # Create annotated image
    annotated = img.copy()
    x2, y2 = x1 + size, y1 + size
    
    # Determine status
    if is_screen:
        box_color = (0, 0, 255)  # Red
        status_text = "SCREEN/DISPLAY - REJECTED"
        status_color = (0, 0, 255)
    elif not obj_present:
        box_color = (100, 100, 100)  # Gray
        status_text = "NO VALID OBJECT"
        status_color = (100, 100, 100)
    elif not color_valid:
        box_color = (0, 100, 255)  # Orange
        status_text = f"NOT DRAGON FRUIT ({color_percent:.1f}%)"
        status_color = (0, 100, 255)
    else:
        # Predict
        label, confidence, class_probs = predict_maturity(roi)
        result['label'] = label
        result['confidence'] = confidence
        result['class_probs'] = class_probs
        
        conf_threshold = st.session_state.confidence_threshold
        if confidence >= conf_threshold:
            box_color = (0, 255, 0)  # Green
            status_text = f"✓ {label} ({confidence:.1f}%)"
            status_color = (0, 255, 0)
        else:
            box_color = (0, 255, 255)  # Yellow
            status_text = f"LOW CONF: {label} ({confidence:.1f}%)"
            status_color = (0, 255, 255)
    
    # Draw box and text
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
    
    # Status text
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x1
    text_y = max(y1 - 10, text_size[1] + 10)
    
    # Background for text
    cv2.rectangle(annotated, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)
    cv2.putText(annotated, status_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    result['annotated'] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return result

# ---------- 8. SAVE CAPTURE ----------
def save_capture(roi, label, confidence):
    """Save captured image"""
    class_dir = os.path.join(BASE_CAPTURE_DIR, label)
    os.makedirs(class_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{confidence:.0f}_{timestamp}.jpg"
    filepath = os.path.join(class_dir, filename)
    
    cv2.imwrite(filepath, roi)
    
    return filename, filepath

# ---------- 9. MAIN UI ----------
def main():
    st.title("🔴 Dragon Fruit Scanner")
    st.markdown("📸 Ambil foto atau upload gambar buah naga untuk deteksi otomatis")
    
    if model is None:
        st.error("❌ Model tidak ditemukan! Pastikan file `knn_buah_naga_optimized.pkl` ada di folder yang sama.")
        st.stop()
    
    # Sidebar
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
            help="Minimum confidence untuk dianggap valid"
        )
        
        st.markdown("---")
        
        st.subheader("Detection Settings")
        
        st.session_state.enable_screen_rejection = st.checkbox(
            "Enable Screen/Display Rejection",
            value=st.session_state.enable_screen_rejection,
            help="Reject images with very high variance (screens/displays). Disable if real fruits are rejected."
        )
        
        if not st.session_state.enable_screen_rejection:
            st.info("ℹ️ Screen rejection disabled - all objects will be processed")
        
        st.markdown("---")
        
        st.subheader("Detection Info")
        st.text(f"Min Color Area: {MIN_DRAGON_FRUIT_AREA}%")
        st.text(f"Min Variance: >{MIN_VARIANCE}")
        if st.session_state.enable_screen_rejection:
            st.text(f"Max Variance: <{MAX_VARIANCE}")
        st.text(f"Edge Density: >{MIN_EDGE_DENSITY}")
        
        st.markdown("---")
        
        st.subheader("📊 Statistics")
        st.metric("Total Scans", st.session_state.total_scans)
        st.metric("Total Saves", len(st.session_state.capture_history))
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.total_scans = 0
            st.session_state.capture_history = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Input Image")
        
        # Tabs for different input methods
        tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload File"])
        
        image_source = None
        
        with tab1:
            st.markdown("**Ambil foto langsung dari kamera:**")
            camera_image = st.camera_input("Arahkan kamera ke buah naga")
            if camera_image:
                image_source = Image.open(camera_image)
        
        with tab2:
            st.markdown("**Upload gambar dari file:**")
            uploaded_file = st.file_uploader("Pilih file gambar", 
                                            type=['jpg', 'jpeg', 'png'],
                                            help="Format: JPG, JPEG, PNG")
            if uploaded_file:
                image_source = Image.open(uploaded_file)
        
        # Process image
        if image_source:
            st.session_state.total_scans += 1
            
            with st.spinner("🔍 Processing..."):
                result = process_image(image_source)
            
            # Display result
            st.image(result['annotated'], use_container_width=True)
            
            # Result details
            st.markdown("---")
            st.subheader("📊 Detection Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                obj_status = "❌ SCREEN" if result['is_screen'] else ("✅ OK" if result['obj_present'] else "❌ FAIL")
                st.metric("Object Detection", obj_status)
                
                # Show detailed variance info
                var_info = f"Variance: {result['variance']:.0f}"
                if st.session_state.enable_screen_rejection and result['is_screen']:
                    var_info += f" (>{MAX_VARIANCE})"
                elif not result['obj_present'] and result['variance'] <= MIN_VARIANCE:
                    var_info += f" (too low)"
                st.caption(var_info)
                st.caption(f"Edges: {result['edge_density']:.4f}")
            
            with col_b:
                color_status = "✅ OK" if result['color_valid'] else "❌ FAIL"
                st.metric("Color Validation", color_status)
                st.caption(f"Dragon Fruit: {result['color_percent']:.1f}%")
            
            with col_c:
                if result['label']:
                    conf_ok = result['confidence'] >= st.session_state.confidence_threshold
                    pred_status = "✅ HIGH" if conf_ok else "⚠️ LOW"
                    st.metric("Prediction", pred_status)
                    st.caption(f"{result['label']}: {result['confidence']:.1f}%")
                else:
                    st.metric("Prediction", "N/A")
            
            # Debug info expander
            with st.expander("🔍 Debug Information"):
                st.markdown("**Object Detection Details:**")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.metric("Variance", f"{result['variance']:.2f}")
                    st.caption(f"Min: {MIN_VARIANCE} | Max: {MAX_VARIANCE if st.session_state.enable_screen_rejection else 'N/A'}")
                with col_d2:
                    st.metric("Edge Density", f"{result['edge_density']:.4f}")
                    st.caption(f"Min: {MIN_EDGE_DENSITY}")
                
                st.markdown("**Color Detection Details:**")
                col_d3, col_d4 = st.columns(2)
                with col_d3:
                    st.write(f"🔴 Dark Red: {result['color_info']['dark_red']:.1f}%")
                    st.write(f"🔴 Bright Red: {result['color_info']['bright_red']:.1f}%")
                    st.write(f"🌸 Pink: {result['color_info']['pink']:.1f}%")
                with col_d4:
                    st.write(f"🟡 Yellow: {result['color_info']['yellow']:.1f}%")
                    st.write(f"🟢 Green: {result['color_info']['green']:.1f}%")
                    st.write(f"📊 **Total: {result['color_info']['total']:.1f}%**")
                
                st.markdown("**Validation Status:**")
                st.write(f"✓ Object Present: {'✅ Yes' if result['obj_present'] else '❌ No'}")
                st.write(f"✓ Color Valid: {'✅ Yes' if result['color_valid'] else '❌ No'} ({result['color_percent']:.1f}% >= {MIN_DRAGON_FRUIT_AREA}%)")
                st.write(f"✓ Screen Rejection: {'⚠️ Enabled' if st.session_state.enable_screen_rejection else '✅ Disabled'}")
                st.write(f"✓ Is Screen: {'❌ Yes' if result['is_screen'] else '✅ No'}")
            
            # Detailed info
            if result['label']:
                st.markdown("---")
                st.subheader("📈 Class Probabilities")
                for cls, prob in sorted(result['class_probs'].items(), key=lambda x: x[1], reverse=True):
                    st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                
                # Save button
                if result['confidence'] >= st.session_state.confidence_threshold:
                    if st.button("💾 Save This Capture", type="primary"):
                        filename, filepath = save_capture(result['roi'], result['label'], result['confidence'])
                        
                        capture_info = {
                            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'label': result['label'],
                            'confidence': result['confidence'],
                            'filename': filename,
                            'filepath': filepath
                        }
                        st.session_state.capture_history.insert(0, capture_info)
                        
                        st.success(f"✅ Saved as: {filename}")
                        st.rerun()
    
    with col2:
        st.subheader("📝 History")
        
        if st.session_state.capture_history:
            for i, cap in enumerate(st.session_state.capture_history[:5]):
                with st.expander(f"#{i+1} - {cap['label']} ({cap['confidence']:.1f}%)"):
                    st.text(f"Time: {cap['time']}")
                    st.text(f"Label: {cap['label']}")
                    st.text(f"Confidence: {cap['confidence']:.1f}%")
                    st.text(f"File: {cap['filename']}")
                    
                    if os.path.exists(cap['filepath']):
                        img = cv2.imread(cap['filepath'])
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
            
            if len(st.session_state.capture_history) > 5:
                st.info(f"Showing 5 of {len(st.session_state.capture_history)} saved captures")
        else:
            st.info("Belum ada capture yang disimpan")
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ How It Works & Troubleshooting"):
        st.markdown("""
        ### 🔍 Detection Process:
        
        1. **Object Detection**
           - Check variance (texture complexity)
           - Check edge density (object presence)
           - Optional: Reject screens/displays (if enabled)
        
        2. **Color Validation**
           - Detect red dragon fruit colors
           - Support dark red, bright red, pink
           - Also detect yellow/white varieties
           - Minimum 12% dragon fruit color required
        
        3. **KNN Classification**
           - Predict maturity level
           - Calculate confidence score
           - Show probability for each class
        
        4. **Save Feature**
           - Save captures with high confidence
           - Organize by classification
           - Auto-filename with timestamp
        
        ### 🛠️ Troubleshooting:
        
        **"SCREEN/DISPLAY - REJECTED":**
        - ✅ **Disable "Screen/Display Rejection"** in sidebar
        - This happens when variance is very high (>8000)
        - Real dragon fruits with detailed scales may trigger this
        - Disable this setting if you're scanning real fruits
        
        **"NO VALID OBJECT":**
        - ✅ Ensure good lighting
        - ✅ Image not too blurry or uniform
        - ✅ Fruit visible in center of frame
        
        **"NOT DRAGON FRUIT":**
        - ✅ Check if it's actually dragon fruit
        - ✅ Color percentage shows detection (<12% = rejected)
        - ✅ Works best with red/pink/yellow dragon fruits
        
        **Tips:**
        - ✅ Use good, even lighting
        - ✅ Center the fruit in frame
        - ✅ Avoid shadows and reflections
        - ✅ Keep camera steady
        """)

if __name__ == "__main__":
    main()
