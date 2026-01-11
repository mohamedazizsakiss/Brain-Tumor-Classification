import streamlit as st
import hashlib
import json
import time
import cv2
import numpy as np
import tempfile
import tensorflow as tf
import io
from PIL import Image, ImageOps

# --- IMPORTS FOR CUSTOM MODEL OBJECTS ---
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras import layers

# --- 1. BLOCKCHAIN BACKEND ---
class MedicalBlock:
    def __init__(self, index, timestamp, image_name, image_hash, previous_hash, prediction_result):
        self.index = index
        self.timestamp = timestamp
        self.image_name = image_name
        self.image_hash = image_hash
        self.previous_hash = previous_hash
        self.prediction_result = prediction_result 
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_data = self.__dict__.copy()
        if 'hash' in block_data: del block_data['hash']
        return hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()

class NeuroChain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = MedicalBlock(0, time.time(), "Genesis Block", "0", "0", "System Init")
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.compute_hash()
        self.chain.append(new_block)
        return True

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

if 'blockchain' not in st.session_state:
    st.session_state.blockchain = NeuroChain()

# --- 2. DEFINE CUSTOM LAYERS ---
@tf.keras.utils.register_keras_serializable()
def channel_attention(input_feature, ratio=4): # Ratio=4 matches your training
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = layers.Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)    
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    
    return layers.Multiply()([input_feature, cbam_feature])

# --- 3. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    custom_objs = {
        'channel_attention': channel_attention,
        'vgg_preprocess': vgg_preprocess,
        'efficientnet_preprocess': efficientnet_preprocess,
        'preprocess_input': efficientnet_preprocess, # Fix for EfficientNet loading
    }

    try:
        # Load your specific file
        model = tf.keras.models.load_model('brain_tumor_model.keras', custom_objects=custom_objs)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'brain_tumor_model.keras' is in the folder.")
        return None

model = load_model()

# Helper Functions
# --- Helper: Preprocess Image for AI ---
def preprocess_image(image, target_size=(224, 224)):
    # 1. Force conversion to RGB to ensure 3 channels (Fixes the shape error)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # 3. Convert to Array
    img_array = np.array(image)
    
    # 4. Batch Dimension -> Final Shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_frame(frame, target_size=(224, 224)):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, target_size)
    batch = np.expand_dims(resized, axis=0)
    return batch

# --- 4. MAIN DASHBOARD UI ---
st.set_page_config(page_title="NeuroChain AI", layout="wide", page_icon="🧠")

st.title("🧠 NeuroChain: Secure Brain Tumor Diagnostics")
st.markdown("### AI Ensemble Classification + Blockchain Data Integrity")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Create Tabs
tab_analysis, tab_ledger = st.tabs(["🔎 Universal Analysis (Image/Video)", "🔗 Ledger Audit"])

# === TAB 1: UNIVERSAL ANALYSIS (IMAGE + VIDEO) ===
with tab_analysis:
    st.subheader("Diagnostic Interface")
    st.write("Upload **any** media (MRI Scan or Video Sequence).")
    
    # 1. Update Uploader to accept BOTH Video and Image formats
    media_file = st.file_uploader("Upload File", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
    
    # Placeholders
    media_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Initialize Session State for 'Best Shot' logic
    if 'best_frame' not in st.session_state:
        st.session_state.best_frame = None
        st.session_state.best_conf = 0.0
        st.session_state.best_label = ""

    if media_file:
        file_type = media_file.name.split('.')[-1].lower()
        
        # --- PATH A: IMAGE HANDLING ---
        if file_type in ['jpg', 'jpeg', 'png']:
            image = Image.open(media_file)
            media_placeholder.image(image, caption="Uploaded Scan", use_container_width=True)
            
            if st.button("Analyze Image"):
                if model:
                    with st.spinner("Analyzing Scan..."):
                        # Process & Predict
                        processed_img = preprocess_image(image)
                        pred = model.predict(processed_img)
                        
                        idx = np.argmax(pred)
                        label = CLASS_NAMES[idx]
                        conf = np.max(pred) * 100
                        
                        # Store as "Best Frame" for Blockchain Evidence
                        st.session_state.best_frame = np.array(image) 
                        st.session_state.best_label = label
                        st.session_state.best_conf = conf
                        
                        # Display Result
                        color = "green" if label == "notumor" else "red"
                        result_placeholder.markdown(f"### Result: :{color}[{label.upper()}] ({conf:.2f}%)")

        # --- PATH B: VIDEO HANDLING ---
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(media_file.read())
            
            if st.button("Start Video Analysis"):
                cap = cv2.VideoCapture(tfile.name)
                
                # Reset tracking
                st.session_state.best_frame = None
                st.session_state.best_conf = 0.0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if model:
                        # Process
                        processed_frame = preprocess_frame(frame)
                        pred = model.predict(processed_frame, verbose=0)
                        
                        idx = np.argmax(pred)
                        label = CLASS_NAMES[idx]
                        conf = np.max(pred) * 100
                        
                        # Capture "Evidence" (Best tumor shot)
                        # Logic: If it's a tumor AND higher confidence than before -> Save it
                        if label != "notumor" and conf > st.session_state.best_conf:
                            st.session_state.best_conf = conf
                            st.session_state.best_label = label
                            st.session_state.best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Draw & Show
                        color = (0, 255, 0) if label == "notumor" else (0, 0, 255)
                        text = f"{label.upper()}: {conf:.1f}%"
                        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        media_placeholder.image(frame, channels="BGR", caption="Real-time Analysis")
                        
                        # Show Best Shot dynamically below video
                        if st.session_state.best_frame is not None:
                             result_placeholder.image(
                                st.session_state.best_frame, 
                                caption=f"⚠️ Evidence Capture: {st.session_state.best_label} ({st.session_state.best_conf:.1f}%)"
                            )
                
                cap.release()
                st.success("Analysis Complete.")

    # --- SHARED BLOCKCHAIN EVIDENCE SECTION ---
    # This works for BOTH Images and Videos because both save to 'st.session_state.best_frame'
    if st.session_state.best_frame is not None:
        st.divider()
        st.subheader("📂 Evidence Locker")
        
        col_ev, col_act = st.columns([1, 1])
        
        with col_ev:
            st.image(st.session_state.best_frame, caption="Captured Evidence", use_container_width=True)
            
        with col_act:
            st.write(f"**Condition:** {st.session_state.best_label}")
            st.write(f"**Confidence:** {st.session_state.best_conf:.2f}%")
            
            if st.button("🔒 Add to Blockchain Ledger"):
                # Convert Numpy Array -> Bytes
                # Force conversion to RGB to remove any Alpha channel (transparency)
                img_pil = Image.fromarray(st.session_state.best_frame).convert("RGB")
                buf = io.BytesIO()
                img_pil.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                
                # Hash & Create Block
                evidence_hash = hashlib.sha256(img_bytes).hexdigest()
                
                new_block = MedicalBlock(
                    index=len(st.session_state.blockchain.chain),
                    timestamp=time.time(),
                    image_name=f"Evidence_{int(time.time())}.jpg",
                    image_hash=evidence_hash,
                    previous_hash=st.session_state.blockchain.get_latest_block().hash,
                    prediction_result=f"{st.session_state.best_label} ({st.session_state.best_conf:.1f}%)"
                )
                
                if st.session_state.blockchain.add_block(new_block):
                    st.toast("Evidence securely hashed & stored!", icon="🔗")
                    st.success(f"Block #{new_block.index} added. Hash: {new_block.hash[:15]}...")

# === TAB 2: AUDIT LEDGER ===
with tab_ledger:
    st.subheader("Immutable Blockchain Ledger")
    
    if st.session_state.blockchain.is_chain_valid():
        st.success("Blockchain Integrity Verified: Valid ✅")
    else:
        st.error("SECURITY ALERT: Blockchain has been tampered with! ❌")
        
    # Table View of Blockchain
    block_data = []
    for block in st.session_state.blockchain.chain:
        block_data.append({
            "Index": block.index,
            "Timestamp": time.ctime(block.timestamp),
            "File": block.image_name,
            "Result": getattr(block, 'prediction_result', 'N/A'),
            "Hash": block.hash[:15] + "..."
        })
    
    st.table(block_data)
# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("NeuroChain Guide")
    
    st.info("""
    **✅ Supported Views:**
    * **Axial (Top-Down):** High Accuracy 🟢
    
    **⚠️ Experimental Views:**
    * **Sagittal (Side):** Lower Accuracy 🟠
    * **Coronal (Front):** Lower Accuracy 🟠
    """)
    
    st.warning("For best results, please upload Axial MRI scans standard for clinical screening.")