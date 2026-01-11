import streamlit as st
import hashlib
import json
import time
import cv2
import numpy as np
import tempfile
import tensorflow as tf
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
        self.prediction_result = prediction_result # Added to store medical result in block
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
        # Genesis block has no previous hash
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

# Initialize Chain in Session State
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = NeuroChain()

# --- 2. DEFINE CUSTOM LAYERS (REQUIRED FOR LOADING) ---
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
    # Helper dictionary for all custom parts of your model
    custom_objs = {
        'channel_attention': channel_attention,
        'vgg_preprocess': vgg_preprocess,
        'efficientnet_preprocess': efficientnet_preprocess
    }

    try:
        # Load the ENSEMBLE model
        model = tf.keras.models.load_model('brain_tumor_model.keras', custom_objects=custom_objs)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'ensemble_vgg_eff_best.keras' is in the folder.")
        return None

model = load_model()

# Helper: Preprocess Image for AI
def preprocess_image(image, target_size=(224, 224)):
    # 1. Resize
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # 2. Convert to Array
    img_array = np.array(image)
    
    # 3. Batch Dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # NOTE: No "/ 255.0" here because your model has VGG/EffNet preprocessing built-in!
    return img_array

# Helper: Preprocess Video Frame
def preprocess_frame(frame, target_size=(224, 224)):
    # 1. Convert OpenCV BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Resize
    resized = cv2.resize(rgb_frame, target_size)
    
    # 3. Batch Dimension (No /255.0)
    batch = np.expand_dims(resized, axis=0)
    return batch

# --- 4. MAIN DASHBOARD UI ---
st.set_page_config(page_title="NeuroChain AI", layout="wide", page_icon="🧠")

st.title("🧠 NeuroChain: Secure Brain Tumor Diagnostics")
st.markdown("### AI Ensemble Classification + Blockchain Data Integrity")

# Class Labels (Alphabetical order is standard for Keras 'from_directory')
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Image & Security", "🎥 Video Analysis", "🔗 Ledger Audit"])

# === TAB 1: IMAGE + BLOCKCHAIN ===
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload MRI Scan")
        uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Scan", use_column_width=True)
            
            # 1. Compute Hash for Security
            img_bytes = uploaded_file.getvalue()
            current_hash = hashlib.sha256(img_bytes).hexdigest()
            st.info(f"🔒 SHA-256 Signature: {current_hash[:10]}...")

            # 2. Analyze Button
            if st.button("Analyze & Secure Data"):
                if model:
                    with st.spinner("Running Ensemble Model..."):
                        processed_img = preprocess_image(image)
                        pred = model.predict(processed_img)
                        
                        result_index = np.argmax(pred)
                        result_label = CLASS_NAMES[result_index]
                        confidence = np.max(pred) * 100
                    
                    # Display Result
                    if result_label == 'notumor':
                        st.success(f"**Prediction:** {result_label.upper()} ({confidence:.2f}%)")
                    else:
                        st.error(f"**Prediction:** {result_label.upper()} ({confidence:.2f}%)")
                    
                    # Add to Blockchain
                    new_block = MedicalBlock(
                        index=len(st.session_state.blockchain.chain),
                        timestamp=time.time(),
                        image_name=uploaded_file.name,
                        image_hash=current_hash,
                        previous_hash=st.session_state.blockchain.get_latest_block().hash,
                        prediction_result=f"{result_label} ({confidence:.1f}%)"
                    )
                    st.session_state.blockchain.add_block(new_block)
                    st.toast("Block added to NeuroChain ledger!", icon="✅")

# === TAB 2: VIDEO ANALYSIS ===
with tab2:
    st.subheader("Real-Time Video Diagnostics")
    st.write("Upload an MRI sequence to detect tumors frame-by-frame.")
    
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if model:
                # Process and Predict
                processed_frame = preprocess_frame(frame)
                pred = model.predict(processed_frame, verbose=0)
                
                idx = np.argmax(pred)
                label = CLASS_NAMES[idx]
                conf = np.max(pred) * 100
                
                # Draw on frame
                color = (0, 255, 0) if label == "notumor" else (0, 0, 255)
                text = f"{label.upper()}: {conf:.1f}%"
                
                cv2.putText(frame, text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Show in Streamlit (Convert BGR to RGB)
                st_frame.image(frame, channels="BGR")
        
        cap.release()

# === TAB 3: AUDIT LEDGER ===
with tab3:
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
            "Result": getattr(block, 'prediction_result', 'N/A'), # Handle genesis block
            "Hash": block.hash[:15] + "..."
        })
    
    st.table(block_data)