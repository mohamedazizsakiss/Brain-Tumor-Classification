import streamlit as st
import hashlib
import json
import time
from PIL import Image
import io

# --- BLOCKCHAIN BACKEND ---
class MedicalBlock:
    def __init__(self, index, timestamp, image_name, image_hash, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.image_name = image_name
        self.image_hash = image_hash
        self.previous_hash = previous_hash
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
        self.chain.append(MedicalBlock(0, time.time(), "GENESIS", "0", "0"))

    def add_image(self, name, image_bytes):
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        last_block = self.chain[-1]
        new_block = MedicalBlock(len(self.chain), time.time(), name, img_hash, last_block.hash)
        self.chain.append(new_block)
        return new_block, img_hash

    def is_valid(self, image_bytes):
        query_hash = hashlib.sha256(image_bytes).hexdigest()
        for block in self.chain:
            if block.image_hash == query_hash:
                return True, block
        return False, None

# --- APP INTERFACE ---
if 'neuro_chain' not in st.session_state:
    st.session_state.neuro_chain = NeuroChain()

st.set_page_config(page_title="NeuroScout Security", page_icon="🛡️")
st.title("🛡️ NeuroScout: Blockchain Security Layer")

# Sidebar
st.sidebar.header("System Status")
st.sidebar.metric("Blocks Mined", len(st.session_state.neuro_chain.chain))
st.sidebar.success("● Network Online (Local)")

# 1. Secure Data
st.subheader("1. 📤 Secure New MRI Scan")
uploaded_file = st.file_uploader("Upload MRI to secure", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(uploaded_file, width=200)
    if st.button("🔒 Seal & Add to Blockchain"):
        block, h = st.session_state.neuro_chain.add_image(uploaded_file.name, bytes_data)
        st.success(f"Success! Secured in Block #{block.index}")
        st.code(f"Hash: {h}")

# 2. Verify Data
st.markdown("---")
st.subheader("2. 🕵️ Validate Image Integrity")
check_file = st.file_uploader("Upload MRI for verification", type=['jpg', 'png'], key="check")

if check_file is not None:
    check_bytes = check_file.getvalue()
    is_safe, found_block = st.session_state.neuro_chain.is_valid(check_bytes)
    if is_safe:
        st.success("✅ **VERIFIED AUTHENTIC**")
        st.write(f"Matches Block #{found_block.index}")
    else:
        st.error("❌ **TAMPERING DETECTED**")
        st.write("This file does not match any digital fingerprint in the ledger.")

# Ledger
st.markdown("---")
st.subheader("🔗 Live Blockchain Ledger")
chain_data = [{"Index": b.index, "Hash": b.hash[:15]+"...", "Img Hash": b.image_hash[:15]+"..."} for b in st.session_state.neuro_chain.chain]
st.table(chain_data)