🧠 Brain Tumor Classification using Ensemble Deep Learning & Attention Mechanism

📌 Project Overview

This project focuses on the multi-class classification of Brain Tumors from MRI images. Diagnosing brain tumors early and accurately is critical for treatment planning.

This solution leverages Transfer Learning and Ensemble methods to classify MRI scans into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

🚀 Methodology

The project compares three individual state-of-the-art architectures and proposes a novel Parallel Ensemble Model.

1. Single Models (Baseline)

We utilized three pre-trained models as feature extractors, enhancing them with a custom Channel Attention block before the classification head:

ResNet50

VGG16

EfficientNetB0

2. Proposed Architecture: Parallel Ensemble with Attention

To improve classification performance, particularly between clinically similar tumor types, we developed a fusion architecture:

Dual-Branch Feature Extraction: Runs VGG16 and ResNet50 in parallel on the same input.

Channel Attention Module: Applied to each branch independently to weigh important feature maps (Squeeze-and-Excitation).

Feature Fusion: The refined features from both models are concatenated.

2-Stage Training Strategy: 1.  Frozen Training: Training only the custom head while backbones are frozen.
 2.  Fine-Tuning: Unfreezing the top layers of both backbones and training with a low learning rate (1e-5) to adapt generic features to medical nuances.


🛠️ Techniques Used

Data Handling: Automatic RGB conversion for grayscale MRIs.

Class Imbalance: Computed and applied class_weights to penalize misclassifications of minority classes.

Regularization: Used Dropout and EarlyStopping to prevent overfitting.

Optimization: ModelCheckpoint to save and restore the best model weights based on validation loss.

📊 Results & Analysis

We evaluated models using Confusion Matrices to visualize misclassifications. The Ensemble model demonstrated superior performance in distinguishing between Glioma and Meningioma, which was a primary challenge for individual models.


📦 How to Run

Clone the repository:

git clone [https://github.com/your-username/brain-tumor-ensemble.git](https://github.com/your-username/brain-tumor-ensemble.git)


### 2. 🔗 Blockchain Security (NeuroChain Ledger)
* **Tamper-Proof Records:** Every diagnosis is hashed (SHA-256) and stored in a local blockchain ledger.
* **Evidence Locker:** Automatically captures the "Best Shot" (highest confidence frame) from videos and secures it on-chain.
* **Audit Trail:** Verify the integrity of past diagnoses via the Ledger Audit tab.

### 3. 🎥 Universal Analysis
* **Multi-Format Support:** Analyze static **MRI Images** (JPG, PNG) or full **Video Sequences** (MP4, AVI).
* **Real-Time Tracking:** Processes video feeds frame-by-frame to identify tumor presence dynamically.

---

## ⚠️ Important: Setup Instructions

Because the AI model file is large (>100MB), it cannot be hosted directly on GitHub. You must download it separately.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/mohamedazizsakiss/Brain-Tumor-Classification.git](https://github.com/mohamedazizsakiss/Brain-Tumor-Classification.git)
    cd Brain-Tumor-Classification
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **🔴 Download the Model (CRITICAL)**
    * Download the trained model file from **[https://drive.google.com/file/d/13MTbMXfZTbAoniGUT4H8DGowTdnrTCen/view?usp=sharing]**.
    * **Rename the file** to: `brain_tumor_model.keras`
    * **Move it** into the root folder of this project.

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV (cv2), PIL
* **Security:** Python `hashlib` (SHA-256 Blockchain implementation)
* **Data Handling:** NumPy, Pandas

---

<<<<<<< HEAD
## 📸 Usage Guide

### Tab 1: Universal Analysis
* Upload an MRI Image or Video.
* The system preprocesses the input (resizing, normalization).
* **For Images:** Instant classification with confidence score.
* **For Videos:** The AI scans every frame and automatically captures the "Best Evidence" frame where the tumor is most visible.

### Tab 2: Ledger Audit
* View the history of all secure transactions.
* Check the cryptographic hashes to ensure data hasn't been altered.
* Status indicator shows if the Blockchain is **Valid ✅** or **Tampered ❌**.

---

## ⚠️ Limitations & Disclaimer

* **Viewpoint Sensitivity:** This model is trained primarily on **Axial (Top-Down)** MRI scans.
    * ✅ **Axial:** High Accuracy
    * ⚠️ **Sagittal (Side) / Coronal (Front):** May yield lower accuracy or false positives due to domain shift.
* **Educational Use:** This project is a prototype for educational purposes and should not be used as a standalone tool for medical diagnosis without clinical validation.

---

### 👨‍💻 Author
**Mohamed Aziz Sakiss**
* *AI Engineering Student @ EPI Digital School*
* [GitHub Profile](https://github.com/mohamedazizsakiss)
=======
**How to run the security dashboard:**
```bash
cd security_dashboard
pip install -r requirements.txt
streamlit run app.py
>>>>>>> ff92661262fbf1d1ec875efbfd9b3f2ec3f51887
