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


Install dependencies:

pip install -r requirements.txt


Update the DRIVE_PROJECT_PATH in the notebook to point to your dataset location.

Run the Jupyter Notebook brain_tumor_finetune.ipynb.

📂 Dataset

The dataset used is the Brain Tumor MRI Dataset available on Kaggle.

Total Images: ~3,000+

Format: JPG (converted to RGB during preprocessing)

📜 License

This project is licensed under the MIT License.

## 🔒 NEW: Blockchain Security Layer (Added Jan 2026)
To prevent **Data Poisoning** attacks (where hackers alter MRI scans to fool the AI), I engineered a cryptographic verification system.

* **Technology:** Custom Python Blockchain + SHA-256 Hashing.
* **Function:** Creates an immutable ledger of all authorized MRI scans.
* **Defense:** Automatically flags tampered images *before* they reach the diagnosis model.

**How to run the security dashboard:**
```bash
cd security_dashboard
pip install -r requirements.txt
streamlit run app.py
