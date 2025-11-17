Drive Link : https://drive.google.com/drive/folders/1a5yyuzkSfjJyllESqWTf-fkCa2-UNHwh?usp=sharing

---

# 🖼️ Tamil Painting Damage Restoration & Classification

### **Deep Learning + Machine Learning | VGG16 Features | Image Inpainting**

This project focuses on **restoring damaged Tamil paintings** and **classifying them** based on extracted deep features.
It uses **OpenCV inpainting**, **VGG16 feature extraction**, and **Random Forest classification** to build an end-to-end pipeline for artwork digitization and preservation.

---

## 🚀 Features

### ✔ **1. Damage Simulation**

Automatically applies artificial damage to clean paintings:

* Gaussian blur patches
* Pixelated blocks
* Random distortions / noise

Used for training and testing restoration quality.

### ✔ **2. Image Restoration (Inpainting)**

Damaged paintings are repaired using:

* Telea Algorithm (OpenCV)
* Neighborhood-based pixel reconstruction

Restores missing or distorted regions.

### ✔ **3. Deep Feature Extraction (VGG16)**

Pretrained VGG16 (without top layers) extracts 512-dimensional feature vectors representing:

* Texture
* Color tone
* Artistic patterns
* Structural attributes

Saved into `features_vgg16.csv`.

### ✔ **4. Machine Learning Classification**

A Random Forest classifier predicts one of four painting categories:

* **Black**
* **Red**
* **White**
* **Other**

Includes:

* Label encoding
* Feature scaling
* Hyperparameter tuning (optional)

### ✔ **5. Evaluation Metrics & Graphs**

The project generates:

* Confusion Matrix
* Precision, Recall, F1 Score
* Accuracy & Error Rate
* Training vs Validation Accuracy Graph
* Training vs Validation Loss Graph

### ✔ **6. Custom Prediction Script**

Loads a CSV file of features and returns:

* Predicted label
* Regression output (optional)
* Results saved as `predictions.csv`

### ✔ **7. Simple Streamlit UI**

A minimal UI to:

* Upload a damaged painting
* Restore the image
* Compare original vs restored
* Download restored image
* Display classification results

---

## 📂 Project Structure

```
tamil-painting-project/
│── all_images/
│   ├── images/            # Original clean images
│   ├── damaged/           # Damaged dataset
│   ├── restored/          # Restored images
│
│── features/
│   ├── features_vgg16.csv # Deep extracted features
│
│── models/
│   ├── classification_model.pkl
│   ├── class_scaler.pkl
│   ├── label_encoder.pkl
│   ├── regression_model.pkl (optional)
│   ├── reg_scaler.pkl (optional)
│
│── extract_features_vgg16.py
│── train_classification.py
│── train_regression.py
│── predict_custom.py
│── damage_generator.py
│── app.py  # Streamlit UI
│
│── README.md
```

---

## 🛠 Installation


### **1. Install Requirements**

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### **1. Generate Damaged Images**

```bash
python damage_generator.py
```

### **2. Extract VGG16 Features**

```bash
python extract_features_vgg16.py
```

### **3. Train Classification Model**

```bash
python train_classification.py
```

### **4. Run Prediction**

```bash
python predict_custom.py
```

### **5. Launch Streamlit App**

```bash
streamlit run app.py
```

---

## 📊 Sample Results

### **Classification Report**

* Accuracy: ~75–100% (dataset dependent)
* Supports error rate calculation
* Confusion matrix included

### **Restoration Example**

| Before (Damaged)  | After (Restored) |
| ----------------- | ---------------- |
| ✔ Pixelated       | ✔ Clear texture  |
| ✔ Blurred patches | ✔ Inpainted      |
| ✔ Noise           | ✔ Smooth output  |

---

## 🧰 Tech Stack

* **Python**
* **OpenCV**
* **TensorFlow / Keras (VGG16)**
* **Scikit-Learn**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**
* **Streamlit**

---

## 🎯 Project Goals

This project bridges **digital art preservation** and **machine intelligence**, offering:

* Automated restoration
* Deep feature extraction
* ML-based artwork classification
* Dataset generation and evaluation
* Visualization-ready results for academic reports

---

## ⭐ Contribute

Pull requests are welcome!
If you'd like new features (CNN classifier, GAN restoration, UI improvements), feel free to ask.


