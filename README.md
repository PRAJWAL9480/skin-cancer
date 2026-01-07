
# Melanoma Spotter – Skin Cancer Detection System

Melanoma Spotter is a deep learning–based web application for automated
skin cancer detection using a hybrid ensemble of VGG16 and DenseNet121
models. The system also includes a CNN-based skin lesion validation
step to reject non-skin images before classification.

---

## Project Overview

The project performs skin cancer classification in the following stages:

1. User uploads an image through a web interface
2. Flask backend receives the image
3. Skin lesion detector validates the image
4. VGG16 and DenseNet121 models run in parallel
5. Ensemble learning combines predictions
6. Final class and confidence score are returned

---

## Execution Flow


---

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- HTML, CSS, JavaScript
- NumPy
- Pillow (PIL)

---

## Files Description (As Uploaded)

| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Flask server entry point |
| `inference.py` | Inference router |
| `final_stub.py` | Core model loading & prediction logic |
| `*.ipynb` | Training and experimentation notebooks |
| `*.h5` | Trained deep learning models |
| `index.html` | Frontend UI |
| `script.js` | Frontend interaction logic |

> Note: Files may be placed in different locations but the project runs
> using relative imports and internal path handling.

---

## System Requirements

- Python 3.8 – 3.10
- Minimum 8 GB RAM
- Internet connection (for installing dependencies)

---

## How to Execute the Project (From Scratch)

### Step 1: Install Python
Check Python:
```bash
python --version

### Step 2: Open Project Folder

Extract or open the folder containing all project files.
### Step 3: Create Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


Linux / macOS

python3 -m venv venv
source venv/bin/activate
### Step 4: Install Dependencies

If requirements.txt exists:

pip install -r requirements.txt


Otherwise:

pip install flask flask-cors tensorflow numpy pillow matplotlib
### Step 5: Ensure Model Files Are Present

Make sure the trained model files are available:

skin_detector.h5

skin_cancer_vgg16_model.h5

densenet_model.h5

These files must remain in the same locations as provided.
### Step 6: Start Flask Server

Run:

python app.py


Expected output:

Running on http://127.0.0.1:5000/

### Step 7: Open Web Application

Open a browser and visit:

http://127.0.0.1:5000

### Step 8: Run Prediction

Upload a skin lesion image

Click Analyze / Upload

View predicted cancer class and confidence score
