# final_stub.py -- safe inference stub created from your notebook (no training)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.densenet import preprocess_input as dn_pre

BASE_DIR = os.path.dirname(__file__)

# Preference: models in flask_api/models
MODEL_DIR = os.path.join(BASE_DIR, 'flask_api', 'models')

# Fallback absolute paths you used in your notebook
FALLBACK_VGG = r'C:\\Users\\pruth\\Downloads\\FINAL PROJECT\\skin_cancer_vgg16_model.h5'
FALLBACK_DN = r'C:\\Users\\pruth\\Downloads\\project3\\densenet_model.h5'
FALLBACK_SKIN = os.path.join(BASE_DIR, 'models', 'skin_detector.h5')

VGG_CANDIDATES = [
    os.path.join(MODEL_DIR, 'skin_cancer_vgg16_model.h5'),
    os.path.join(MODEL_DIR, 'skin_cancer_vgg16_model.keras'),
    FALLBACK_VGG
]
DN_CANDIDATES = [
    os.path.join(MODEL_DIR, 'densenet_model.h5'),
    os.path.join(MODEL_DIR, 'densenet_model.keras'),
    FALLBACK_DN
]
SKIN_CANDIDATES = [
    os.path.join(MODEL_DIR, 'skin_detector.h5'),
    os.path.join(MODEL_DIR, 'skin_detector.keras'),
    FALLBACK_SKIN
]

def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

VGG_PATH = first_existing(VGG_CANDIDATES)
DN_PATH = first_existing(DN_CANDIDATES)
SKIN_PATH = first_existing(SKIN_CANDIDATES)

print('final_stub: looking for models...')
print('  VGG_PATH =', VGG_PATH)
print('  DN_PATH  =', DN_PATH)
print('  SKIN_PATH=', SKIN_PATH)

vgg_model = None
densenet_model = None
skin_detector = None

if VGG_PATH:
    try:
        vgg_model = load_model(VGG_PATH, compile=False)
        print('final_stub: loaded vgg_model')
    except Exception as e:
        print('final_stub: failed to load vgg_model:', e)

if DN_PATH:
    try:
        densenet_model = load_model(DN_PATH, compile=False)
        print('final_stub: loaded densenet_model')
    except Exception as e:
        print('final_stub: failed to load densenet_model:', e)

if SKIN_PATH:
    try:
        skin_detector = load_model(SKIN_PATH, compile=False)
        print('final_stub: loaded skin_detector')
    except Exception as e:
        print('final_stub: failed to load skin_detector:', e)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DEFAULT_VGG_WEIGHT = 0.5

def preprocess_vgg(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, 0)
    return vgg_pre(arr)

def preprocess_dn(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, 0)
    return dn_pre(arr)

def is_skin_lesion_model(image_path, detector=skin_detector, img_size=IMG_SIZE, threshold=0.5):
    if detector is None:
        return False, 0.0, 'no_skin_detector_loaded'
    try:
        img = Image.open(image_path).convert('RGB').resize(img_size, Image.LANCZOS)
    except Exception as e:
        return False, 0.0, f'cannot_open_image: {e}'
    arr = np.asarray(img).astype('float32') / 255.0
    x = np.expand_dims(arr, 0)
    p = float(detector.predict(x)[0][0])
    if p >= threshold:
        return True, p, f'model_accepted (p={p:.3f})'
    else:
        return False, p, f'model_rejected (p={p:.3f})'

def predict_single_image(image_path, vgg_model_override=None, densenet_model_override=None, vgg_weight=DEFAULT_VGG_WEIGHT):
    vm = vgg_model_override if vgg_model_override is not None else vgg_model
    dm = densenet_model_override if densenet_model_override is not None else densenet_model
    if vm is None or dm is None:
        raise RuntimeError('vgg_model or densenet_model not loaded')

    accepted, prob, reason = is_skin_lesion_model(image_path)
    if not accepted:
        return {'accepted': False, 'reject_reason': reason}

    img = Image.open(image_path).convert('RGB')
    x_vgg = preprocess_vgg(img)
    x_dn = preprocess_dn(img)

    v = vm.predict(x_vgg)[0]
    d = dm.predict(x_dn)[0]

    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    if v.max() > 1 or d.max() > 1:
        v = softmax(v)
        d = softmax(d)

    h = vgg_weight * v + (1 - vgg_weight) * d
    idx = int(np.argmax(h))
    cls = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)

    return {
        'accepted': True,
        'predicted_class': cls,
        'predicted_prob': float(h[idx]),
        'vgg_probs': v.tolist(),
        'densenet_probs': d.tolist(),
        'hybrid_probs': h.tolist()
    }
