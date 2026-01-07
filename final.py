#%%Cell 1 - imports & config
import os
import random
import shutil
import math
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
# Configuration - EDIT THIS
POSITIVE_DIR = r"C:\Users\pruth\Downloads\project3\HAM_Dataset"   # <-- put your HAM10000 lesion images folder here (one folder with jpgs OR subfolders allowed)
AUTO_NEGATIVE_SOURCE_DIR = r"C:\Users\pruth\Downloads\negative" # directory containing non-skin example images you uploaded (the code will pick likely negative images here)
OUTPUT_DATA_DIR = "auto_skin_detector_dataset"
MODEL_OUTPUT_PATH = "models/skin_detector.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
RANDOM_SEED = 42

os.makedirs("models", exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# %%
# Cell 2 - build dataset
def collect_positive_images(pos_dir):
    # Accept images in subfolders or flat folder
    patterns = [os.path.join(pos_dir, "**", "*.jpg"),
                os.path.join(pos_dir, "**", "*.jpeg"),
                os.path.join(pos_dir, "**", "*.png")]
    files = []
    for p in patterns:
        files += glob(p, recursive=True)
    files = sorted(list(set(files)))
    if len(files) == 0:
        raise FileNotFoundError(f"No positive images found in POSITIVE_DIR: {pos_dir}")
    return files

def collect_negative_candidates(src_dir, min_count=20):
    # Find candidate images in the source dir, exclude known positive dirs
    patterns = [os.path.join(src_dir, "**", "*.jpg"),
                os.path.join(src_dir, "**", "*.jpeg"),
                os.path.join(src_dir, "**", "*.png")]
    files = []
    for p in patterns:
        files += glob(p, recursive=True)
    # exclude files deep inside our output dataset and model directories
    files = [f for f in files if "auto_skin_detector_dataset" not in f and "models" not in f]
    # remove duplicates and ensure file readable
    files = sorted(list(set(files)))
    # filter very small images
    cand = []
    for f in files:
        try:
            with Image.open(f) as im:
                if im.size[0] >= 128 and im.size[1] >= 128:
                    cand.append(f)
        except Exception:
            continue
    if len(cand) < min_count:
        print(f"Warning: only found {len(cand)} negative candidates in {src_dir}; augmentation will be used heavily.")
    return cand

def build_auto_dataset(pos_dir, neg_src_dir, out_dir, img_size=(224,224), neg_per_source=6):
    pos_files = collect_positive_images(pos_dir)
    neg_candidates = collect_negative_candidates(neg_src_dir)
    if len(neg_candidates) == 0:
        raise FileNotFoundError("No negative source images found in AUTO_NEGATIVE_SOURCE_DIR")

    # create folders
    pos_out = os.path.join(out_dir, "train", "pos")
    neg_out = os.path.join(out_dir, "train", "neg")
    val_pos_out = os.path.join(out_dir, "val", "pos")
    val_neg_out = os.path.join(out_dir, "val", "neg")
    for p in [pos_out, neg_out, val_pos_out, val_neg_out]:
        os.makedirs(p, exist_ok=True)

    # --- copy positives (80/20 split) ---
    random.shuffle(pos_files)
    split = int(0.8 * len(pos_files))
    train_pos = pos_files[:split]
    val_pos = pos_files[split:]
    # copy (or symlink) resized copies to out_dir
    def copy_and_resize(src_list, dest_folder):
        for i, src in enumerate(src_list):
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im = im.resize(img_size, Image.LANCZOS)
                    dest = os.path.join(dest_folder, f"pos_{i:05d}.jpg")
                    im.save(dest, quality=95)
            except Exception:
                continue

    copy_and_resize(train_pos, pos_out)
    copy_and_resize(val_pos, val_pos_out)

    # --- generate negatives by cropping & augmenting negative candidates ---
    idx = 0
    def save_neg_from_img(src_file, dest_folder, crops=6):
        nonlocal idx
        try:
            with Image.open(src_file) as im:
                im = im.convert("RGB")
                w,h = im.size
                # generate several random crops + resize
                for k in range(crops):
                    crop_w = int(w * random.uniform(0.3, 0.95))
                    crop_h = int(h * random.uniform(0.3, 0.95))
                    if crop_w < 64 or crop_h < 64:
                        continue
                    x = random.randint(0, max(0, w - crop_w))
                    y = random.randint(0, max(0, h - crop_h))
                    crop = im.crop((x,y,x+crop_w,y+crop_h)).resize(img_size, Image.LANCZOS)
                    # random flip/rotate
                    if random.random() < 0.5:
                        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                    if random.random() < 0.2:
                        crop = crop.rotate(random.choice([90,180,270]))
                    dest = os.path.join(dest_folder, f"neg_{idx:06d}.jpg")
                    crop.save(dest, quality=90)
                    idx += 1
        except Exception:
            pass

    # create train negatives by sampling each candidate and saving several crops
    # ensure roughly balanced: aim for #negatives ~= #positives_train
    target_negatives = max(len(train_pos), 200)
    i = 0
    while idx < target_negatives:
        src = random.choice(neg_candidates)
        save_neg_from_img(src, neg_out, crops=random.randint(2,6))
        i += 1
        if i > len(neg_candidates) * 10:
            break

    # produce a small val negative set
    idx_val = 0
    for src in neg_candidates[:min(30, len(neg_candidates))]:
        save_neg_from_img(src, val_neg_out, crops=2)
        idx_val += 1
        if idx_val > 200:
            break

    # final counts
    def count_files(folder):
        return len([p for p in glob(os.path.join(folder, "*")) if p.lower().endswith((".jpg",".png",".jpeg"))])

    print("Dataset built:")
    print("  train pos:", count_files(pos_out))
    print("  train neg:", count_files(neg_out))
    print("  val pos:  ", count_files(val_pos_out))
    print("  val neg:  ", count_files(val_neg_out))

    return out_dir

# build dataset
dataset_dir = build_auto_dataset(POSITIVE_DIR, AUTO_NEGATIVE_SOURCE_DIR, OUTPUT_DATA_DIR, img_size=IMG_SIZE)
#%%
# Cell 3 - data generators
train_dir = os.path.join(OUTPUT_DATA_DIR, "train")
val_dir = os.path.join(OUTPUT_DATA_DIR, "val")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.06,
    zoom_range=0.12,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=RANDOM_SEED
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    seed=RANDOM_SEED
)
#%%
# Cell 4 - tiny CNN
def make_tiny_cnn(input_shape=(224,224,3)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(96, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

skin_model = make_tiny_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
skin_model.summary()
#%%
# Cell 5 - compile & train
skin_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_loss', save_best_only=True, verbose=1)

history = skin_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[es, mc],
    verbose=2
)

# Save final weights also (ModelCheckpoint already saved best)
skin_model.save(MODEL_OUTPUT_PATH)
print("Saved skin detector to:", MODEL_OUTPUT_PATH)
#%%
# Cell 6 - evaluation and a few sample predictions
import matplotlib.pyplot as plt
plt.plot(history.history.get('loss', []), label='loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend()
plt.show()

# load best model
detector = tf.keras.models.load_model(MODEL_OUTPUT_PATH, compile=False)

def detect_skin_lesion_with_model(image_path, detector, img_size=IMG_SIZE, threshold=0.5):
    img = Image.open(image_path).convert("RGB").resize(img_size, Image.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(arr, 0)
    p = float(detector.predict(x)[0][0])
    is_skin = p >= threshold
    return is_skin, p

# try on a few images from AUTO_NEGATIVE_SOURCE_DIR and a positive
neg_samples = [p for p in glob(os.path.join(AUTO_NEGATIVE_SOURCE_DIR, "**", "*.jpg"), recursive=True)][:6]
print("Negative samples check:")
for p in neg_samples:
    try:
        ok, prob = detect_skin_lesion_with_model(p, detector)
        print("  ", os.path.basename(p), "->", ok, f"({prob:.3f})")
    except Exception:
        pass

# pick one positive for a sanity check (if exists)
pos_list = collect_positive_images(POSITIVE_DIR)
if pos_list:
    print("Positive sample check:")
    for p in pos_list[:3]:
        ok, prob = detect_skin_lesion_with_model(p, detector)
        print("  ", os.path.basename(p), "->", ok, f"({prob:.3f})")
#%%
# Cell 7 - integration wrapper (paste into your inference file)
from tensorflow.keras.models import load_model

SKIN_DETECTOR_PATH = MODEL_OUTPUT_PATH  # "models/skin_detector.h5"
skin_detector = load_model(SKIN_DETECTOR_PATH, compile=False)

def is_skin_lesion_model(image_path, detector=skin_detector, img_size=IMG_SIZE, threshold=0.5):
    """Returns (accepted_bool, prob_float, reason_str)"""
    try:
        img = Image.open(image_path).convert("RGB").resize(img_size, Image.LANCZOS)
    except Exception as e:
        return False, 0.0, f"cannot_open_image: {e}"
    arr = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(arr, 0)
    p = float(detector.predict(x)[0][0])
    if p >= threshold:
        return True, p, f"model_accepted (p={p:.3f})"
    else:
        return False, p, f"model_rejected (p={p:.3f})"
#%%
# Cell 8 - single image check (end-to-end)
TEST_IMAGE_PATH = r"C:\Users\pruth\Downloads\project3\HAM_Dataset\MEL\ISIC_0033655.jpg" or other
accepted, prob, reason = is_skin_lesion_model(TEST_IMAGE_PATH, detector=skin_detector, img_size=IMG_SIZE, threshold=0.5)
print("Detector result:", accepted, prob, reason)
if not accepted:
    print("Image rejected as non-skin. STOP.")
else:
    # call your predict_single_image or hybrid inference here
    res = predict_single_image(TEST_IMAGE_PATH, vgg_model, densenet_model, vgg_weight=0.5, verbose=True, save_debug=True)
    print("Hybrid result:", res["predicted_class"], res["predicted_prob"])
#%%
# Cell A - Load your trained lesion models
from tensorflow.keras.models import load_model

VGG_MODEL_PATH = r"C:\Users\pruth\Downloads\FINAL PROJECT\skin_cancer_vgg16_model.h5"
DENSENET_MODEL_PATH = r"C:\Users\pruth\Downloads\project3\densenet_model.h5"# <-- YOUR PATH

vgg_model = load_model(VGG_MODEL_PATH, compile=False)
densenet_model = load_model(DENSENET_MODEL_PATH, compile=False)
print("Loaded VGG16 + DenseNet121 successfully.")
#%%
# Cell B - class labels (edit to match your model)
CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]
DEFAULT_VGG_WEIGHT = 0.5
#%%
# Cell C - preprocessing for vgg + densenet
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.densenet import preprocess_input as dn_pre

def preprocess_vgg(img):
    img = img.resize((224,224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr,0)
    return vgg_pre(arr)

def preprocess_dn(img):
    img = img.resize((224,224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr,0)
    return dn_pre(arr)
#%%
# Cell D - full hybrid classifier
def predict_single_image(image_path, vgg_model, densenet_model, vgg_weight=DEFAULT_VGG_WEIGHT,
                         verbose=True, save_debug=False):

    # 1. Skin detector check (from Cell 7)
    accepted, prob, reason = is_skin_lesion_model(
        image_path, detector=skin_detector, img_size=IMG_SIZE, threshold=0.5
    )
    if not accepted:
        print("❌ REJECTED:", reason)
        return {"accepted": False, "reject_reason": reason}

    # 2. Load image
    img = Image.open(image_path).convert("RGB")

    # 3. Preprocess for each model
    x_vgg = preprocess_vgg(img)
    x_dn  = preprocess_dn(img)

    # 4. Predict
    v = vgg_model.predict(x_vgg)[0]
    d = densenet_model.predict(x_dn)[0]

    # 5. Convert logits → probabilities if needed
    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    if v.max() > 1 or d.max() > 1:
        v = softmax(v)
        d = softmax(d)

    # 6. Hybrid combination
    h = vgg_weight * v + (1 - vgg_weight) * d
    idx = np.argmax(h)
    cls = CLASS_NAMES[idx]

    return {
        "accepted": True,
        "predicted_class": cls,
        "predicted_prob": float(h[idx]),
        "vgg_probs": v,
        "densenet_probs": d,
        "hybrid_probs": h,
    }
#%%
# Cell 8 - final test
TEST_IMAGE_PATH = r"C:\Users\pruth\Downloads\negative\hand.png"

accepted, prob, reason = is_skin_lesion_model(TEST_IMAGE_PATH, detector=skin_detector)
print("Detector result:", accepted, prob, reason)

if accepted:
    res = predict_single_image(TEST_IMAGE_PATH, vgg_model, densenet_model)
    print("Hybrid:", res["predicted_class"], res["predicted_prob"])
else:
    print("Image rejected:", reason)
#%%import tensorflow as tf
from tensorflow.keras.models import Model 

# The trained model object from your Cell 4/5 is named 'skin_model'
# Use the correct object name here:
my_trained_model = skin_model 

# Define the path where you want to save the model
MODEL_PATH = "my_trained_model.h5" # You can use your original path: "models/skin_detector.h5"

# Save the model
my_trained_model.save(MODEL_PATH)
print(f"✅ Model saved successfully to: {MODEL_PATH}")
#%%
