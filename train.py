# train.py
# -------------------------------
# Install compatible versions in requirements.txt
# -------------------------------

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from flask import Flask, jsonify

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR = "./mushroom_dataset"  # relative path for Render
CLASSES = [
    "contamination_bacterialblotch",
    "contamination_cobweb",
    "contamination_greenmold",
    "healthy_bag",
    "healthy_mushroom"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_STATE = 42

# -------------------------------
# Training and Feature Extraction
# -------------------------------
def main():
    print("Starting training pipeline...")

    # Load dataset
    data = []
    for cls in CLASSES:
        folder = os.path.join(BASE_DIR, cls)
        for file in os.listdir(folder):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                data.append([os.path.join(folder, file), cls])
    df = pd.DataFrame(data, columns=["path", "label"])
    print(df.head(), "\nTotal images:", len(df))

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE)
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    # Image generators
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=efficientnet_preprocess
    )
    val_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
    test_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="path", y_col="label",
        class_mode="categorical",
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col="path", y_col="label",
        class_mode="categorical",
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col="path", y_col="label",
        class_mode="categorical",
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Feature extractor
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base_model.trainable = False
    inp = tf.keras.Input(shape=(224,224,3))
    x = base_model(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(inputs=inp, outputs=x)
    print("Feature extractor ready.")

    # Feature extraction function
    def extract_all(gen, extractor):
        feats, labels = [], []
        for x, y in gen:
            f = extractor.predict(x, verbose=0)
            feats.append(f)
            labels.append(np.argmax(y, axis=1))
            if len(feats) * gen.batch_size >= gen.n:
                break
        return np.vstack(feats), np.hstack(labels)

    X_train, y_train = extract_all(train_gen, feature_extractor)
    X_val, y_val = extract_all(val_gen, feature_extractor)
    X_test, y_test = extract_all(test_gen, feature_extractor)
    print("Train features shape:", X_train.shape)

    # Random Forest classifier
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
    cal_rf.fit(X_val, y_val)

    val_probs = cal_rf.predict_proba(X_val)
    val_maxp = np.max(val_probs, axis=1)
    THRESHOLD = np.mean(val_maxp) - 2*np.std(val_maxp)
    print("Robust threshold for not_mushroom:", THRESHOLD)

    mean_vec = np.mean(X_train, axis=0)
    cov_mat = np.cov(X_train.T)
    inv_cov = np.linalg.pinv(cov_mat)
    print("Mahalanobis distance ready.")

    # Save models for later use if needed
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_extractor, "models/feature_extractor.pkl")
    joblib.dump(cal_rf, "models/cal_rf.pkl")
    joblib.dump({"mean": mean_vec, "inv_cov": inv_cov, "threshold": THRESHOLD}, "models/mahalanobis.pkl")

    print("Training completed and models saved!")

# -------------------------------
# Flask Web Service
# -------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return "Mushroom training service is running!"

@app.route("/train", methods=["POST"])
def train_endpoint():
    try:
        main()
        return jsonify({"status": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
