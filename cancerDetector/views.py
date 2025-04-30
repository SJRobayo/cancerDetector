import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings

# Rutas de los modelos / encoders
RF_MODEL_PATH   = os.path.join(settings.BASE_DIR, "model", "modelo_random_forest_optimizado.pkl")
ENCODERS_PATH   = os.path.join(settings.BASE_DIR, "model", "label_encoders.joblib")
CNN_MODEL_PATH  = os.path.join(settings.BASE_DIR, "model", "CNN.keras")  # NUEVO MODELO

# Cargar modelos
rf_model  = joblib.load(RF_MODEL_PATH)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)  # 👈 TensorFlow
encoders  = joblib.load(ENCODERS_PATH)

FEATURE_COLUMNS = [
    'relapse', 'tumor_size', 'Age', 'early_detection', 'cancer_stage',
    'obesity', 'Screening_History', 'Healthcare_Access', 'diet'
]

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def clasificar(prob):
    if prob is None:
        return "Desconocido"
    elif prob > 0.40:
        return "⚠️ Prioritario"
    elif prob > 0.20:
        return "🔍 Revisable"
    else:
        return "✔️ Normal"

def homepage(request):
    resultado = None
    predictions = None

    if request.method == 'POST' and 'csv_file' in request.FILES and 'image_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        img_file = request.FILES['image_file']

        csv_path = default_storage.save(f"uploads/{csv_file.name}", csv_file)
        img_path = default_storage.save(f"uploads/{img_file.name}", img_file)

        abs_csv_path = os.path.join(settings.MEDIA_ROOT, csv_path)
        abs_img_path = os.path.join(settings.MEDIA_ROOT, img_path)

        # CSV
        df = pd.read_csv(abs_csv_path)
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        X_new = df[FEATURE_COLUMNS].copy()
        rf_pred = rf_model.predict(X_new)

        try:
            rf_prob = rf_model.predict_proba(X_new)[:, 1]
        except AttributeError:
            rf_prob = [None] * len(rf_pred)

        # Imagen: predicción con CNN Keras
        image_data = preprocess_image(abs_img_path)
        cnn_prob = float(cnn_model.predict(image_data)[0][0])  # 👈 [0][0] si es array 2D

        # Añadir columnas
        df['Prediction_RF']  = rf_pred
        df['Prob_RF']        = [round(p, 4) if p is not None else None for p in rf_prob]
        df['Prob_CNN']       = round(cnn_prob, 4)
        df['Clasificación']  = df['Prob_RF'].apply(clasificar)

        predictions = df[['Prediction_RF', 'Prob_RF', 'Prob_CNN', 'Clasificación']].to_dict(orient='records')
        resultado = f"✅ Predicciones generadas para {len(df)} registro(s). Imagen analizada: {img_file.name}"

    return render(request, 'homepage.html', {
        'resultado': resultado,
        'predictions': predictions,
    })
