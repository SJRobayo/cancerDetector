import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
import joblib  # o keras.models.load_model si es una CNN de Keras

# Ruta del modelo CNN
CNN_MODEL_PATH = os.path.join(settings.BASE_DIR, "model", "CNN_modelo.pkl")

# Cargar modelo CNN al arrancar el servidor
cnn_model = joblib.load(CNN_MODEL_PATH)

def homepage(request):
    resultado = None
    predictions = None

    if request.method == 'POST':
        # Guardar imagen
        image_file = request.FILES.get('image_file')

        if not image_file:
            resultado = "❗ Debes subir una imagen."
        else:
            img_path = default_storage.save(f'uploads/{image_file.name}', image_file)
            img_abs_path = os.path.join(settings.MEDIA_ROOT, img_path)

            # Procesar imagen
            img = Image.open(img_abs_path).convert('RGB')
            img = img.resize((224, 224))  # ajustamos tamaño a lo que espera la CNN
            arr = np.array(img, dtype='float32') / 255.0
            arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

            # Predicción CNN
            cnn_out = cnn_model.predict(arr)
            cnn_prob = float(cnn_out[0][0]) if cnn_out.shape[-1] == 1 else float(cnn_out[0, 1])
            cnn_pred = int(round(cnn_prob))

            resultado = "✅ Predicción completada."
            predictions = {
                "cnn_pred": cnn_pred,
                "cnn_prob": round(cnn_prob, 4)
            }

    return render(request, 'homepage.html', {
        'resultado': resultado,
        'predictions': predictions,
    })
