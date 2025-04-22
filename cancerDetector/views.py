from django.shortcuts import render
from django.core.files.storage import default_storage
import os

def homepage(request):
    resultado = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            file_path = default_storage.save(f'uploads/{csv_file.name}', csv_file)
            resultado = f"✅ CSV subido correctamente: {csv_file.name}"

        elif 'image_file' in request.FILES:
            image_file = request.FILES['image_file']
            file_path = default_storage.save(f'uploads/{image_file.name}', image_file)
            resultado = f"🖼️ Imagen subida correctamente: {image_file.name}"

    return render(request, 'homepage.html', {
        'resultado': resultado
    })
