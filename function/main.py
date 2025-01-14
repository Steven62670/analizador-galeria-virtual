import os
from google.cloud import storage, firestore, vision
from flask import jsonify

# Inicializar los clientes de GCP
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
firestore_client = firestore.Client()

# Nombre del bucket de GCP
BUCKET_NAME = os.getenv("BUCKET_NAME", "analisis-imagenes")

# Función para subir la imagen al bucket
def upload_to_bucket(file):
    """Sube el archivo al bucket de GCP y devuelve la URL pública."""
    filename = file.filename
    blob = storage_client.bucket(BUCKET_NAME).blob(filename)
    blob.upload_from_file(file)
    blob.make_public()
    file_url = blob.public_url
    return file_url

# Función para hacer la transcripción usando Google Cloud Vision
def transcribe_image(image_path):
    """Transcribe el texto de una imagen usando la API de Google Cloud Vision."""
    image = vision.Image()
    image.source.image_uri = image_path
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    return None

# Función para guardar los datos en Firestore
def save_to_firestore(image_url, extracted_text):
    """Guarda los resultados en Firestore."""
    doc_ref = firestore_client.collection('notas_transcritas').document()
    doc_ref.set({
        'image_url': image_url,
        'extracted_text': extracted_text
    })

# Función que se llamará desde Google Cloud Functions
def process_image(request):
    """Maneja la subida de una imagen y su procesamiento."""

    # Verificar que la solicitud sea POST
    if request.method != 'POST':
        return 'Invalid request method', 405

    # Obtener el archivo de la solicitud
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Subir la imagen al bucket
        image_url = upload_to_bucket(file)

        # Transcribir la imagen usando Cloud Vision
        extracted_text = transcribe_image(image_url)

        # Guardar los datos en Firestore
        if extracted_text:
            save_to_firestore(image_url, extracted_text)

        # Retornar la URL de la imagen y el texto extraído
        return jsonify({
            'image_url': image_url,
            'extracted_text': extracted_text or 'No se pudo extraer texto'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500