import logging
import os
from flask import Flask, render_template, request, jsonify
from google.cloud import storage, firestore, vision

# Configurar Logging
logging.basicConfig(level=logging.INFO)

# Inicializar Flask y Google Cloud Clients
app = Flask(__name__)
storage_client = storage.Client()
firestore_client = firestore.Client()
vision_client = vision.ImageAnnotatorClient()

# Nombre del bucket fijo
BUCKET_NAME = 'analisis-imagenes'

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Sube una imagen al bucket y procesa la transcripción."""
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Subir al bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(uploaded_file.filename)
    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )
    logging.info(f"File {uploaded_file.filename} uploaded to {BUCKET_NAME}.")

    # Procesar con Cloud Vision
    image_uri = f"gs://{BUCKET_NAME}/{uploaded_file.filename}"
    text = analyze_text(image_uri)

    # Guardar en Firestore
    save_to_firestore(uploaded_file.filename, text)

    return jsonify({
        "message": "File uploaded and processed successfully.",
        "text": text
    })

def analyze_text(image_uri):
    """Utiliza Cloud Vision API para analizar texto."""
    image = vision.Image(source=vision.ImageSource(image_uri=image_uri))
    response = vision_client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Cloud Vision Error: {response.error.message}")

    return response.text_annotations[0].description if response.text_annotations else ""

def save_to_firestore(file_name, text):
    """Guarda los resultados de transcripción en Firestore."""
    doc_ref = firestore_client.collection('notes').document(file_name)
    doc_ref.set({
        "file_name": file_name,
        "text": text
    })
    logging.info(f"Results saved to Firestore for {file_name}.")

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


