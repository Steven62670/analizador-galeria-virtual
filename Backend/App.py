import os
from google.cloud import storage, vision, firestore
from flask import Flask, request, jsonify

app = Flask(__name__)

# Inicializar clientes
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
firestore_client = firestore.Client()

bucket_name = "tu-bucket-en-gcp"  # Reemplaza con tu bucket
bucket = storage_client.get_bucket(bucket_name)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Subir la imagen al bucket de Cloud Storage
    file = request.files['image']
    blob = bucket.blob(file.filename)
    blob.upload_from_file(file)

    # Procesar imagen con la API de Cloud Vision
    image = vision.Image(source=vision.ImageSource(gcs_image_uri=f"gs://{bucket_name}/{file.filename}"))
    response = vision_client.annotate_image({
        'image': image,
        'features': [
            {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
            {'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION},
            {'type_': vision.Feature.Type.FACE_DETECTION},
            {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
        ],
    })

    # Guardar los resultados en Firestore
    doc_ref = firestore_client.collection('imagenes').document(file.filename)
    doc_ref.set({
        'filename': file.filename,
        'objects': response.localized_object_annotations,
        'text': response.text_annotations,
        'faces': response.face_annotations,
        'colors': response.image_properties.dominant_colors.colors,
    })

    return jsonify({"status": "success", "filename": file.filename})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

