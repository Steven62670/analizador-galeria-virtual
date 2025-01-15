import os
from google.cloud import firestore, vision

def process_image(event, context):
    """Analiza una imagen al detectarse en el bucket."""
    bucket_name = event['bucket']
    file_name = event['name']

    # Validar que el bucket sea el esperado
    if bucket_name != 'analisis-imagenes':
        print(f"Skipping file {file_name} in bucket {bucket_name}. Not the expected bucket.")
        return

    # Procesar la imagen con Vision API
    client = vision.ImageAnnotatorClient()
    image = vision.Image(source=vision.ImageSource(image_uri=f'gs://{bucket_name}/{file_name}'))
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Cloud Vision Error: {response.error.message}")

    text = response.text_annotations[0].description if response.text_annotations else ""

    # Guardar resultados en Firestore
    db = firestore.Client()
    db.collection('notes').document(file_name).set({
        "file_name": file_name,
        "text": text
    })
    print(f"Processed file {file_name} and saved results to Firestore.")
