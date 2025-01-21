import os
from google.cloud import firestore, vision, translate_v2 as translate

def process_image(event, context):
    """Analiza una imagen al detectarse en el bucket y traduce el texto si está en inglés."""
    bucket_name = event['bucket']
    file_name = event['name']

    # Validar que el bucket sea el esperado
    if bucket_name != 'analisis-imagenes':
        print(f"Skipping file {file_name} in bucket {bucket_name}. Not the expected bucket.")
        return

    # Procesar la imagen con Vision API
    vision_client = vision.ImageAnnotatorClient()
    image = vision.Image(source=vision.ImageSource(image_uri=f'gs://{bucket_name}/{file_name}'))
    response = vision_client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Cloud Vision Error: {response.error.message}")

    text = response.text_annotations[0].description if response.text_annotations else ""

    # Traducir texto si está en inglés
    translated_text = translate_text(text, target_language='es') if text else ""

    # Guardar resultados en Firestore
    db = firestore.Client()
    db.collection('notes').document(file_name).set({
        "file_name": file_name,
        "text": text,
        "translated_text": translated_text
    })
    print(f"Processed file {file_name}, translated if needed, and saved results to Firestore.")

def translate_text(text, target_language='es'):
    """Traduce el texto detectado si está en inglés."""
    translate_client = translate.Client()
    detection = translate_client.detect_language(text)
    source_language = detection.get("language", "")

    if source_language == 'en':
        translation = translate_client.translate(text, target_language=target_language)
        return translation.get("translatedText", "")

    return text

