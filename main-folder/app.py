from flask import Flask, render_template, request, redirect, url_for
import os
from image_processor import process_and_store_image
from search import (
    build_faiss_index, search_top_k, search_face_image,
    search_face_then_clip, search_multiple_faces_from_images
)
from grouping import find_images_by_face_clustering, group_images_by_clip_dbscan_with_text
from batch_ops import resize_image, add_watermark, blur_background, convert_format
from PIL import Image
import uuid
from datetime import datetime
from flask import jsonify
from pymongo import MongoClient

import db
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = '../static/uploads'
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'output_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["new_database"]
collection = db["your_collection_name"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        files = request.files.getlist('images')
        print("Uploaded:", [f.filename for f in files])
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)

                # Store the relative accessible path in DB (to be used in <img src>)
                source = os.path.join('static', 'uploads', filename).replace("\\", "/")
                collection.insert_one({'source': source})

        return redirect('/uploads')

    # GET request - fetch all images from DB and pass to template
    images = list(collection.find({}, {'_id': 0, 'source': 1}))
    image_sources = [img['source'] for img in images]

    return render_template('upload.html', uploaded_images=image_sources)

@app.route('/images')
def get_images():
    images = collection.find({}, {"_id": 0, "source": 1})
    return jsonify([img["source"].replace("\\", "/") for img in images])



@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        mode = request.form.get('mode')
        query = request.form.get('query', '')
        face_img = request.files.get('face_image')
        text_prompt = request.form.get('text_prompt', '')

        if mode == 'text':
            results = search_top_k(query, k=5)

        elif mode == 'face':
            if face_img:
                path = os.path.join(UPLOAD_FOLDER, face_img.filename)
                face_img.save(path)
                results = search_face_image(path, k=5)

        elif mode == 'face_text':
            if face_img:
                path = os.path.join(UPLOAD_FOLDER, face_img.filename)
                face_img.save(path)
                results = search_face_then_clip(path, text_prompt, k=5)

        elif mode == 'multi_face':
            image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
            results = search_multiple_faces_from_images(image_paths, k=5)

    return render_template('search.html', results=results)


@app.route('/grouping', methods=['GET', 'POST'])
def grouping():
    groups = {}
    if request.method == 'POST':
        mode = request.form.get('mode')

        if mode == 'face_grouping':
            image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
            matched = find_images_by_face_clustering(image_paths)
            groups = {0: matched}

        elif mode == 'clip_text_grouping':
            prompt = request.form.get('prompt', 'people wearing red jackets')
            groups = group_images_by_clip_dbscan_with_text(prompt, eps=0.18, similarity_threshold=0.23)

    return render_template('grouping.html', groups=groups)


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    processed = []
    if request.method == 'POST':
        resize = request.form.get('resize') == 'yes'
        watermark = request.form.get('watermark') == 'yes'
        blur = request.form.get('blur') == 'yes'
        convert = request.form.get('convert', 'none')

        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))
        watermark_text = request.form.get('watermark_text', '')
        watermark_position = request.form.get('watermark_position', 'bottom_right')

        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:30]

        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)
            img = Image.open(img_path).convert("RGBA")

            if resize and width and height:
                img = resize_image(img, width, height)

            if blur:
                img = blur_background(img)

            if watermark and watermark_text:
                img = add_watermark(img, watermark_text, position=watermark_position)

            if convert != 'none':
                out_name = os.path.splitext(img_name)[0] + f"_processed.{convert.lower()}"
                img = convert_format(img, convert)
            else:
                out_name = os.path.splitext(img_name)[0] + "_processed.png"

            out_path = os.path.join(OUTPUT_DIR, out_name)
            img.save(out_path)
            processed.append(out_name)

    return render_template('batch.html', processed=processed)


if __name__ == '__main__':
    build_faiss_index()  # Build index once at startup
    app.run(debug=True)
