# from flask import Flask, render_template, request, redirect, url_for
# import os
# from image_processor import process_and_store_image
# from search import (
#     build_faiss_index, search_top_k, search_face_image,
#     search_face_then_clip, search_multiple_faces_from_images
# )
# from grouping import find_images_by_face_clustering, group_images_by_clip_dbscan_with_text
# from batch_ops import resize_image, add_watermark, blur_background, convert_format
# from PIL import Image
# import uuid
# from datetime import datetime
# from flask import jsonify
# from pymongo import MongoClient

# import db
# from werkzeug.utils import secure_filename


# UPLOAD_FOLDER = '../static/uploads'
# INPUT_DIR = 'input_images'
# OUTPUT_DIR = 'output_images'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# mongo_client = MongoClient("mongodb://localhost:27017/")
# db = mongo_client["new_database"]
# collection = db["your_collection_name"]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/uploads', methods=['GET', 'POST'])
# def uploads():
#     if request.method == 'POST':
#         files = request.files.getlist('images')
#         print("Uploaded:", [f.filename for f in files])
#         for file in files:
#             if file:
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#                 file.save(filepath)

#                 # Store the relative accessible path in DB (to be used in <img src>)
#                 source = os.path.join('static', 'uploads', filename).replace("\\", "/")
#                 collection.insert_one({'source': source})

#         return redirect('/uploads')

#     # GET request - fetch all images from DB and pass to template
#     images = list(collection.find({}, {'_id': 0, 'source': 1}))
#     image_sources = [img['source'] for img in images]

#     return render_template('upload.html', uploaded_images=image_sources)

# @app.route('/images')
# def get_images():
#     images = collection.find({}, {"_id": 0, "source": 1})
#     return jsonify([img["source"].replace("\\", "/") for img in images])



# @app.route('/search', methods=['GET', 'POST'])
# def search():
#     results = []
#     if request.method == 'POST':
#         mode = request.form.get('mode')
#         query = request.form.get('query', '')
#         face_img = request.files.get('face_image')
#         text_prompt = request.form.get('text_prompt', '')

#         if mode == 'text':
#             results = search_top_k(query, k=5)

#         elif mode == 'face':
#             if face_img:
#                 path = os.path.join(UPLOAD_FOLDER, face_img.filename)
#                 face_img.save(path)
#                 results = search_face_image(path, k=5)

#         elif mode == 'face_text':
#             if face_img:
#                 path = os.path.join(UPLOAD_FOLDER, face_img.filename)
#                 face_img.save(path)
#                 results = search_face_then_clip(path, text_prompt, k=5)

#         elif mode == 'multi_face':
#             image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
#             results = search_multiple_faces_from_images(image_paths, k=5)

#     return render_template('search.html', results=results)


# @app.route('/grouping', methods=['GET', 'POST'])
# def grouping():
#     groups = {}
#     if request.method == 'POST':
#         mode = request.form.get('mode')

#         if mode == 'face_grouping':
#             image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
#             matched = find_images_by_face_clustering(image_paths)
#             groups = {0: matched}

#         elif mode == 'clip_text_grouping':
#             prompt = request.form.get('prompt', 'people wearing red jackets')
#             groups = group_images_by_clip_dbscan_with_text(prompt, eps=0.18, similarity_threshold=0.23)

#     return render_template('grouping.html', groups=groups)


# @app.route('/batch', methods=['GET', 'POST'])
# def batch():
#     processed = []
#     if request.method == 'POST':
#         resize = request.form.get('resize') == 'yes'
#         watermark = request.form.get('watermark') == 'yes'
#         blur = request.form.get('blur') == 'yes'
#         convert = request.form.get('convert', 'none')

#         width = int(request.form.get('width', 0))
#         height = int(request.form.get('height', 0))
#         watermark_text = request.form.get('watermark_text', '')
#         watermark_position = request.form.get('watermark_position', 'bottom_right')

#         image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:30]

#         for img_name in image_files:
#             img_path = os.path.join(INPUT_DIR, img_name)
#             img = Image.open(img_path).convert("RGBA")

#             if resize and width and height:
#                 img = resize_image(img, width, height)

#             if blur:
#                 img = blur_background(img)

#             if watermark and watermark_text:
#                 img = add_watermark(img, watermark_text, position=watermark_position)

#             if convert != 'none':
#                 out_name = os.path.splitext(img_name)[0] + f"_processed.{convert.lower()}"
#                 img = convert_format(img, convert)
#             else:
#                 out_name = os.path.splitext(img_name)[0] + "_processed.png"

#             out_path = os.path.join(OUTPUT_DIR, out_name)
#             img.save(out_path)
#             processed.append(out_name)

#     return render_template('batch.html', processed=processed)


# if __name__ == '__main__':
#     build_faiss_index()  # Build index once at startup
#     app.run(debug=True)

from flask import jsonify
from flask import Flask, render_template, request, redirect, url_for, flash
from search import build_faiss_index, search_top_k, search_face_image, search_face_then_clip, search_multiple_faces_from_images, search_clip_histogram_combined
import os
from werkzeug.utils import secure_filename
from image_processor import process_and_store_image
from pymongo import MongoClient
import tempfile
from PIL import Image
import requests
from io import BytesIO
from grouping import find_images_by_face_clustering, group_images_by_clip_dbscan_with_text
from batch_ops import resize_image, add_watermark, blur_background, convert_format
from batch_ops import add_border, enhance_image, remove_background, add_watermark, blur_background, resize_image
from bson import ObjectId
import uuid
import secrets



# Setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
# app_root = os.path.dirname(os.path.abspath(__file__))  # Get the correct root
# upload_path = os.path.join(app_root, 'static', 'uploads')
# app.config['UPLOAD_FOLDER'] = upload_path
# os.makedirs(upload_path, exist_ok=True)

app_root = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(app_root, 'static', 'uploads')
app.config['BATCH_OUTPUT_FOLDER'] = os.path.join(app_root, 'static', 'batch_outputs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['BATCH_OUTPUT_FOLDER'], exist_ok=True)



# MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["photo_manager"]
collection = db["images"]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         uploaded_files = request.files.getlist('images')
#         for file in uploaded_files:
#             if file:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(file_path)

#                 # Process image & store metadata
#                 process_and_store_image(file_path)

#         return redirect(url_for('gallery'))

#     return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('images')
        if len(files) > 30:
            return "Too many files. Limit is 30.", 400

        for file in files:
            if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
                if len(file.read()) > 10 * 1024 * 1024:
                    return f"File {file.filename} exceeds 5MB limit.", 400
                file.seek(0)  # Reset file pointer after size check

                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                # Store in DB
                full_path = os.path.abspath(save_path)
                process_and_store_image(full_path)

        # return redirect(url_for('gallery'))
    images = collection.find().sort("timestamp", -1)
    
    return render_template('upload.html', images=images)



# @app.route('/gallery')
# def gallery():
#     images = collection.find().sort("timestamp", -1)
#     return render_template('gallery.html', images=images)

build_faiss_index()

# @app.route('/search', methods=['GET', 'POST'])
# def search():
#     if request.method == 'POST':
#         mode = request.form.get("mode")

#         results = []
#         if mode == "text":
#             query = request.form.get("text_query")
#             results = search_top_k(query, k=10)

#         elif mode == "face":
#             file = request.files.get("face_image")
#             if file:
#                 path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
#                 file.save(path)
#                 results = search_face_image(path, k=10)

#         elif mode == "face_text":
#             text = request.form.get("text_query")
#             file = request.files.get("face_image")
#             if file and text:
#                 path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
#                 file.save(path)
#                 results = search_face_then_clip(path, text, k=10)

#         elif mode == "multi_face":
#             files = request.files.getlist("face_images")
#             image_paths = []
#             for file in files:
#                 if file:
#                     path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
#                     file.save(path)
#                     image_paths.append(path)
#             if image_paths:
#                 results = search_multiple_faces_from_images(image_paths, k=10)
#         for res in results:
#             if not res['image'].startswith("http"):
#                 res['web_path'] = url_for('static', filename=res['image'])
#             else:
#                 res['web_path'] = res['image']

#         return render_template("upload.html", results=results)

#     return render_template("upload.html", results=None)


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        mode = request.form.get("mode")
        results = []

        if mode == "text":
            query = request.form.get("text_query")
            results = search_top_k(query, k=10)

        elif mode == "face":
            file = request.files.get("face_image")
            if file and file.mimetype.startswith("image/"):
                path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
                file.save(path)
                results = search_face_image(path, k=10)
                

        # elif mode == "face_text":
        #     text = request.form.get("text_query")
        #     file = request.files.get("face_image")
        #     if file and text and file.mimetype.startswith("image/"):
        #         path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        #         file.save(path)
        #         results = search_face_then_clip(path, text, k=10)

        elif mode == "multi_face":
            file = request.files.get("face_images")
            if file and file.mimetype.startswith("image/"):
                path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
                file.save(path)
                results = search_clip_histogram_combined(path)

        if results:
            for res in results:
                if not res['image'].startswith("http"):
                    res['web_path'] = url_for('static', filename=res['image'])
                else:
                    res['web_path'] = res['image']

        return render_template("searches.html", results=results)

    return render_template("searches.html", results=None)

# @app.route('/album')
# def album():
#     albums = list(db.albums.find().sort("created_at", -1))
#     images = list(db.images.find())  # all images, or preload only needed ones

#     # Create a map for quick access
#     image_map = {img['image_id']: img for img in images}

#     for album in albums:
#         album['images'] = [image_map[i] for i in album['image_ids'] if i in image_map]

#     return render_template("album.html", albums=albums)


from bson import ObjectId

@app.route('/album')
def album():
    albums = list(db.albums.find().sort("created_at", -1))
    images = list(db.images.find())

    # Convert ObjectId to string for JSON compatibility
    for img in images:
        img['_id'] = str(img['_id'])
        img['image_id'] = str(img['image_id'])  # if image_id is also ObjectId

    image_map = {img['image_id']: img for img in images}

    for album in albums:
        album['_id'] = str(album['_id'])
        album['image_ids'] = [str(i) for i in album.get('image_ids', [])]
        album['images'] = [image_map[i] for i in album['image_ids'] if i in image_map]

    return render_template("album.html", albums=albums, images=images)




# @app.route("/start_album_creation", methods=["POST"])
# def start_album_creation():
#     selected_ids = request.form.getlist("selected_images")

#     if not selected_ids:
#         flash("Please select at least one image to create an album.")
#         return redirect(url_for("upload", select=1))

#     # Fetch the selected image data from your database
#     images = get_images_by_ids(selected_ids)  # Replace this with your actual DB call

    # Render album creation page with selected images
    # return render_template("album.html", selected_ids=selected_ids, images=images)

from werkzeug.utils import secure_filename
import os
from datetime import datetime
import uuid
import json
from flask import request, redirect, flash

@app.route('/create_album', methods=['POST'])
def create_album():
    album_name = request.form.get("album_name")
    album_type = request.form.get("album_type")
    print("Album Type:", album_type)
    print("Album Name:", album_name)

    if not album_name or not album_type:
        flash("Album name or type missing.", "error")
        return redirect("/album")

    image_ids = []  # ✅ initialize before any branch


    if album_type == "manual":
        selected_images = request.form.get("selected_images")
        print("Selected Images:", selected_images)
        if not selected_images:
            flash("No images selected.", "error")
            return redirect("/album")
        image_ids = json.loads(selected_images)

    elif album_type in ["faces", "similarity"]:
        uploaded_file = request.files.get("uploaded_image")
        print("Uploaded File:", uploaded_file)

        if not uploaded_file:
            flash("No image uploaded.", "error")
            return redirect("/album")

        # Save temporarily
        filename = secure_filename(uploaded_file.filename)
        upload_path = os.path.join("static/uploads", filename)
        uploaded_file.save(upload_path)
        print("Saved image to:", upload_path)

        images = list(collection.find())



        # AI processing
        if album_type == "faces":
            image_idss = find_images_by_face_clustering([upload_path])  # ✅ correct
            


        elif album_type == "similarity":
            image_idss = search_clip_histogram_combined(upload_path)
            print("Image IDs from AI:", image_idss)

        for img in images:
                for img_id in image_idss:
                    if img['source'].replace("static/", "") == img_id['image']:

                        image_ids.append(img['image_id'])
                        print("Matched Image ID:", img['image_id'])

        print("AI result:", image_ids)


        if not image_ids:
            flash("No matching images found.", "error")
            return redirect("/album")

    else:
        flash("Invalid album type.", "error")
        return redirect("/album")

    # Save album
    album = {
        "album_id": str(uuid.uuid4()),
        "name": album_name,
        "image_ids": image_ids,
        "created_at": datetime.utcnow()
    }

    db.albums.insert_one(album)
    print("Album created:", album)

    return redirect("/album")




@app.route("/finalize_album", methods=["POST"])
def finalize_album():
    album_name = request.form.get("album_name")
    selected_ids = request.form.getlist("selected_images")

    if not selected_ids:
        flash("Please select some images.")
        return redirect(url_for("album"))

    # Save to DB — your own logic
    create_album(album_name, selected_ids)

    flash("Album created!")
    return redirect(url_for("album"))


@app.route('/searches', methods=['GET', 'POST'])
def searches():
    return render_template('searches.html', images=collection.find().sort("timestamp", -1))

from flask import request, jsonify
from PIL import Image
import io, base64

from flask import request, send_file
from PIL import Image
import io
import cv2
import numpy as np

@app.route('/apply_tool', methods=['POST'])
def apply_tool():
    tool = request.form.get("tool")
    image_file = request.files.get("image")

    if not tool or not image_file:
        return "Missing data", 400

    # Load PIL image
    img = Image.open(image_file).convert("RGB")

    # Convert to OpenCV if needed
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply tool
    if tool == "resize":
        width = int(request.form.get("width", 300))
        height = int(request.form.get("height", 300))
        img_cv = cv2.resize(img_cv, (width, height))

    elif tool == "watermark":
        text = request.form.get("text", "Watermark")
        img_cv = add_watermark(img_cv, text=text)

    elif tool == "enhance":
        img_cv = enhance_image(img_cv)

    elif tool == "border":
        img_cv = add_border(img_cv)

    elif tool == "blur":
        blur_background(img_cv)
        img_cv = cv2.imread('blurred_background3.jpg')

    elif tool == "removebg":
        img = remove_background(img)
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    # Convert back to PIL
    result_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Return image
    output = io.BytesIO()
    result_pil.save(output, format='JPEG')
    output.seek(0)
    return send_file(output, mimetype='image/jpeg')



@app.route("/delete_images", methods=["POST"])
def delete_images():
    selected_ids = request.form.getlist("selected_images")
    for img_id in selected_ids:
        image = collection.find_one({"image_id": img_id})
        if image:
            # delete from file system
            try:
                os.remove(os.path.join("static", image["source"]))
            except Exception:
                pass
            # delete from database
            collection.delete_one({"image_id": img_id})
    return redirect(url_for("upload"))


@app.route("/delete_image", methods=["POST"])
def delete_image():
    data = request.get_json()
    image_id = data.get("image_id")
    image = collection.find_one({"image_id": image_id})
    if image:
        try:
            os.remove(os.path.join("static", image["source"]))
        except Exception:
            pass
        collection.delete_one({"image_id": image_id})
    return jsonify({"success": True})


@app.route("/editor_room", methods=['GET', 'POST'])
def editor_room():
    if request.method == 'POST':
        image_id = request.form.get("image_id")
        if image_id:
            image = collection.find_one({"image_id": image_id})
            if image:
                try:
                    os.remove(os.path.join("static", image["source"]))
                except Exception:
                    pass
                collection.delete_one({"image_id": image_id})
                flash("Image deleted successfully!", "success")
            else:
                flash("Image not found.", "error")
        else:
            flash("No image ID provided.", "error")

    images = collection.find().sort("timestamp", -1)
    return render_template('editor_room.html', images=images)


import atexit
import glob

@atexit.register
def cleanup_temp_images():
    for temp_img in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        try:
            os.remove(temp_img)
        except:
            pass


@app.route('/group', methods=['GET', 'POST'])
def group():
    results = []

    if request.method == 'POST':
        mode = request.form.get("mode")

        if mode == "face":
            files = request.files.getlist("face_images")
            image_paths = []
            for file in files:
                path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
                file.save(path)
                image_paths.append(path)

            if image_paths:
                matched = find_images_by_face_clustering(image_paths)
                results = matched

        elif mode == "clip_text":
            text_prompt = request.form.get("text_prompt")
            if text_prompt:
                matched = group_images_by_clip_dbscan_with_text(text_prompt=text_prompt, similarity_threshold=0.23)
                results = matched

        # Web-accessible paths
        for res in results:
            if not res.startswith("http"):
                res_web = url_for('static', filename=res)
            else:
                res_web = res
            results[results.index(res)] = {"web_path": res_web}

        return render_template("group_results.html", results=results)

    return render_template("group.html")

@app.route("/batch", methods=["GET", "POST"])
def batch():
    upload_folder = app.config['UPLOAD_FOLDER']
    available_images = [f for f in os.listdir(upload_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    processed_files = []

    if request.method == "POST":
        selected_images = request.form.getlist("selected_images")
        width = request.form.get("width", type=int)
        height = request.form.get("height", type=int)
        blur = request.form.get("blur") == "on"
        watermark = request.form.get("watermark_text")
        position = request.form.get("watermark_position")
        format_choice = request.form.get("format")

        os.makedirs("static/batch_outputs", exist_ok=True)

        for filename in selected_images:
            input_path = os.path.join(upload_folder, filename)
            img = Image.open(input_path).convert("RGBA")

            if width and height:
                img = resize_image(img, width, height)

            if blur:
                img = blur_background(img)

            if watermark:
                img = add_watermark(img, watermark, position=position)

            ext = format_choice.lower() if format_choice != "skip" else "png"
            output_path = os.path.join(app.config['BATCH_OUTPUT_FOLDER'], f"{os.path.splitext(filename)[0]}_processed.{ext}")
            img.save(output_path)
            processed_files.append(url_for('static', filename='batch_outputs/' + os.path.basename(output_path)))


        flash(f"{len(processed_files)} images processed successfully!", "success")

    return render_template("batch.html", available_images=available_images, processed=processed_files)


if __name__ == '__main__':
    app.run(debug=True)
    # for img in collection.find():
    #     print("Stored source:", img['source'])


