from flask import Flask, send_from_directory, Response, render_template, request, jsonify, session
import os
import dotenv
from PIL import Image, ImageDraw
import pandas as pd
import sqlite3 as sql
import time
import threading
from glob import glob
import json
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
from io import BytesIO
import base64
import cv2

StartTime = time.time()

settings = json.load(open("settings.json"))
base_dir = settings['directory_base']
http_port = settings['http_port']

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for using session

# To load images from drops
dotenv.load_dotenv()
username = os.getenv('D_USERNAME')
password = os.getenv('D_PASSWORD')
base_url = os.getenv('DATABASE_URL')


def load_sql_db():
    # Prepare the data for the POST request if needed
    db_url = base_dir + 'db.sqlite3'
    # Sending the POST request to the server with authentication
    response = requests.get(db_url, stream=True)

    if response.status_code == 200:
        # Save the database locally and returns the path to it
        filename = 'db_from_pithy2.sqlite3'
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return filename
    else:
        # Handling errors or unsuccessful responses
        return "Database not found", response.status_code


def initialize_SAM(model_size='small'):
    import torch
    import cv2
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    if model_size == 'small':  # ~366MB
        sam_checkpoint = "SAM_checkpoints/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    elif model_size == 'huge':  # ~2.5GB
        sam_checkpoint = "SAM_checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:  # Choose small by default
        sam_checkpoint = "SAM_checkpoints/sam_vit_b_01ec64.pth"
        model_type = "vit_b"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator


sam, mask_generator = initialize_SAM()


@app.route("/")
def home(): return open("index.html").read()


@app.route("/measure/<path:path>")
def measure(path): return open("measure.html").read()


@app.route("/dird")
def dird(): return json.dumps(glob(f"{base_dir}/sem/**/*.json"))


@app.route('/image/<path:filename>')
def serve_image(filename):
    # Prepare the data for the POST request if needed
    image_url = base_url + filename
    # Sending the POST request to the server with authentication
    response = requests.get(image_url)

    if response.status_code == 200:
        # Forward the image content as received with the correct content type
        #
        return Response(response.content, mimetype=response.headers['Content-Type'])
    else:
        # Handling errors or unsuccessful responses
        return "Image not found", response.status_code


@app.route("/sems")
def sems():
    sql_db = load_sql_db()
    q = "SELECT * from `table`"
    df = pd.read_sql(q, con=sql.connect(sql_db))
    df['img'] = df['img'].apply(lambda x: x.replace('/drops/', '', 1))
    df['img'] = df['img'].apply(
        lambda filename: f'/image/{filename}')
    return df[['time', 'user', 'label', 'img']].to_html(table_id='data', index=None)


@app.route('/drops/<path:path>')
def send_file(path):
    return send_from_directory('/drops', path)


TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)


def save_image_to_disk(image, filename):
    """Save the PIL image to the disk."""
    filepath = os.path.join(TEMP_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath


def load_image_from_disk(filename):
    """Load an image from disk as a PIL image."""
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None


@app.route("/sam/<path:filename>", methods=['GET', 'POST'])
def sam(filename):
    image_name = filename.split('/')[-1]
    image_key = f'image_{image_name}'  # Unique identifier for the image

    if request.method == 'GET':
        # Fetch the original image if not in disk cache
        image_url = base_url + filename
        response = requests.get(image_url)

        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, flags=0)
            img = preprocess_image(img)
            # Save the original image to disk for future modifications
            save_image_to_disk(img, image_key)
        else:
            return "Image not found", response.status_code

        # Convert the image to base64 for display in the template
        _, img_buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(img_buffer).decode('utf-8')
        return render_template('sam.html', img_data=img_str)

    if request.method == 'POST':
        # Load the current image from disk
        img = load_image_from_disk(image_key)
        if img is None:
            return "No image on disk", 400

        if 'generate_masks' in request.form:
            print('Starting mask generation...')
            masks = mask_generator.generate(img)
            print(f'Masks generated...: {len(masks)}')
            new_img = add_masks(masks, img)
            save_image_to_disk(new_img, image_key)
            print('New image saved to disk')
            # Convert the modified image to base64 for display
            _, img_buffer = cv2.imencode('.png', new_img)
            img_str = base64.b64encode(img_buffer).decode('utf-8')
            return jsonify({'result': f'{len(masks)} were generated.', 'image': img_str})

        elif 'add_circle' in request.form:
            print('Adding red circle...')
            # Button 2: Add a circle to the image
            height, width = img.shape[:2]
            # Generate random position for the circle
            x = np.random.randint(100, width - 100)
            y = np.random.randint(100, height - 100)
            # Draw a circle on the current image (using OpenCV)
            cv2.circle(img, (x, y), 100, (0, 0, 255), 5)  # Red circle

            # Save the modified image back to disk
            save_image_to_disk(img, image_key)

            # Convert the modified image to base64 for display
            _, img_buffer = cv2.imencode('.png', img)
            img_str = base64.b64encode(img_buffer).decode('utf-8')
            return jsonify({'image': img_str})


def preprocess_image(img):
    """
    Preprocess the image using OpenCV.
    Example: Convert to grayscale and apply Gaussian blur.
    """

    # img = img[0:1080, 0:1920]

    return img


def add_masks(masks, img):
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    alpha = 0.3  # Opacity of the mask
    img2 = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 3), dtype=np.uint8)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.randint(0, 255, 3)])
        img2[m] = color_mask

    new_img = cv2.addWeighted(img, 1, img2, alpha, gamma=0)
    return new_img


app.run(port=http_port, debug=True)
