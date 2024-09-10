from flask import Flask, send_from_directory, Response
import os
import dotenv
from PIL import Image
import pandas as pd
import sqlite3 as sql
import time
import threading
from glob import glob
import json
import requests
from requests.auth import HTTPBasicAuth

StartTime = time.time()

settings = json.load(open("settings.json"))
base_dir = settings['directory_base']
http_port = settings['http_port']

app = Flask(__name__)
# fndb = f"{base_dir}/db.sqlite3"
fndb = r"C:\Users\matth\OneDrive - CentraleSupelec\Columbia Engineering\Steingart Research\Code\Seminal with SAM\seminal_with_sam\db.sqlite3"
print(fndb)

# To load images from drops
dotenv.load_dotenv()
username = os.getenv('D_USERNAME')
password = os.getenv('D_PASSWORD')
base_url = os.getenv('DATABASE_URL')


def load_sql_db():
    # Prepare the data for the POST request if needed
    db_url = base_dir + 'db.sqlite3'
    print(db_url)
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
        # , mimetype=response.headers['Content-Type'])
        return Response(response.content)
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


app.run(host="0.0.0.0", port=http_port, debug=True)
