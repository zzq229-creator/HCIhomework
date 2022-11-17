import face_recognition
from flask import Flask, render_template, jsonify, request, redirect
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)


@app.route('/')
def index():
    msg = "my name is caojianhua, China up!"
    return render_template("index.html", data=msg)  # 加入变量传递


@app.route('/camera')
def camera():
    return render_template("camera.html", )  # 加入变量传递


# You can change this to any folder on your system

known_face_encodings, known_face_names = [], []


def read_face(name, path):
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)


def read_all_face():
    for root, dirs, files in os.walk('face'):
        print('root_dir:', root)  # 当前路径
        print('sub_dirs:', dirs)  # 子文件夹
        print('files:', files)  # 文件名称，返回list类型
    for file in files:
        name = file.split('.')[0]
        read_face(name, 'face/' + file)


def base642PIL(base64_str):
    base64_str = base64_str.split(',')[1]
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def detect_faces_in_image(img):  # numpy * * 3
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    face_name = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_name.append(name)
    print({'face_locations': face_locations, 'name': face_name})
    res = jsonify({'face_locations': face_locations, 'name': face_name})
    return res


@app.route('/reco_face', methods=['POST'])
def recognize_image():
    img = request.form['image']
    img = base642PIL(img)
    img = np.array(img)
    return detect_faces_in_image(img)


@app.route('/add', methods=['POST'])
def upload_face():
    img = request.form['image']
    name = request.form['name']
    img = base642PIL(img)
    path = 'face/' + name + '.jpg'
    img.save(path)
    read_face(name, path)
    return 'success add'
@app.route('/add_audio', methods=['POST'])
def upload_audio():
    data = request.form['audioData']
    return 'success add audio'

@app.route('/audio')
def audio():
    return render_template("audio.html", )




def audio():
    return render_template("audio.html", )

if __name__ == "__main__":
    read_all_face()
    app.run(host="0.0.0.0", port=5001, debug=True)