import json
from flask import Flask, jsonify, request, json
#from requests import request
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd

test = tf.keras.models.load_model('resnet50.h5')

labels = {
    0 : 'apple_pie',
    1 : 'chicken_wings',
    2 : 'fish_and_chips',
    3 : 'omelette',
    4 : 'spaghetti_bolognese'
}

pred_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
)


app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])

def index():
    if request.method == "GET":
        return jsonify({'responce': 'get request called' })


@app.route("/im_size", methods=['GET', 'POST'])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img = np.asarray(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.imwrite('images/image.jpg',img)

    image_dir = Path('images/')
    pred_img_path = list(image_dir.glob(r'*.jpg'))
    image_paths_df = pd.Series(pred_img_path, name='imagepath').astype(str)
    fake_label = pd.Series('test', name='label')
    pred_df = pd.concat([image_paths_df, fake_label], axis=1)

    pred_images = pred_generator.flow_from_dataframe(
        dataframe=pred_df,
        x_col='imagepath',
        y_col='label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True
    )

    pred_img = pred_images.next()[0]

    prediction = np.argmax(test.predict(pred_img))
    label = labels[prediction]

    return jsonify({'response' : label})
#app.run()
if __name__ == '__main__':
    app.run(debug=True)