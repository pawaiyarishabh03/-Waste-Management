from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import urllib
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.utils import get_file
import pathlib
import tensorflow as tf
import numpy as np

app = Flask(__name__)
cors = CORS(app)

model_path = "classification.h5"


def loadImage(URL):
  with urllib.request.urlopen(URL) as url:
    img = tf.keras.utils.load_img(
      BytesIO(url.read()),
      target_size=(224, 224),
    )

    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    return img


@app.route('/test', methods=['POST'])
def test():
  model = tf.keras.models.load_model('classification.h5')
  print("Loaded Model")
  model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(), "acc"])
  print("Compiled Model")
  path = request.json["link"]
  print(path)

  img = loadImage(path)

  p = model.predict(img, batch_size=10)
  predicted_class = np.argmax(p[0])
  waste_labels = {0: "Non - recyclable", 1: "Organic", 2: "Recyclable"}

  return jsonify({
    'result': waste_labels[predicted_class],
    'confidence': str(np.max(p[0], axis=-1))
  })


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)
