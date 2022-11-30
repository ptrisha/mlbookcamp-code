
# import libraries
import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image


interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(url):
   # download image
   img = download_image(url)

   # prepare image
   target_size = (150, 150)   # from homework 8
   img = prepare_image(img, target_size)

   # turn the image into numpy array and pre-process it by rescaling
   x = np.array(img, dtype='float32')
   X = np.array([x])
   # rescale the image
   X = X/255

   return X

def predict(url):
    # prepare image input from its URL
    X = preprocess(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    float_prediction = pred[0].tolist()

    return float_prediction


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
