#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

MODEL_PATH = 'bees-wasps-v2.tflite'

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['bee', 'wasp']

TARGET_SIZE = (150, 150)


# url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    response = build_response(result)

    return response


def predict(url):
    img = download_image(url)
    prepared_img = resize_image(img, target_size=TARGET_SIZE)

    X = convert_image_to_np_array(prepared_img)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return prediction[0, 0]


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def resize_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def convert_image_to_np_array(img):
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X /= 255
    return X


def build_response(prediction: np.float64):
    return {
        'prediction': float(prediction)
    }