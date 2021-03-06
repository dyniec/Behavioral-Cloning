import os
import base64

import socketio
import eventlet
import eventlet.wsgi

import numpy as np

import argparse

import h5py
from keras.models import load_model

from random import uniform

from flask import Flask
from io import BytesIO
from PIL import Image

sio = socketio.Server()
app = Flask(__name__)
model=None
from keras import metrics
from keras import backend as K

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def top_2(y_true, y_pred):
    return K.mean(tf.nn.in_top_k(y_pred, K.argmax(y_true, axis=-1), 2))

metrics.rmse = rmse
metrics.top_2 = top_2


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = data["speed"]
        angularSpeed = data["angularSpeed"]
        imgStr = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgStr)))
        image_array = np.asarray(image)
        res=model.predict(image_array[None, :, :, :], batch_size=1)

        print( speed, angularSpeed )
        print(res)
        #send_control(uniform(-1,1), uniform(-1,1))
        sp=res[0,0]
        send_control(sp/2,res[0,1])
        sio.emit("request_telemetry", data = {})


def send_control(accel, steering):
    sio.emit(
        "steer",
        data={
            'accel': str(accel),
            'steering': str(steering)
        })
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model',type=str,
        help='Path to model h5 file. Model should be on the same path.')
    args=parser.parse_args()
    model=load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
