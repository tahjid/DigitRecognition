# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:07:48 2025

@author: USER
"""

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_recognition.h5")

img = Image.open("uploaded_digit.jpeg").convert("L")
img = img.resize((28,28))                         
arr = np.array(img)

arr = 255 - arr
arr = arr / 255.0
arr = arr.reshape(1,28,28,1)
pred = model.predict(arr)
digit = np.argmax(pred)
print("Predicted digit:", digit)
