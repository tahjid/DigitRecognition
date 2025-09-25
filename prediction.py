# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:07:48 2025

@author: USER
"""

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load saved model
model = load_model("digit_recognition.h5")

# Load user-uploaded image
img = Image.open("uploaded_digit.jpeg").convert("L")  # grayscale
img = img.resize((28,28))                            # resize
arr = np.array(img)

# Invert colors if needed
arr = 255 - arr

# Normalize
arr = arr / 255.0

# Reshape
arr = arr.reshape(1,28,28,1)

# Predict
pred = model.predict(arr)
digit = np.argmax(pred)
print("Predicted digit:", digit)
