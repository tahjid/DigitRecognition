Handwritten Digit Recognition

This project implements a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize single-digit handwritten numbers (0–9) from drawings or uploaded photos.

Project Structure:
- digit.py      : Trains the CNN model on MNIST and saves it as digit_recognition.h5
- prediction_model.py : Loads the saved model and predicts digits from uploaded images

How to Use:

1. Train the Model
   Run:
       python digit.py
   - Trains the CNN model on MNIST
   - Saves the model as digit_recognition.h5

2. Predict from Image
   Run:
       python prediction_model.py --image path/to/your_digit.jpg
   - Loads the saved model
   - Preprocesses the image (grayscale, resize 28x28, invert colors, normalize)
   - Predicts the digit and prints the result

Notes:
- Input images can be handwritten digits from notebook photos
- Ensure the digit is clearly written on a light background
- Preprocessing ensures compatibility with MNIST-trained model

Requirements:
- tensorflow
- numpy
- pillow
