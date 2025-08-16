# Brain Tumor Classifier Web App

This project is a **Flask-based web application** for classifying brain tumor images. Users can upload MRI or brain scan images and get AI predictions using a trained CNN model.

## Features

- Modern, sleek, and responsive web interface  
- Classifies brain tumor images into: Glioma, Meningioma, Notumor, Pituitary  
- Quick prediction with a single image upload  
- Glassmorphism and gradient-based premium UI  
- Flask API backend integrated with Python AI model  

## File Structure

brain-tumor-classifier/
│
├─ app.py # Flask API and AI integration
templates/
│
├─ index.html # Frontend web interface
├─ cnn_pixel_model.h5 # Trained CNN model
└─ README.md # Project documentation

## Example Images

 
![Glioma Example](https://github.com/AhmetFarukTUNC/tumor_detection/blob/main/1.png)




 
![Glioma Example](https://github.com/AhmetFarukTUNC/tumor_detection/blob/main/2.png)



## Installation & Run

1. Install required packages:

```bash
pip install flask flask-cors tensorflow numpy pillow opencv-python



2. Run the application:

python app.py

3.Open your browser at http://127.0.0.1:8000 and upload brain tumor images to get predictions.

Usage

Click the “Choose an image” button and select an MRI or brain scan image

Click the Predict button

View the results: predicted class and confidence score

Model Details

Model: CNN (Convolutional Neural Network)

Input size: 224x224 (RGB or Grayscale)

Output: 4 classes (Glioma, Meningioma, Notumor, Pituitary)

Preprocessing: Image resizing and Harris Corner Detection

Contributing

You can improve preprocessing or update the model

UI enhancements and mobile responsiveness are welcome

