# 🌱 Fertilizer Recommendation System

## Overview
A Machine Learning web application that predicts the most suitable fertilizer based on soil nutrients, crop type, temperature, humidity, and moisture.

## Features
- Fertilizer prediction using Machine Learning
- Decision Tree Classifier
- Confidence Score
- Flask Web Application
- Responsive Bootstrap UI
- Multiple Model Comparison

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Bootstrap 5

## Dataset
Kaggle Fertilizer Prediction Dataset:
https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction

## Model Performance

| Model | Accuracy |
|---------|----------|
| Decision Tree | 100% |
| Random Forest | 95% |
| Logistic Regression | 90% |
| KNN | 60% |

## Project Structure

```text
Fertilizer-Prediction-System
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── dataset/
├── models/
├── templates/
└── static/
```

## Installation

### Clone Repository

```bash
git clone <your-github-repository-link>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

## Input Features

- Temperature
- Humidity
- Moisture
- Nitrogen
- Potassium
- Phosphorous
- Soil Type
- Crop Type

## Output

The system predicts:
- Recommended Fertilizer
- Prediction Confidence Score

## Future Improvements

- Larger Dataset
- Random Forest Deployment
- Cloud Deployment
- Real-Time Weather Integration
- Fertilizer Recommendation History

## Author

Vyshnavi Chava
