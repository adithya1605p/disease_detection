# 🩺 Multi-Disease Detection System

A machine learning-powered web application for predicting the risk of multiple diseases using trained RandomForest models and Streamlit.

## Diseases Supported

1. **Diabetes** - Accuracy: 74.0%
2. **Heart Disease** - Accuracy: 82.0%
3. **Breast Cancer** - Accuracy: 96.5%
4. **Kidney Disease** - Accuracy: 100%
5. **Liver Disease** - Accuracy: 61.5%
6. **Stroke** - Accuracy: 94.8%

## Features

- User-friendly web interface built with Streamlit
- Real-time risk predictions based on patient health parameters
- Trained models using scikit-learn's RandomForestClassifier
- Data preprocessing including handling missing values and categorical encoding
- Support for both numeric and categorical inputs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adithya1605p/disease_detection.git
cd disease_detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train Models

Run the training script to train all models:
```bash
python train_all.py
```

This will:
- Load datasets from the `data/` folder
- Preprocess the data (handle missing values, encode categorical variables)
- Train RandomForest models
- Save trained models, scalers, and encoders to the `models/` folder

### Run the Web Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Project Structure

```
disease_detection/
├── app.py                    # Streamlit web application
├── train_all.py              # Model training script
├── config.py                 # Disease configuration
├── data/                     # Dataset files
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── breast-cancer.csv
│   ├── kidney_disease.csv
│   ├── indian_liver_patient.csv
│   └── healthcare-dataset-stroke-data.csv
├── models/                   # Trained models (generated after training)
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Details

All models are trained using:
- **Algorithm**: RandomForestClassifier with 200 estimators
- **Train/Test Split**: 80/20 with stratification
- **Preprocessing**: StandardScaler for feature normalization
- **Cross-validation**: Test set evaluation with accuracy metrics

## Data Requirements

Each disease requires specific input features. The app dynamically loads the required features for each disease and provides appropriate input fields for categorical and numeric variables.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib
- streamlit

Install all requirements:
```bash
pip install pandas scikit-learn joblib streamlit
```

## License

This project is open source and available under the MIT License.

## Author

Adithya
