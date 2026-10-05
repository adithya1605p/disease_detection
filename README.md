# 🩺 VitaScan — Multi-Disease Screening Dashboard

VitaScan is a presentation-ready Streamlit application that demonstrates machine-learning-based health screening across six disease datasets. It loads the trained Random Forest artifacts in `models/`, provides a dynamic input form for each disease, and returns an easy-to-understand screening estimate with confidence information.

> **Medical disclaimer:** This project is an educational demonstration and is not a medical diagnosis. Results should not be used to make healthcare decisions or replace advice from a qualified healthcare professional.

## ✨ Current dashboard features

- Modern VitaScan dashboard with responsive dark-theme styling
- Sidebar model selector for six supported diseases
- Overview tab with:
  - Disease descriptions
  - Training-record counts
  - Reported model accuracy
  - Number of model input signals
  - Disease summary cards
- Run a screening tab with:
  - Dynamic numeric and categorical input fields
  - Input fields generated from each model's saved feature list
  - Readable labels for encoded categorical values
  - Example profile loader for quick presentations
  - Positive/negative screening result
  - Model confidence score
  - CSV screening-report download
- Model details tab with:
  - Random Forest explanation
  - Input feature table
  - Numeric/categorical input types
- Cached model and dataset loading for faster interaction
- Explicit warning that predictions are screening estimates only

## 🧬 Supported screening models

| Model | Dataset | Reported accuracy |
|---|---|---:|
| Diabetes | `data/diabetes.csv` | 74.0% |
| Heart Disease | `data/heart.csv` | 82.0% |
| Breast Cancer | `data/breast-cancer.csv` | 96.5% |
| Kidney Disease | `data/kidney_disease.csv` | 100% |
| Liver Disease | `data/indian_liver_patient.csv` | 61.5% |
| Stroke | `data/healthcare-dataset-stroke-data.csv` | 94.8% |

The reported values come from the existing training results documented by the project. They are not a guarantee of real-world clinical performance.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/adithya1605p/disease_detection.git
cd disease_detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the dashboard

From the project directory, run:

```bash
streamlit run app.py
```

Open the URL shown in the terminal, normally:

```text
http://localhost:8501
```

### Suggested presentation flow

1. Select a disease from the sidebar.
2. Open the **Overview** tab to explain the model and dataset size.
3. Open **Run a screening**.
4. Click **Load example profile** to populate a realistic demonstration profile.
5. Adjust one or two measurements to show how the estimate responds.
6. Click **Run screening estimate**.
7. Show the prediction, confidence, and model metrics.
8. Download the CSV report if a result artifact is needed.
9. Use **Model details** to explain the feature inputs and preprocessing.

## 🧪 Retrain the models

The repository includes the source datasets and a training pipeline. To retrain all six models:

```bash
python train_all.py
```

The training script:

1. Loads each dataset using `config.py`.
2. Drops configured ID and unused columns.
3. Replaces medically invalid zero values where configured.
4. Encodes configured categorical columns with `LabelEncoder`.
5. Fills missing numeric values with column medians.
6. Splits data into 80% training and 20% testing data.
7. Scales features with `StandardScaler`.
8. Trains a `RandomForestClassifier` with 200 estimators.
9. Saves the model artifacts to `models/`.

To run the basic data-loading check:

```bash
python test_train.py
```

## 🧠 Model artifacts

Each disease has four saved artifacts in `models/`:

```text
<disease>_model.pkl       Trained Random Forest classifier
<disease>_scaler.pkl      Fitted StandardScaler
<disease>_features.pkl    Feature names and model input order
<disease>_encoders.pkl    Saved categorical LabelEncoder objects
```

The dashboard reads these artifacts at runtime. If models are retrained, the app automatically uses the updated files after restarting Streamlit.

## 📁 Project structure

```text
disease_detection/
├── app.py                         # VitaScan Streamlit dashboard
├── config.py                      # Dataset and preprocessing configuration
├── train_all.py                   # Train and save all disease models
├── run_train.py                   # Training entry point
├── test_train.py                  # Dataset-loading smoke test
├── requirements.txt               # Python dependencies
├── data/                          # Source CSV datasets
├── models/                        # Saved models and preprocessing artifacts
├── .streamlit/
│   └── config.toml                # Streamlit theme and client settings
└── README.md
```

## 📦 Dependencies

The application uses:

- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- joblib
- streamlit-folium

Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

## ⚠️ Troubleshooting

### The app cannot find a model file

Run the application from the project directory and confirm that the `models/` folder contains the four artifacts for the selected disease. If they are missing, run:

```bash
python train_all.py
```

### PowerShell blocks virtual-environment activation

Run PowerShell as permitted by your local policy or use Command Prompt activation instead:

```bat
.venv\Scripts\activate
```

### Port 8501 is already in use

Start Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Adithya
