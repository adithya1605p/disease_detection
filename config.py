DISEASE_CONFIG = {
    "Diabetes": {
        "file": "data/diabetes.csv",
        "target": "Outcome",
        "categorical_cols": [],
        "zero_invalid_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "drop_cols": []
    },
    "Heart Disease": {
        "file": "data/heart.csv",
        "target": "target",
        "categorical_cols": [],
        "zero_invalid_cols": [],
        "drop_cols": []
    },
    "Breast Cancer": {
        "file": "data/breast-cancer.csv",
        "target": "diagnosis",
        "categorical_cols": ["diagnosis"],
        "zero_invalid_cols": [],
        "drop_cols": ["id", "Unnamed: 32"]
    },
    "Kidney Disease": {
        "file": "data/kidney_disease.csv",
        "target": "classification",
        "categorical_cols": ["classification", "rbc", "pc", "pcc", "ba",
                              "htn", "dm", "cad", "appet", "pe", "ane"],
        "zero_invalid_cols": [],
        "drop_cols": ["id"]
    },
    "Liver Disease": {
        "file": "data/indian_liver_patient.csv",
        "target": "Dataset",
        "categorical_cols": ["Gender"],
        "zero_invalid_cols": [],
        "drop_cols": []
    },
    "Stroke": {
        "file": "data/healthcare-dataset-stroke-data.csv",
        "target": "stroke",
        "categorical_cols": ["gender", "ever_married", "work_type",
                              "Residence_type", "smoking_status"],
        "zero_invalid_cols": [],
        "drop_cols": ["id"]
    }
}