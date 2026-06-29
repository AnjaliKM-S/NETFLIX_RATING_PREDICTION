
# Netflix Rating Predictor

A Flask deployment project that predicts Netflix rating categories from cleaned Netflix metadata. The project reuses preprocessing logic from the training notebook and loads a trained model from `models/model.pkl`.

## Project Overview

This repository contains:

- `app.py` — Flask web application for rating prediction.
- `data/dataset.csv` — cleaned Netflix dataset used for training.
- `models/model.pkl` — trained classification model.
- `models/scaler.pkl` — numeric scaler used during preprocessing.
- `notebooks/training_file.py` — training script with preprocessing and model training logic.
- `templates/index.html` — Bootstrap-based front end.
- `static/style.css` — clean custom styles.

## Folder Structure

```
project-root/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── training_file.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

Start the Flask app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Flask Commands

If you prefer Flask CLI:

```bash
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

## Retraining the Model

If you want to retrain the model using the cleaned dataset:

```bash
python notebooks/training_file.py
```

## Deployment to PythonAnywhere

1. Upload the repository to PythonAnywhere.
2. In the web app settings, set the working directory to the project root.
3. Configure WSGI to use `app.py`.
4. Install dependencies in the PythonAnywhere virtual environment.
5. Restart the web application.
