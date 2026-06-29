
import os
import importlib.util
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
TRAINING_PATH = os.path.join(BASE_DIR, 'notebooks', 'training_file.py')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')


def load_training_module():
    spec = importlib.util.spec_from_file_location('training_file', TRAINING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_listed_in(value):
    value = str(value)
    cleaned = value.replace('[', '').replace(']', '').replace("'", '').replace('"', '').strip()
    return cleaned or 'Unknown'


def load_dashboard_data():
    df = pd.read_csv(DATA_PATH)
    df['listed_in'] = df['listed_in'].fillna('Unknown').apply(clean_listed_in)
    rating_cols = [c for c in df.columns if c.startswith('rating_')]
    df['rating_label'] = df[rating_cols].idxmax(axis=1).str.replace('rating_', '', regex=False)

    type_counts = df['type'].value_counts().reset_index().rename(columns={'index': 'type', 'type': 'count'}).to_dict('records')
    top_countries = df['country'].fillna('Unknown').value_counts().head(10).reset_index().rename(columns={'index': 'country', 'country': 'count'}).to_dict('records')
    top_genres = df['listed_in'].value_counts().head(10).reset_index().rename(columns={'index': 'genre', 'listed_in': 'count'}).to_dict('records')
    rating_counts = df['rating_label'].value_counts().reset_index().rename(columns={'index': 'rating', 'rating_label': 'count'}).to_dict('records')
    summary_stats = {
        'total_items': int(len(df)),
        'average_duration': float(df['duration_minutes'].mean().round(1)),
        'average_seasons': float(df['duration_seasons'].mean().round(2)),
        'average_release_year': float(df['release_year'].mean().round(1)),
        'average_year_added': float(df['year_added'].mean().round(1)),
    }
    return {
        'type_counts': type_counts,
        'top_countries': top_countries,
        'top_genres': top_genres,
        'rating_counts': rating_counts,
        'summary_stats': summary_stats,
    }


TRAINING = load_training_module()
MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
DASHBOARD_DATA = load_dashboard_data()

TYPE_OPTIONS = TRAINING.TYPE_CATEGORIES
COUNTRY_OPTIONS = TRAINING.COUNTRY_CATEGORIES + ['Other']
LISTED_IN_OPTIONS = TRAINING.LISTED_IN_CATEGORIES + ['Other']
NUMERIC_FEATURES = TRAINING.NUMERIC_FEATURES

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret'


def encode_one_hot(value, categories):
    return np.array([[1 if value == category else 0 for category in categories]], dtype=float)


def normalize_category(value, categories):
    value = str(value).strip()
    return value if value in categories else 'Other'


def build_feature_vector(form):
    try:
        duration_minutes = float(form.get('duration_minutes', '').strip())
        duration_seasons = float(form.get('duration_seasons', '').strip())
        release_year = float(form.get('release_year', '').strip())
        year_added = float(form.get('year_added', '').strip())
    except ValueError as exc:
        raise ValueError('Please enter valid numeric values for duration, release year, and year added.') from exc

    media_type = form.get('type', '').strip()
    country = normalize_category(form.get('country', ''), TRAINING.COUNTRY_CATEGORIES)
    listed_in = normalize_category(form.get('listed_in', ''), TRAINING.LISTED_IN_CATEGORIES)

    if media_type not in TYPE_OPTIONS:
        raise ValueError('Invalid media type selected.')

    numeric_vector = np.array([[duration_minutes, duration_seasons, release_year, year_added]], dtype=float)
    numeric_scaled = SCALER.transform(numeric_vector)

    type_vector = encode_one_hot(media_type, TYPE_OPTIONS)
    country_vector = encode_one_hot(country, COUNTRY_OPTIONS)
    listed_in_vector = encode_one_hot(listed_in, LISTED_IN_OPTIONS)

    predictor_vector = np.concatenate([numeric_scaled, type_vector, country_vector, listed_in_vector], axis=1)
    return predictor_vector, {
        'duration_minutes': duration_minutes,
        'duration_seasons': duration_seasons,
        'type': media_type,
        'release_year': int(release_year),
        'country': country,
        'listed_in': listed_in,
        'year_added': int(year_added),
    }


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        type_options=TYPE_OPTIONS,
        country_options=COUNTRY_OPTIONS,
        listed_in_options=LISTED_IN_OPTIONS,
        dashboard_data=DASHBOARD_DATA,
        result=None,
        probability=None,
        error=None,
        input_data={},
    )


@app.route('/predict', methods=['POST'])
def predict():
    error = None
    result = None
    probability = None
    input_data = {
        'duration_minutes': request.form.get('duration_minutes', ''),
        'duration_seasons': request.form.get('duration_seasons', ''),
        'type': request.form.get('type', ''),
        'release_year': request.form.get('release_year', ''),
        'country': request.form.get('country', ''),
        'listed_in': request.form.get('listed_in', ''),
        'year_added': request.form.get('year_added', ''),
    }

    try:
        features, input_data = build_feature_vector(request.form)
        prediction = MODEL.predict(features)[0]
        result = prediction
        if hasattr(MODEL, 'predict_proba'):
            probability = float(np.max(MODEL.predict_proba(features)))
    except Exception as exc:
        error = str(exc)

    return render_template(
        'index.html',
        type_options=TYPE_OPTIONS,
        country_options=COUNTRY_OPTIONS,
        listed_in_options=LISTED_IN_OPTIONS,
        dashboard_data=DASHBOARD_DATA,
        result=result,
        probability=probability,
        error=error,
        input_data=input_data,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
