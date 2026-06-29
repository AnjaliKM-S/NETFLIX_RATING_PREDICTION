
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

TYPE_CATEGORIES = ['Movie', 'TV Show']
COUNTRY_CATEGORIES = [
    'United States',
    'India',
    'United Kingdom',
    'Japan',
    'France',
    'Canada',
    'Spain',
    'South Korea',
    'Germany',
    'Mexico',
    'China',
    'Australia',
]
LISTED_IN_CATEGORIES = [
    'Dramas',
    'International Movies',
    'Comedies',
    'International TV Shows',
    'Action & Adventure',
    'Independent Movies',
    'TV Dramas',
    'Children & Family Movies',
    'Thrillers',
    'Romantic Movies',
    'Crime TV Shows',
    'TV Comedies',
]
NUMERIC_FEATURES = ['duration_minutes', 'duration_seasons', 'release_year', 'year_added']
CATEGORICAL_FEATURES = ['type', 'country', 'listed_in']
RATING_COLUMNS = [
    'rating_NC-17',
    'rating_NR',
    'rating_PG',
    'rating_PG-13',
    'rating_R',
    'rating_TV-14',
    'rating_TV-G',
    'rating_TV-MA',
    'rating_TV-PG',
    'rating_TV-Y',
    'rating_TV-Y7',
    'rating_TV-Y7-FV',
    'rating_UR',
]

FEATURE_NAMES = NUMERIC_FEATURES + [
    f'type_{t}' for t in TYPE_CATEGORIES
] + [
    f'country_{c}' for c in COUNTRY_CATEGORIES + ['Other']
] + [
    f'listed_in_{g}' for g in LISTED_IN_CATEGORIES + ['Other']
]


def clean_listed_in(value):
    value = str(value)
    cleaned = value.replace('[', '').replace(']', '').replace("'", '').replace('"', '').strip()
    return cleaned or 'Other'


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df['country'] = df['country'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].apply(clean_listed_in)
    df['rating_label'] = df[RATING_COLUMNS].idxmax(axis=1).str.replace('rating_', '', regex=False)
    return df


def prepare_features(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X['country'] = X['country'].where(X['country'].isin(COUNTRY_CATEGORIES), 'Other')
    X['listed_in'] = X['listed_in'].where(X['listed_in'].isin(LISTED_IN_CATEGORIES), 'Other')

    encoder = OneHotEncoder(
        categories=[TYPE_CATEGORIES, COUNTRY_CATEGORIES + ['Other'], LISTED_IN_CATEGORIES + ['Other']],
        handle_unknown='ignore',
        sparse_output=False,
    )
    X_encoded = encoder.fit_transform(X[CATEGORICAL_FEATURES])
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X[NUMERIC_FEATURES])
    X_final = np.hstack([X_numeric, X_encoded])
    return X_final, scaler


def train():
    df = load_dataset()
    X, scaler = prepare_features(df)
    y = df['rating_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    score = model.score(X_test, y_test)
    print(f'Training complete. Test accuracy: {score:.4f}')
    print(f'Model saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')


if __name__ == '__main__':
    train()
