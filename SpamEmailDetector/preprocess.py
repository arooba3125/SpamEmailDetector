import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

def load_and_preprocess_data(path):
    # Load and combine multiple datasets if available
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Add more datasets here if available
    # df2 = pd.read_csv("another_dataset.csv")
    # df = pd.concat([df, df2])
    return df

def clean_text(text):
    text = text.lower()
    # Keep meaningful punctuation and symbols
    text = re.sub(r"[^\w\s$€£¥%!?]", "", text)  
    # Normalize numbers
    text = re.sub(r"\d+", " NUMBER ", text)
    # Handle excessive punctuation
    text = re.sub(r"(!|\?){2,}", " MULTI_EXCLAM ", text)
    return text

class ExtraFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for text in X:
            # Count special characters
            num_excl = text.count('!')
            num_quest = text.count('?')
            # Check for common spam patterns
            has_http = int('http://' in text or 'https://' in text)
            has_dollar = int('$' in text or '€' in text or '£' in text)
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
            features.append([num_excl, num_quest, has_http, has_dollar, uppercase_ratio])
        return np.array(features)

def vectorize_text(df):
    df['message'] = df['message'].apply(clean_text)
    X = df['message']
    y = df['label']
    
    # Create feature union of text and extra features
    feature_union = FeatureUnion([
        ('text', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))),
        ('extra', ExtraFeatures())
    ])
    
    X_vec = feature_union.fit_transform(X)
    return X_vec, y, feature_union