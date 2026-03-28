import re
<<<<<<< HEAD
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
=======

def clean_text(text):
    # 1. Handle non-string inputs
    if not isinstance(text, str):
        return ""
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove special characters but KEEP numbers (they matter for ratings/prices!)
    # We remove punctuation but keep the words intact
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 4. Remove extra whitespace
    text = " ".join(text.split())
    
    return text
>>>>>>> teammate/main
