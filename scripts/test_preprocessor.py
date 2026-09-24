import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

sw = set(stopwords.words('english'))
NEGATIONS = {'not', 'no', 'never', 'nor', 'neither', 'cannot', 'hardly', 'barely', 'none', 'without', 'nothing'}
SENTIMENT_STOPWORDS = sw - NEGATIONS

CONTRACTIONS = {
    r"can't": "cannot",
    r"won't": "will not",
    r"n't": " not",
    r"doesn't": "does not",
    r"don't": "do not",
    r"didn't": "did not",
    r"isn't": "is not",
    r"wasn't": "was not",
    r"haven't": "have not",
    r"hasn't": "has not",
    r"couldn't": "could not",
    r"shouldn't": "should not",
    r"wouldn't": "would not"
}

lemmatizer = WordNetLemmatizer()

def clean_sentiment(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower()
    for pattern, rep in CONTRACTIONS.items():
        t = re.sub(pattern, rep, t)
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)
    tokens = t.split()
    filtered = [w for w in tokens if w not in SENTIMENT_STOPWORDS and len(w) > 1]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

tests = [
    'The assignments were not clear and instructions were not helpful.',
    'The lecturer did not explain concepts well at all.',
    'Very poor teaching quality and disorganized labs.',
    'I really loved the practical labs with Python and Docker.'
]

for s in tests:
    print('ORIG: ', s)
    print('CLEAN:', clean_sentiment(s))
