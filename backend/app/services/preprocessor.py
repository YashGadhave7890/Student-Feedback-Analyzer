import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure required NLTK resources
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
    }

DOMAIN_STOPWORDS = {"course", "unit", "class", "classes", "teacher", "lecturer", "learning", "teaching", "students", "student"}
ALL_STOPWORDS = STOP_WORDS.union(DOMAIN_STOPWORDS)

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
    USE_LEMMATIZER = True
except Exception:
    USE_LEMMATIZER = False

stemmer = PorterStemmer()


def clean_text_basic(text: str) -> str:
    """Basic normalization for LDA topic modeling."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_for_sentiment(text: str) -> str:
    """Enhanced cleaning for sentiment analysis with tokenization & lemmatization."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    # Filter stopwords but preserve negations if any
    filtered = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    
    if USE_LEMMATIZER:
        try:
            processed = [lemmatizer.lemmatize(w) for w in filtered]
        except Exception:
            processed = [stemmer.stem(w) for w in filtered]
    else:
        processed = [stemmer.stem(w) for w in filtered]
        
    return " ".join(processed)


def get_token_list(text: str) -> list[str]:
    """Tokenizes text for token explainability and n-gram exploration."""
    cleaned = clean_text_basic(text)
    return [w for w in cleaned.split() if w not in STOP_WORDS and len(w) > 2]
