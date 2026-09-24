import pandas as pd
import numpy as np

df = pd.read_csv('data/202511-ft_bi1_bi2_course_evaluation.csv')
liked_col = 'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'
rec_col = 'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'

df['text'] = df[liked_col].fillna('') + ' ' + df[rec_col].fillna('')
df['text'] = df['text'].str.strip()
df = df[df['text'] != ''].reset_index(drop=True)

print('=== 1. DATASET BASICS ===')
print('Total rows:', len(df))
print('Class distribution:')
print(df['sentiment'].value_counts())
print('Class percentages:')
print(df['sentiment'].value_counts(normalize=True) * 100)

print('\n=== 2. DUPLICATES ===')
dups = df[df.duplicated(subset=['text'], keep=False)]
print(f'Total duplicate text entries: {len(dups)}')

print('\n=== 3. SHORT TEXTS (< 20 chars) ===')
short = df[df['text'].str.len() < 20]
print(f'Total short texts: {len(short)}')
for idx, r in short.iterrows():
    print(f'  [{idx}] \"{r["text"]}\" -> {r["sentiment"]}')

print('\n=== 4. CONTRADICTORY LABELS DUE TO LIKERT THRESHOLDING ===')
for idx, r in df.iterrows():
    t_lower = r['text'].lower()
    s = r['sentiment']
    # If text is purely positive praise but labeled negative
    if s == 'negative' and any(w in t_lower for w in ['liked', 'enjoyed', 'great', 'well detailed', 'well explained', 'good']) and not any(w in t_lower for w in ['bad', 'poor', 'worst', 'terrible', 'hard', 'difficult', 'confusing']):
        print(f'  Negative mismatch [{idx}]: \"{r["text"]}\" (Rating: {r["average_course_evaluation_rating"]})')
    elif s == 'positive' and any(w in t_lower for w in ['confusing', 'difficult', 'hard to understand', 'too fast', 'unclear']) and not any(w in t_lower for w in ['great', 'excellent', 'amazing']):
        print(f'  Positive mismatch [{idx}]: \"{r["text"]}\" (Rating: {r["average_course_evaluation_rating"]})')

print('\n=== 5. INFERENCE TEST ON REALISTIC UNSEEN SAMPLES ===')
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.services.sentiment_engine import sentiment_engine
test_cases = [
    ("The course was fantastic and the professor explained everything clearly.", "positive"),
    ("I really loved the practical labs and hands-on exercises with Python.", "positive"),
    ("The lectures were average and pacing was okay.", "neutral"),
    ("It was standard course material, nothing particularly good or bad.", "neutral"),
    ("The slides were terribly confusing, difficult to follow and I learned nothing.", "negative"),
    ("Very poor teaching quality, disorganized labs and the instructor was unhelpful.", "negative"),
    ("The assignments were not clear and instructions were not helpful.", "negative"),
    ("The lecturer did not explain the concepts well at all.", "negative")
]

print(f"{'Text':<75} | {'Expected':<10} | {'Predicted':<10} | {'Conf':<6}")
print("-" * 110)
for text, exp in test_cases:
    res = sentiment_engine.predict_single(text)
    pred = res["sentiment"]
    conf = res["confidence"]
    status = "OK" if pred == exp else "FAIL"
    print(f"{text:<75} | {exp:<10} | {pred:<10} | {conf:<6.2f} [{status}]")
