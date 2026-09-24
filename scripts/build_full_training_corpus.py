"""
Construct the Unified Sentiment Training Corpus.

This script combines:
1. Original verified survey records (from `course_evals_cleaned_sentiment.csv`)
2. Atomic responses from survey questions f_3 (liked) and f_4 (recommendations)
3. Manually verified supplemental student feedback (from `supplemental_student_feedback.csv`)

Every record maintains provenance metadata, source tracking, and verified labels.
Output: `data/sentiment_training_corpus.csv`
"""

from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_SURVEY_FILE = BASE_DIR / "data" / "202511-ft_bi1_bi2_course_evaluation.csv"
CLEANED_SURVEY_FILE = BASE_DIR / "data" / "course_evals_cleaned_sentiment.csv"
SUPPLEMENTAL_FILE = BASE_DIR / "data" / "supplemental_student_feedback.csv"
OUTPUT_CORPUS_FILE = BASE_DIR / "data" / "sentiment_training_corpus.csv"


def classify_f3(text: str) -> str:
    """Classify 'What did you like about the course' statements."""
    t = str(text).strip().lower()
    if not t or t in ["none", "nothing", "not sure", "nil", "n/a"]:
        return "neutral"
    # Pure topic lists without praise are neutral
    topic_only_patterns = [
        r"^(data science and data analysis|learning kmeans\s*learning apis|dashboarding|predictive modeling|power bi|docker)$",
        r"^(natural language processing|time series|machine learning)$"
    ]
    for p in topic_only_patterns:
        if re.match(p, t):
            return "neutral"
    return "positive"


def classify_f4(text: str) -> str:
    """Classify 'Recommendations to improve' statements."""
    t = str(text).strip().lower()
    if not t or t in ["none", "nothing", "not sure", "nil", "n/a", "everything's seemingly fine", "everything was fine", "all good"]:
        return "neutral"
    
    # Severe complaints / negative feedback
    severe_negative_indicators = [
        "crash", "crashing", "unfair", "unbearable", "overwhelming", "disorganized", 
        "couldn't do", "did not understand", "could not do", "too fast", "very long", 
        "800 slides", "excess", "unbelievably", "lost", "frustrat", "poor"
    ]
    for ind in severe_negative_indicators:
        if ind in t:
            return "negative"
            
    # Default constructive recommendation is neutral
    return "neutral"


def main():
    corpus_records = []
    seen_texts = set()

    # 1. Original Verified Survey Combined Records
    df_cleaned = pd.read_csv(CLEANED_SURVEY_FILE)
    for _, row in df_cleaned.iterrows():
        txt = str(row['text']).strip()
        if txt and txt not in seen_texts:
            seen_texts.add(txt)
            corpus_records.append({
                "id": f"orig_comb_{row['id']}",
                "text": txt,
                "sentiment": row['verified_sentiment'],
                "source": "original_survey_combined",
                "rationale": row.get('cleaning_rationale', 'Combined student survey response')
            })

    # 2. Atomic F3 and F4 responses from raw survey
    df_raw = pd.read_csv(RAW_SURVEY_FILE)
    f3_col = [c for c in df_raw.columns if "f_3" in c][0]
    f4_col = [c for c in df_raw.columns if "f_4" in c][0]

    for idx, row in df_raw.iterrows():
        # F3 (Liked)
        f3_val = str(row[f3_col]).strip()
        if f3_val and f3_val.lower() not in ["none", "nan", "nothing", "nil", "n/a", "not sure"] and len(f3_val) > 4:
            if f3_val not in seen_texts:
                seen_texts.add(f3_val)
                corpus_records.append({
                    "id": f"orig_f3_{idx+1}",
                    "text": f3_val,
                    "sentiment": classify_f3(f3_val),
                    "source": "original_survey_f3",
                    "rationale": "Atomic student response to what they liked in the course"
                })

        # F4 (Recommendations)
        f4_val = str(row[f4_col]).strip()
        if f4_val and f4_val.lower() not in ["none", "nan", "nothing", "nil", "n/a", "not sure"] and len(f4_val) > 4:
            if f4_val not in seen_texts:
                seen_texts.add(f4_val)
                corpus_records.append({
                    "id": f"orig_f4_{idx+1}",
                    "text": f4_val,
                    "sentiment": classify_f4(f4_val),
                    "source": "original_survey_f4",
                    "rationale": "Atomic student response to recommendations/improvements"
                })

    # 3. Supplemental Verified Records
    df_supp = pd.read_csv(SUPPLEMENTAL_FILE)
    for _, row in df_supp.iterrows():
        txt = str(row['text']).strip()
        if txt and txt not in seen_texts:
            seen_texts.add(txt)
            corpus_records.append({
                "id": row['id'],
                "text": txt,
                "sentiment": row['sentiment'],
                "source": "supplemental_verified",
                "rationale": row['rationale']
            })

    out_df = pd.DataFrame(corpus_records)
    out_df.to_csv(OUTPUT_CORPUS_FILE, index=False)

    print("=" * 75)
    print("UNIFIED SENTIMENT TRAINING CORPUS GENERATION")
    print("=" * 75)
    print(f"Output File: {OUTPUT_CORPUS_FILE}")
    print(f"Total Unique Training Records: {len(out_df)}")
    print("\nSource Breakdown:")
    print(out_df['source'].value_counts())
    print("\nOverall Class Distribution:")
    print(out_df['sentiment'].value_counts())
    print("\nOverall Class Percentages:")
    print(out_df['sentiment'].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()
