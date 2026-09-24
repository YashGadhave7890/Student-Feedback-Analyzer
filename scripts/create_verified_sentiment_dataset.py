import pandas as pd
import numpy as np

# Load original raw data (DO NOT MODIFY RAW DATA)
raw_df = pd.read_csv('data/202511-ft_bi1_bi2_course_evaluation.csv')

liked_col = 'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'
rec_col = 'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'

raw_df['text'] = raw_df[liked_col].fillna('') + ' ' + raw_df[rec_col].fillna('')
raw_df['text'] = raw_df['text'].str.strip()
raw_df = raw_df[raw_df['text'] != ''].reset_index(drop=True)

# Ground-truth semantic label verification based on actual textual meaning
# Annotation guidelines:
# 1. Positive: Strong praise/appreciation, satisfied tone, minor/no critique.
# 2. Neutral: Balanced constructive suggestions, factual/topical listings, neither strong praise nor strong complaints.
# 3. Negative: Clear critique, dissatisfaction, confusion, complaint about workload, long/confusing slides, difficult exams/pacing.

def get_verified_sentiment(row_idx, text, liked, rec, orig_sentiment, rating):
    liked_lower = str(liked).lower()
    rec_lower = str(rec).lower()
    full_lower = str(text).lower()
    
    # Analyze row by row systematically
    # Specific documented overrides and reasons:
    pass

