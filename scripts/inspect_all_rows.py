import pandas as pd
import numpy as np

raw_path = 'data/202511-ft_bi1_bi2_course_evaluation.csv'
df = pd.read_csv(raw_path)

liked_col = 'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'
rec_col = 'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'

df['text'] = df[liked_col].fillna('') + ' ' + df[rec_col].fillna('')
df['text'] = df['text'].str.strip()
df = df[df['text'] != ''].reset_index(drop=True)

print(f"Total non-empty records: {len(df)}")

for idx, r in df.iterrows():
    liked = str(r[liked_col]).strip()
    rec = str(r[rec_col]).strip()
    orig_sent = str(r['sentiment']).strip().lower()
    rating = r['average_course_evaluation_rating']
    full_text = r['text']
    print(f"--- ID: {idx} | Orig: {orig_sent} | Rating: {rating:.2f} ---")
    print(f"  LIKED: {liked}")
    print(f"  REC:   {rec}")
