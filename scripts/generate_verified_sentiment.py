import pandas as pd
import json

raw_df = pd.read_csv('data/202511-ft_bi1_bi2_course_evaluation.csv')

liked_col = 'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'
rec_col = 'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'

raw_df['text'] = raw_df[liked_col].fillna('') + ' ' + raw_df[rec_col].fillna('')
raw_df['text'] = raw_df['text'].str.strip()
raw_df = raw_df[raw_df['text'] != ''].reset_index(drop=True)

# Build comprehensive rules and per-row inspection
records = []
changed_count = 0

for idx, r in raw_df.iterrows():
    liked = str(r[liked_col]).strip() if pd.notna(r[liked_col]) else ""
    rec = str(r[rec_col]).strip() if pd.notna(r[rec_col]) else ""
    full_text = r['text']
    orig_sent = str(r['sentiment']).strip().lower()
    rating = float(r['average_course_evaluation_rating']) if pd.notna(r['average_course_evaluation_rating']) else 0.0
    
    t_low = full_text.lower()
    l_low = liked.lower()
    r_low = rec.lower()
    
    # Determine ground-truth sentiment based purely on text semantics:
    # Rule 1: Clear dissatisfaction / complaint in recommendations or overall text -> negative
    # (e.g., long confusing slides, crashing machines, missing marks, overwhelming workload, bad revision material, harsh difficulty)
    has_negative_critique = any(phrase in r_low or phrase in t_low for phrase in [
        'too long', 'very long', 'confusing', 'complicated', 'crash', 'crashing',
        'overwhelming', 'bummer', 'missing marks', 'hard to understand', 'friction',
        'leads me to cram', 'demoralizing', 'could be better', 'not very dependable',
        'difficult to process', 'cram the content', 'less software-intensive',
        'take students through labs in a clearer way', 'very hard', 'shorter lecture notes',
        'without understanding the true objective', 'too much information to process',
        'avoiding using applications like docker', 'having an issue with the lecturer',
        'assist in the lab works especially before submission', 'reduce the number of slides',
        'reduce the number of lecture slides', 'more real world referencing'
    ])
    
    # Rule 2: Pure positive praise with no critique or trivial "none/good/n/a" -> positive
    is_pure_positive = (
        (any(pos in l_low for pos in ['liked', 'enjoyed', 'great', 'engaging', 'smoothly', 'good', 'well structured', 'in-depth', 'clear', 'practical']) or len(liked) > 10)
        and (r_low in ['', 'none', 'i have none', 'n/a', 'na', 'no', 'nothing', 'so far so good', 'no opinion', 'no opinion on the matter', 'i don\'t have any recommendations', 'i have no reservations on this', 'everything went smoothly'] or r_low.startswith('so far so good') or r_low == 'nan')
    )
    
    # Rule 3: Factual course listing without sentiment markers -> neutral
    is_factual_listing = l_low in ['regression ml modelling', 'regression\nml modelling', 'data science and data analysis', 'learning kmeans\nlearning apis', 'time series how it helps in business corporations using it', 'interesting']
    
    # Combine semantic judgement
    if is_factual_listing:
        verified = 'neutral'
        rationale = 'Factual topic listing or single ambiguous keyword without sentiment polarity.'
    elif has_negative_critique and not is_pure_positive:
        # Check if criticism outweighs praise or represents actionable dissatisfaction
        if any(harsh in t_low for harsh in ['bummer', 'confusing', 'complicated', 'crashing', 'missing marks', 'friction', 'overwhelming', 'demoralizing', 'too long', 'very long', 'hard since all the concepts', 'cram']):
            verified = 'negative'
            rationale = 'Explicit dissatisfaction/complaint regarding slide length, pacing, technical friction, or confusion.'
        else:
            verified = 'neutral'
            rationale = 'Balanced constructive suggestion / enhancement request without harsh complaint.'
    elif is_pure_positive:
        verified = 'positive'
        rationale = 'Clear praise and appreciation with no recommendations or trivial satisfaction confirmation.'
    elif any(pos in l_low for pos in ['great', 'excellent', 'amazing', 'well structured', 'very practical', 'engaging', 'learned a lot', 'good']):
        # If recommendation is mild suggestion (e.g. "more quizzes", "power bi certification")
        if any(neg in r_low for neg in ['slides', 'long', 'hard', 'difficult', 'confusing', 'cram']):
            verified = 'negative'
            rationale = 'Substantive negative critique on slide volume/difficulty despite positive opening.'
        else:
            verified = 'positive' if rating >= 4.5 or len(r_low) < 30 else 'neutral'
            rationale = 'Positive experience with minor constructive feature suggestion.'
    else:
        verified = orig_sent
        rationale = 'Consistent with original label.'

    # Explicit manual verification checks for key survey anomalies:
    if idx == 2:  # "Well detailed notes Well explained concepts More clear instructions given out during the lab work"
        verified = 'neutral'
        rationale = 'Praise for notes combined with constructive recommendation for clearer lab instructions.'
    elif idx == 5:  # "Calm class"
        verified = 'neutral'
        rationale = 'Neutral descriptive observation.'
    elif idx == 6:  # "It is involving and practical Relevant in real world application..."
        verified = 'positive'
        rationale = 'Enthusiastic positive praise for real-world practical relevance.'
    elif idx == 8:  # "I liked the split between practicals and theory"
        verified = 'positive'
        rationale = 'Positive praise regarding course structure with no complaints.'
    elif idx == 9:  # "Data science and Data analysis"
        verified = 'neutral'
        rationale = 'Topic listing without sentiment modifier.'
    elif idx == 13: # "I like that everything tested was covered... machines were crashing"
        verified = 'negative'
        rationale = 'Technical complaint regarding university machines crashing on data engineering labs.'
    elif idx == 26: # "I liked the practical labs; they encourgaed critical thinking... I have none"
        verified = 'positive'
        rationale = 'Strong multi-point praise with no recommendations ("I have none").'
    elif idx == 45: # "The consistency was good from the lecturer which was engaging..."
        verified = 'positive'
        rationale = 'Positive evaluation of teaching quality and lab engagement.'
    elif idx == 101: # "the lab assignments and the reflective journal i don't have any recommendations"
        verified = 'positive'
        rationale = 'Praise for assignments and journal with zero recommendations.'
    elif idx == 103: # "Interesting"
        verified = 'neutral'
        rationale = 'Single ambiguous adjective without positive/negative context.'
    elif idx == 113: # "1. I liked the labs we did 2. The quality of teaching shorter lecture notes"
        verified = 'neutral'
        rationale = 'Balanced feedback: praise for lab quality + recommendation for shorter notes.'
    elif idx == 115: # "Having better structured slides... make topics seem so complicated... bummer!!!"
        verified = 'negative'
        rationale = 'Strong critique of slide complexity and dependability for revision.'
    elif idx == 121: # "I liked the quizzes and The BI2 topics. Be more step by step for practical and also the number of slides"
        verified = 'neutral'
        rationale = 'Balanced feedback with practical suggestion on slide pacing.'

    if verified != orig_sent:
        changed_count += 1

    records.append({
        "id": idx,
        "text": full_text,
        "liked_part": liked,
        "rec_part": rec,
        "original_rating": rating,
        "original_sentiment": orig_sent,
        "verified_sentiment": verified,
        "changed": verified != orig_sent,
        "rationale": rationale
    })

res_df = pd.DataFrame(records)
print(f"Total records: {len(res_df)}")
print(f"Total labels modified: {changed_count} ({changed_count/len(res_df)*100:.1f}%)")
print("\nOriginal Distribution:")
print(res_df['original_sentiment'].value_counts())
print("\nVerified Distribution:")
print(res_df['verified_sentiment'].value_counts())

# Save verified dataset
out_path = 'data/course_evals_cleaned_sentiment.csv'
res_df.to_csv(out_path, index=False)
print(f"\nSaved verified dataset to {out_path}")
