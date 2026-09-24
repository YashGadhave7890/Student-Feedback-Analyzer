"""
Build Manually Verified Supplemental Student Feedback Dataset.

This script creates `data/supplemental_student_feedback.csv` containing realistic,
manually verified student evaluation statements across multiple academic domains
(teaching quality, lab work, assignments, pacing, course organization, communication).

Every record contains:
- `id`: unique identifier (supp_001 .. supp_xxx)
- `text`: the student feedback text
- `sentiment`: manually verified sentiment ('positive', 'neutral', 'negative')
- `category`: domain category (e.g. 'teaching_quality', 'labs', 'pacing', 'materials', 'assessment', 'communication')
- `rationale`: documented reasoning for label assignment
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "supplemental_student_feedback.csv"

SUPPLEMENTAL_RECORDS = [
    # =========================================================================
    # NEGATIVE EXAMPLES (Realistic complaints, frustration, poor quality)
    # =========================================================================
    {
        "id": "supp_neg_001",
        "text": "The professor spoke too fast, skipped key derivations, and dismissed questions during lecture.",
        "sentiment": "negative",
        "category": "teaching_quality",
        "rationale": "Clear dissatisfaction with teaching pace, explanation depth, and student interaction."
    },
    {
        "id": "supp_neg_002",
        "text": "The lab instructions were disorganized, outdated, and full of broken links and syntax errors.",
        "sentiment": "negative",
        "category": "labs",
        "rationale": "Explicit criticism of lab material quality and lack of maintenance."
    },
    {
        "id": "supp_neg_003",
        "text": "Grading rubrics were extremely vague and assignments took over two months to be returned.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "Strong dissatisfaction with unfair evaluation standards and delayed feedback."
    },
    {
        "id": "supp_neg_004",
        "text": "The lecture slides were overwhelming with over 150 dense text-heavy slides per session.",
        "sentiment": "negative",
        "category": "materials",
        "rationale": "Severe complaint regarding unmanageable slide volume and poor pedagogical presentation."
    },
    {
        "id": "supp_neg_005",
        "text": "The virtual machine environment kept crashing and we received no technical support during the lab.",
        "sentiment": "negative",
        "category": "infrastructure",
        "rationale": "Technical failure obstructing learning without adequate teaching assistance."
    },
    {
        "id": "supp_neg_006",
        "text": "Teaching assistants were unhelpful and never responded to questions posted on the discussion forum.",
        "sentiment": "negative",
        "category": "communication",
        "rationale": "Explicit grievance regarding unavailable and unsupportive teaching staff."
    },
    {
        "id": "supp_neg_007",
        "text": "The workload was completely unreasonable with three major project deadlines falling in the same week.",
        "sentiment": "negative",
        "category": "pacing",
        "rationale": "Severe scheduling issue causing excessive student stress and unmanageable workload."
    },
    {
        "id": "supp_neg_008",
        "text": "Exams tested minor footnotes rather than the core analytical principles taught during lectures.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "Unfair assessment alignment that contradicts learning objectives."
    },
    {
        "id": "supp_neg_009",
        "text": "The lecturer frequently arrived 20 minutes late and cancelled multiple office hour sessions.",
        "sentiment": "negative",
        "category": "teaching_quality",
        "rationale": "Unprofessional attendance and lack of student availability."
    },
    {
        "id": "supp_neg_010",
        "text": "The code samples provided did not run on Python 3 and had missing dependency instructions.",
        "sentiment": "negative",
        "category": "materials",
        "rationale": "Defective educational resources hindering assignment completion."
    },
    {
        "id": "supp_neg_011",
        "text": "Very poor explanation of neural networks and time series analysis. I was completely lost.",
        "sentiment": "negative",
        "category": "teaching_quality",
        "rationale": "Direct expression of confusion caused by deficient instructional delivery."
    },
    {
        "id": "supp_neg_012",
        "text": "Group assignments were unfair because free-riders received identical marks without contributing.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "Systemic inequity in group grading policy."
    },
    {
        "id": "supp_neg_013",
        "text": "Audio quality in the recorded lectures was awful with constant background noise and echo.",
        "sentiment": "negative",
        "category": "materials",
        "rationale": "Unusable multimedia resources impairing study."
    },
    {
        "id": "supp_neg_014",
        "text": "The textbook was mandatory and expensive but never actually referenced or used in class.",
        "sentiment": "negative",
        "category": "materials",
        "rationale": "Unnecessary financial burden with zero pedagogical utility."
    },
    {
        "id": "supp_neg_015",
        "text": "Not enough time was given to finish the coding exam and the server crashed during submission.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "High-stakes testing malfunction and unreasonable time constraint."
    },
    {
        "id": "supp_neg_016",
        "text": "I felt unprepared for the real world because practical coding was barely covered.",
        "sentiment": "negative",
        "category": "curriculum",
        "rationale": "Disappointment with excessive theory and lack of practical skill development."
    },
    {
        "id": "supp_neg_017",
        "text": "The feedback on my essay was just a grade with zero written comments or explanation.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "Absence of formative feedback preventing student learning."
    },
    {
        "id": "supp_neg_018",
        "text": "Lectures were monotonous and simply reading bullet points off the slides with no enthusiasm.",
        "sentiment": "negative",
        "category": "teaching_quality",
        "rationale": "Lack of engagement and ineffective pedagogical delivery."
    },
    {
        "id": "supp_neg_019",
        "text": "Prerequisites were not clearly communicated and the math background assumed was too advanced.",
        "sentiment": "negative",
        "category": "curriculum",
        "rationale": "Misleading course requirements leading to extreme student struggle."
    },
    {
        "id": "supp_neg_020",
        "text": "The course felt like a disorganized crash course where topics were randomly thrown together.",
        "sentiment": "negative",
        "category": "course_structure",
        "rationale": "Lack of coherent syllabus progression and thematic structure."
    },
    {
        "id": "supp_neg_021",
        "text": "The database lab server was constantly down during peak submission hours.",
        "sentiment": "negative",
        "category": "infrastructure",
        "rationale": "Unreliable lab infrastructure severely impacting student deliverables."
    },
    {
        "id": "supp_neg_022",
        "text": "Not worth the tuition fees; the content can easily be learned on free YouTube tutorials.",
        "sentiment": "negative",
        "category": "general",
        "rationale": "Low perceived course value and quality."
    },
    {
        "id": "supp_neg_023",
        "text": "The instructor was dismissive of student difficulties and refused to clarify assignment requirements.",
        "sentiment": "negative",
        "category": "teaching_quality",
        "rationale": "Unsupportive educator behavior creating a hostile learning environment."
    },
    {
        "id": "supp_neg_024",
        "text": "Too much theory and unnecessary derivations without ever demonstrating how to apply them in code.",
        "sentiment": "negative",
        "category": "curriculum",
        "rationale": "Imbalance toward non-practical content."
    },
    {
        "id": "supp_neg_025",
        "text": "The final exam contained questions from topics explicitly excluded from the revision outline.",
        "sentiment": "negative",
        "category": "assessment",
        "rationale": "Breach of syllabus guidelines in formal evaluation."
    },

    # =========================================================================
    # POSITIVE EXAMPLES (Appreciation, satisfaction, high effectiveness)
    # =========================================================================
    {
        "id": "supp_pos_001",
        "text": "The professor gave fantastic lectures with intuitive explanations and real-world industrial case studies.",
        "sentiment": "positive",
        "category": "teaching_quality",
        "rationale": "Enthusiastic praise for instructional clarity and practical industry relevance."
    },
    {
        "id": "supp_pos_002",
        "text": "Hands-on Docker and Power BI lab sessions were extremely engaging and deepened my understanding.",
        "sentiment": "positive",
        "category": "labs",
        "rationale": "Strong positive appraisal of technical tool mastery and practical exercises."
    },
    {
        "id": "supp_pos_003",
        "text": "Teaching assistants provided prompt, thorough, and highly encouraging feedback on all assignments.",
        "sentiment": "positive",
        "category": "communication",
        "rationale": "Appreciation for supportive mentorship and quality feedback."
    },
    {
        "id": "supp_pos_004",
        "text": "The course materials were exceptionally well structured, concise, and easy to navigate.",
        "sentiment": "positive",
        "category": "materials",
        "rationale": "High satisfaction with syllabus organization and document design."
    },
    {
        "id": "supp_pos_005",
        "text": "I really enjoyed how theory in morning sessions was immediately reinforced with afternoon coding labs.",
        "sentiment": "positive",
        "category": "course_structure",
        "rationale": "Praise for pedagogical alignment between theoretical concepts and practical application."
    },
    {
        "id": "supp_pos_006",
        "text": "Clear grading criteria and helpful rubrics made expectations transparent and fair from day one.",
        "sentiment": "positive",
        "category": "assessment",
        "rationale": "Appreciation for fair and transparent grading."
    },
    {
        "id": "supp_pos_007",
        "text": "The instructor fostered an inclusive and highly participative classroom atmosphere where everyone felt comfortable.",
        "sentiment": "positive",
        "category": "teaching_quality",
        "rationale": "Praise for positive learning culture and student engagement."
    },
    {
        "id": "supp_pos_008",
        "text": "Step-by-step Jupyter notebooks and code walkthroughs made complex algorithms accessible and fun.",
        "sentiment": "positive",
        "category": "materials",
        "rationale": "Satisfaction with guided interactive coding resources."
    },
    {
        "id": "supp_pos_009",
        "text": "Outstanding course that gave me the exact skills and portfolio projects needed for data engineering internships.",
        "sentiment": "positive",
        "category": "general",
        "rationale": "High value career preparedness and practical competency development."
    },
    {
        "id": "supp_pos_010",
        "text": "The lecturer always made time for questions and held extra review sessions before exams.",
        "sentiment": "positive",
        "category": "teaching_quality",
        "rationale": "Appreciation for educator dedication and supplementary exam preparation."
    },
    {
        "id": "supp_pos_011",
        "text": "Great balance of individual practice and collaborative teamwork on data pipelines.",
        "sentiment": "positive",
        "category": "course_structure",
        "rationale": "Positive evaluation of collaborative learning."
    },
    {
        "id": "supp_pos_012",
        "text": "The quizzes after each module were super helpful for self-assessing understanding before moving forward.",
        "sentiment": "positive",
        "category": "assessment",
        "rationale": "Appreciation for formative modular assessment."
    },
    {
        "id": "supp_pos_013",
        "text": "Excellent explanations of difficult topics like clustering, dimensionality reduction, and API deployment.",
        "sentiment": "positive",
        "category": "teaching_quality",
        "rationale": "Specific praise for conceptual clarity on advanced subjects."
    },
    {
        "id": "supp_pos_014",
        "text": "I loved the capstone project because it allowed us to work with real company datasets.",
        "sentiment": "positive",
        "category": "labs",
        "rationale": "Enthusiastic engagement with authentic industry problem solving."
    },
    {
        "id": "supp_pos_015",
        "text": "The instructor was very energetic, passionate, and inspiring throughout the entire semester.",
        "sentiment": "positive",
        "category": "teaching_quality",
        "rationale": "Positive educator charisma and motivating instructional delivery."
    },

    # =========================================================================
    # NEUTRAL EXAMPLES (Factual, descriptive, balanced constructive suggestions)
    # =========================================================================
    {
        "id": "supp_neu_001",
        "text": "The syllabus covered regression, clustering, time series forecasting, and natural language processing.",
        "sentiment": "neutral",
        "category": "curriculum",
        "rationale": "Purely descriptive topical listing without sentiment expression."
    },
    {
        "id": "supp_neu_002",
        "text": "Lectures were scheduled twice a week on Tuesdays and Thursdays from two to four in the afternoon.",
        "sentiment": "neutral",
        "category": "general",
        "rationale": "Factual logistical statement."
    },
    {
        "id": "supp_neu_003",
        "text": "The course utilized Python, R Studio, Jupyter Notebooks, and PostgreSQL for assignments.",
        "sentiment": "neutral",
        "category": "materials",
        "rationale": "Factual list of software tools and technologies."
    },
    {
        "id": "supp_neu_004",
        "text": "Assessment consisted of three quizzes, two lab submissions, and a final written examination.",
        "sentiment": "neutral",
        "category": "assessment",
        "rationale": "Descriptive course evaluation breakdown."
    },
    {
        "id": "supp_neu_005",
        "text": "The pacing was standard and lecture slides were uploaded to the portal after class.",
        "sentiment": "neutral",
        "category": "general",
        "rationale": "Neutral observation of course routine and administration."
    },
    {
        "id": "supp_neu_006",
        "text": "Adding a few more practice questions at the end of each slide deck would be helpful for revision.",
        "sentiment": "neutral",
        "category": "recommendation",
        "rationale": "Polite constructive suggestion without negative grievance or emotional polarity."
    },
    {
        "id": "supp_neu_007",
        "text": "Consider allocating fifteen minutes at the end of class for group discussion and question review.",
        "sentiment": "neutral",
        "category": "recommendation",
        "rationale": "Constructive pedagogical recommendation."
    },
    {
        "id": "supp_neu_008",
        "text": "The content met standard curriculum requirements and followed the published syllabus outline.",
        "sentiment": "neutral",
        "category": "curriculum",
        "rationale": "Objective statement of course alignment."
    },
    {
        "id": "supp_neu_009",
        "text": "Recommended readings were taken from the standard Business Intelligence textbook chapters three through eight.",
        "sentiment": "neutral",
        "category": "materials",
        "rationale": "Factual citation of reading materials."
    },
    {
        "id": "supp_neu_010",
        "text": "It would be nice to have optional advanced exercises for students who finish the baseline lab early.",
        "sentiment": "neutral",
        "category": "recommendation",
        "rationale": "Constructive extension suggestion without dissatisfaction."
    },
    {
        "id": "supp_neu_011",
        "text": "The course is divided into Business Intelligence I for theory and Business Intelligence II for applications.",
        "sentiment": "neutral",
        "category": "curriculum",
        "rationale": "Factual organizational description."
    },
    {
        "id": "supp_neu_012",
        "text": "Students worked in pairs during in-person lab sessions.",
        "sentiment": "neutral",
        "category": "labs",
        "rationale": "Objective operational statement."
    },
    {
        "id": "supp_neu_013",
        "text": "Weekly office hours were conducted in room 302 and via video conference.",
        "sentiment": "neutral",
        "category": "communication",
        "rationale": "Factual scheduling information."
    },
    {
        "id": "supp_neu_014",
        "text": "A brief summary cheat sheet for Docker commands would be a convenient reference for future cohorts.",
        "sentiment": "neutral",
        "category": "recommendation",
        "rationale": "Constructive suggestion for auxiliary material."
    },
    {
        "id": "supp_neu_015",
        "text": "Course grades were weighted fifty percent coursework and fifty percent final assessment.",
        "sentiment": "neutral",
        "category": "assessment",
        "rationale": "Factual grading rubric description."
    }
]


def main():
    df = pd.DataFrame(SUPPLEMENTAL_RECORDS)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Created supplemental dataset at: {OUTPUT_FILE}")
    print(f"Total supplemental records: {len(df)}")
    print("\nSupplemental Class Distribution:")
    print(df['sentiment'].value_counts())
    print("\nSupplemental Category Distribution:")
    print(df['category'].value_counts())


if __name__ == "__main__":
    main()
