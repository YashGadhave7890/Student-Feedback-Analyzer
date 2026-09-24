import io
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


PRIORITY_KEYWORDS = [
    "feedback",
    "comment",
    "review",
    "response",
    "suggestion",
    "evaluation",
    "liked",
    "recommendation",
    "text",
    "opinion",
    "remarks",
]

DISALLOWED_KEYWORDS = [
    "timestamp",
    "date",
    "time",
    "student_id",
    "roll_number",
    "roll_no",
    "id",
    "email",
    "phone",
    "consent",
    "gender",
    "class_group",
    "group",
    "rating",
    "score",
    "marks",
    "grade",
    "status",
    "clean_sentiment_text",
    "clean_topic_text",
    "predicted_sentiment",
    "topic_label",
    "topic_probability",
    "dominant_topic",
    "sentiment_confidence",
]


class DatasetValidator:
    @staticmethod
    def score_column(col_name: str, series: pd.Series) -> float:
        name = col_name.lower().strip()
        score = 0.0

        if pd.api.types.is_numeric_dtype(series):
            return -1000.0

        for dis in DISALLOWED_KEYWORDS:
            if dis in name:
                score -= 300.0

        for pri in PRIORITY_KEYWORDS:
            if pri in name:
                score += 60.0

        if name in ["text", "feedback", "comments", "reviews", "student_feedback", "course_feedback"]:
            score += 100.0

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            return -1000.0

        sample = non_null.head(100)
        avg_words = sample.apply(lambda x: len(x.split())).mean()
        avg_chars = sample.apply(len).mean()

        if avg_words >= 3:
            score += 30.0
        if avg_words >= 6:
            score += 30.0
        if avg_chars >= 20:
            score += 20.0

        if avg_words < 1.4 and avg_chars < 8:
            score -= 200.0

        # Check date/timestamp pattern frequency
        date_like = sum(
            1
            for val in sample
            if any(sep in val for sep in ["-", "/", ":"])
            and any(c.isdigit() for c in val)
            and len(val.split()) <= 2
        )
        if date_like / len(sample) > 0.5:
            score -= 400.0

        return score

    @classmethod
    def find_best_text_column(cls, df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None

        scores = {col: cls.score_column(col, df[col]) for col in df.columns}
        sorted_cols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_cols and sorted_cols[0][1] > 0:
            return sorted_cols[0][0]

        # Fallback: check if any column passes validation
        for col in df.columns:
            is_valid, _ = cls.validate_text_column(df, col)
            if is_valid:
                return col

        return None

    @classmethod
    def validate_text_column(cls, df: pd.DataFrame, col_name: str) -> Tuple[bool, str]:
        if col_name not in df.columns:
            return False, f"Selected column '{col_name}' does not exist in dataset."

        series = df[col_name]
        name = col_name.lower().strip()

        # Disallow non-text columns
        if pd.api.types.is_numeric_dtype(series):
            return False, "Please select a feedback/text column containing natural-language responses."

        for dis in ["timestamp", "date", "student_id", "roll_number", "roll_no", "email", "phone", "consent", "gender"]:
            if dis in name:
                return False, "Please select a feedback/text column containing natural-language responses."

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            return False, "Please select a feedback/text column containing natural-language responses."

        sample = non_null.head(100)
        avg_words = sample.apply(lambda x: len(x.split())).mean()
        avg_chars = sample.apply(len).mean()

        numeric_like = sum(1 for v in sample if v.replace(".", "", 1).replace("-", "", 1).isdigit())
        date_like = sum(
            1
            for v in sample
            if any(sep in v for sep in ["-", "/", ":"])
            and any(c.isdigit() for c in v)
            and len(v.split()) <= 2
        )

        if (numeric_like + date_like) / len(sample) > 0.6:
            return False, "Please select a feedback/text column containing natural-language responses."

        if avg_words < 1.3 and avg_chars < 6:
            return False, "Please select a feedback/text column containing natural-language responses."

        return True, ""

    @classmethod
    def validate_csv(cls, content: bytes) -> Dict[str, Any]:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            return {
                "is_valid": False,
                "total_rows": 0,
                "detected_columns": [],
                "recommended_text_column": None,
                "missing_values_count": 0,
                "preview": [],
                "errors": [f"CSV parsing error: {str(e)}"],
            }

        if df.empty:
            return {
                "is_valid": False,
                "total_rows": 0,
                "detected_columns": [],
                "recommended_text_column": None,
                "missing_values_count": 0,
                "preview": [],
                "errors": ["Uploaded CSV file is empty."],
            }

        columns = list(df.columns)
        recommended_column = cls.find_best_text_column(df)
        missing_count = int(df[recommended_column].isna().sum()) if recommended_column in df.columns else 0

        # Create preview of first 5 rows
        preview_data = df.head(5).fillna("").to_dict(orient="records")

        errors = []
        if not recommended_column:
            errors.append("No natural-language text column could be automatically detected.")

        return {
            "is_valid": True,
            "total_rows": len(df),
            "detected_columns": columns,
            "recommended_text_column": recommended_column,
            "missing_values_count": missing_count,
            "preview": preview_data,
            "errors": errors,
        }


dataset_validator = DatasetValidator()
