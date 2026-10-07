from datetime import datetime

HARD_CUTOFF_STR = "2024-06-30"
HARD_CUTOFF = datetime.strptime(HARD_CUTOFF_STR, "%Y-%m-%d").date()

def validate_cutoff(date_to_check):
    """Raises ValueError if date is after or on the HARD_CUTOFF (if strict) or just greater.
    The TDD says: No training data after 2024-06-30. 
    So matches strictly AFTER 2024-06-30 are forbidden.
    """
    if isinstance(date_to_check, str):
        date_to_check = datetime.strptime(date_to_check, "%Y-%m-%d").date()
    if date_to_check > HARD_CUTOFF:
        raise ValueError(f"Data leak: date {date_to_check} is strictly after HARD_CUTOFF {HARD_CUTOFF}")
    return True
