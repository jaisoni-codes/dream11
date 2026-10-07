import pandas as pd
from datetime import datetime

class FeatureContext:
    def __init__(self, target_date):
        if isinstance(target_date, str):
            self.target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            self.target_date = target_date
            
    def filter_historical_data(self, df, date_column='date'):
        """
        Leakage wall: Only allows rows strictly before the target date.
        """
        # Ensure date_column is converted to datetime/date for comparison
        df_date = pd.to_datetime(df[date_column]).dt.date
        
        # STRICT CUTOFF: < target_date
        safe_df = df[df_date < self.target_date].copy()
        
        # Double check assertion
        if not safe_df.empty:
            max_date = pd.to_datetime(safe_df[date_column]).dt.date.max()
            assert max_date < self.target_date, "LEAKAGE DETECTED: Historical data contains records >= target date."
            
        return safe_df
