import pandas as pd
import os
from datetime import datetime

def save_record(text, screen, sleep, score):
    """Save prediction record to CSV"""
    os.makedirs("data", exist_ok=True)
    
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_preview": text[:50] + "..." if len(text) > 50 else text,
        "screen_hours": screen,
        "sleep_hours": sleep,
        "burnout_score": score
    }
    
    file_path = "data/history.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(file_path, index=False)
    return True

def load_history():
    """Load prediction history"""
    file_path = "data/history.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except:
            return None
    return None