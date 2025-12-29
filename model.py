import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

class BurnoutPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        print("✅ BERT model loaded successfully")
    
    def analyze_sentiment(self, text):
        """Extract emotional score from text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get CLS token embedding
        text_vector = outputs.pooler_output.numpy()[0]
        
        # Calculate emotional intensity (absolute mean)
        emotional_score = abs(np.mean(text_vector))
        
        # Normalize to 0-1 range
        emotional_score = min(max(emotional_score, 0), 1)
        
        return emotional_score, text_vector
    
    def calculate_screen_factor(self, screen_hours):
        """Calculate impact of screen time"""
        # Optimal: 4-6 hours, Risk: >8 hours
        if screen_hours <= 4:
            return 0.2
        elif screen_hours <= 6:
            return 0.4
        elif screen_hours <= 8:
            return 0.6
        elif screen_hours <= 10:
            return 0.8
        else:
            return 1.0
    
    def calculate_sleep_factor(self, sleep_hours):
        """Calculate impact of sleep duration"""
        # Optimal: 7-8 hours
        if sleep_hours >= 8:
            return 0.1
        elif sleep_hours >= 7:
            return 0.3
        elif sleep_hours >= 6:
            return 0.5
        elif sleep_hours >= 5:
            return 0.7
        else:
            return 0.9
    
    def predict(self, text, screen_hours, sleep_hours):
        """Main prediction function"""
        # Get emotional score
        emotional_score, _ = self.analyze_sentiment(text)
        
        # Calculate factors
        screen_factor = self.calculate_screen_factor(screen_hours)
        sleep_factor = self.calculate_sleep_factor(sleep_hours)
        
        # Weighted risk calculation
        risk_score = (
            0.45 * emotional_score +      # Emotional state (45%)
            0.35 * screen_factor +        # Screen time impact (35%)
            0.20 * sleep_factor           # Sleep deprivation (20%)
        )
        
        # Convert to percentage
        risk_percentage = min(max(risk_score, 0), 1) * 100
        
        return round(risk_percentage, 2), round(emotional_score, 4)

# Global predictor instance
predictor = BurnoutPredictor()

def predict_burnout(text, screen, sleep):
    """Wrapper function for Streamlit"""
    return predictor.predict(text, screen, sleep)