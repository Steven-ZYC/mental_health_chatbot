from typing import Dict, List, Tuple
import re
from config.emotion_config import EMOTION_KEYWORDS, CONFIDENCE_THRESHOLD

class LanguageProcessor:
    def __init__(self):
        self.emotion_keywords = EMOTION_KEYWORDS
        self.confidence_threshold = CONFIDENCE_THRESHOLD
    
    def detect_text_emotion(self, text: str) -> Tuple[str, float]:
        """Detect emotion from text input using keyword matching and intensity analysis."""
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        # Calculate emotion scores based on keyword matches
        for emotion, config in self.emotion_keywords.items():
            keywords = config['keywords']
            intensity_threshold = config['intensity_threshold']
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                # Calculate confidence score based on matches and intensity
                base_score = matches / len(keywords)
                emotion_scores[emotion] = min(base_score + intensity_threshold, 1.0)
        
        # Get emotion with highest score
        detected_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion, confidence = detected_emotion
        
        # Return neutral if confidence is below threshold
        if confidence < self.confidence_threshold:
            return 'neutral', confidence
        
        return emotion, confidence
    
    def analyze_sentiment_intensity(self, text: str) -> float:
        """Analyze the intensity of emotional expression in text."""
        # Simple intensity analysis based on:
        # - Exclamation marks
        # - Capitalization
        # - Repetition of letters
        # - Intensity words
        
        intensity = 0.0
        
        # Check for exclamation marks
        exclamations = text.count('!')
        intensity += min(exclamations * 0.1, 0.3)
        
        # Check for capitalized words
        words = text.split()
        caps_ratio = sum(1 for word in words if word.isupper()) / len(words) if words else 0
        intensity += caps_ratio * 0.2
        
        # Check for letter repetition (e.g., 'sooo', 'noooo')
        repeats = len(re.findall(r'(\w)\1{2,}', text))
        intensity += min(repeats * 0.1, 0.2)
        
        # Check for intensity words
        intensity_words = ['very', 'extremely', 'really', 'so', 'totally', 'absolutely']
        intensity_count = sum(1 for word in intensity_words if word in text.lower())
        intensity += min(intensity_count * 0.1, 0.3)
        
        return min(intensity, 1.0)
    
    def extract_crisis_indicators(self, text: str) -> List[str]:
        """Extract potential crisis indicators from text."""
        from config.safety_config import CRISIS_KEYWORDS
        
        indicators = []
        text = text.lower()
        
        for category, keywords in CRISIS_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                indicators.append(category)
        
        return indicators
    
    def get_conversation_context(self, history: List[Dict[str, str]], window_size: int = 3) -> str:
        """Generate conversation context from history."""
        from config.prompt_templates import ENGLISH_TEMPLATES
        
        if not history:
            return ""
        
        # Get last N turns
        recent_history = history[-window_size:]
        
        # Format history using context template
        formatted_history = "\n".join(
            f"User: {turn['user_input']}\n"
            f"Emotion: {turn['emotion']}\n"
            f"Assistant: {turn['response']}"
            for turn in recent_history
        )
        
        return ENGLISH_TEMPLATES['context_template'].format(history=formatted_history)