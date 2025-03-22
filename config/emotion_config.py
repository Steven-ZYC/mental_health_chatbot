# Emotion configuration and thresholds

# Mapping of emotions to their corresponding keywords and intensity levels
EMOTION_KEYWORDS = {
    'happy': {
        'keywords': ['joy', 'excited', 'delighted', 'pleased', 'content'],
        'intensity_threshold': 0.6
    },
    'sad': {
        'keywords': ['depressed', 'unhappy', 'miserable', 'down', 'upset'],
        'intensity_threshold': 0.4
    },
    'angry': {
        'keywords': ['furious', 'annoyed', 'irritated', 'frustrated', 'mad'],
        'intensity_threshold': 0.5
    },
    'fear': {
        'keywords': ['scared', 'anxious', 'worried', 'nervous', 'terrified'],
        'intensity_threshold': 0.4
    },
    'surprise': {
        'keywords': ['amazed', 'astonished', 'shocked', 'startled', 'stunned'],
        'intensity_threshold': 0.7
    },
    'disgust': {
        'keywords': ['repulsed', 'disgusted', 'revolted', 'appalled', 'horrified'],
        'intensity_threshold': 0.6
    },
    'neutral': {
        'keywords': ['okay', 'fine', 'normal', 'alright', 'calm'],
        'intensity_threshold': 0.3
    }
}

# Confidence threshold for emotion detection
CONFIDENCE_THRESHOLD = 0.65

# Emotion intensity levels
EMOTION_INTENSITY = {
    'LOW': 0.3,
    'MEDIUM': 0.6,
    'HIGH': 0.8
}

# Emotion state verification settings
VERIFICATION_CONFIG = {
    'visual_weight': 0.7,  # Weight for visual emotion detection
    'text_weight': 0.3,    # Weight for text-based emotion detection
    'agreement_threshold': 0.8  # Threshold for emotion agreement between visual and text
}