# Safety configuration for mental health chatbot

# Keywords that indicate potential crisis situations
CRISIS_KEYWORDS = {
    'self_harm': [
        'suicide', 'kill myself', 'end my life', 'self harm',
        'hurt myself', 'die', 'death', 'overdose'
    ],
    'violence': [
        'hurt someone', 'kill', 'attack', 'violent',
        'weapon', 'fight', 'abuse'
    ],
    'emergency': [
        'emergency', 'crisis', 'urgent', 'immediate help',
        'critical', 'life-threatening'
    ]
}

# Response templates for crisis situations
CRISIS_RESPONSES = {
    'self_harm': (
        "I'm very concerned about what you're saying and your safety is important. "
        "Please know that help is available 24/7. Would you like me to provide you "
        "with crisis helpline numbers? In an emergency, please call 911 or your local "
        "emergency services immediately."
    ),
    'violence': (
        "I understand you're experiencing intense feelings. Your safety and the safety "
        "of others is crucial. Please reach out to emergency services (911) if you or "
        "anyone else is in immediate danger."
    ),
    'emergency': (
        "This sounds like a situation that requires immediate professional attention. "
        "Please contact emergency services (911) right away. Would you like information "
        "about crisis support services?"
    )
}

# Mental health resources and helplines
SUPPORT_RESOURCES = {
    'crisis_lines': {
        'national_suicide_prevention': '1-800-273-8255',
        'crisis_text_line': 'Text HOME to 741741',
        'domestic_violence': '1-800-799-7233',
    },
    'online_resources': {
        'mental_health_america': 'https://www.mhanational.org',
        'nami': 'https://www.nami.org',
        'psychology_today': 'https://www.psychologytoday.com'
    }
}

# Threshold settings for crisis detection
SAFETY_THRESHOLDS = {
    'crisis_confidence': 0.7,  # Minimum confidence score to trigger crisis response
    'escalation_threshold': 0.8,  # Threshold for escalating to crisis resources
    'consecutive_crisis_limit': 3  # Number of consecutive crisis indicators before escalation
}

# Safety monitoring configuration
MONITORING_CONFIG = {
    'check_frequency': 1,  # Check safety keywords every message
    'history_window': 5,   # Number of messages to maintain in safety history
    'cool_down_period': 300  # Seconds to wait before repeating crisis resources
}