# Multi-language prompt templates for mental health chatbot

ENGLISH_TEMPLATES = {
    'base_template': (
        "You are a mental health support assistant. The user's current emotional state is [{emotion}]. "
        "Please generate a response based on the following principles:\n"
        "1. Express empathy in 1-2 sentences\n"
        "2. Ask an open-ended question to guide the user's thinking\n"
        "3. Provide simple coping suggestions\n\n"
        "Example dialogue:\n"
        "User: I feel very anxious\n"
        "Emotion: anxiety\n"
        "Assistant: Anxiety can indeed be overwhelming to bear. Have you noticed when this feeling tends "
        "to be strongest? Try taking three deep breaths and feel your feet connecting with the ground - "
        "this can help bring you back to the present moment.\n\n"
        "Current dialogue:\n"
        "User: {user_input}\n"
        "Emotion: {emotion}\n"
        "Assistant: "
    ),
    'context_template': "\nRecent conversation history (last 3 turns):\n{history}",
    
    'emotion_responses': {
        'happy': {
            'validation': "I can see that you're feeling positive right now.",
            'questions': [
                "What has contributed to your good mood today?",
                "How can you maintain this positive energy?",
                "Would you like to share what's making you feel happy?"
            ]
        },
        'sad': {
            'validation': "I understand that you're going through a difficult time.",
            'questions': [
                "Would you like to talk about what's troubling you?",
                "When did you start feeling this way?",
                "What usually helps you feel better when you're down?"
            ]
        },
        'angry': {
            'validation': "I can sense that you're feeling frustrated and angry.",
            'questions': [
                "What triggered this feeling of anger?",
                "How do you usually cope with anger?",
                "Would you like to explore what's behind this anger?"
            ]
        },
        'fear': {
            'validation': "It's completely normal to feel scared or anxious sometimes.",
            'questions': [
                "What specifically are you worried about?",
                "How does this fear affect your daily life?",
                "What helps you feel safe and secure?"
            ]
        },
        'surprise': {
            'validation': "That must have been quite unexpected for you.",
            'questions': [
                "How are you processing this surprise?",
                "What thoughts are going through your mind?",
                "How do you usually handle unexpected situations?"
            ]
        },
        'disgust': {
            'validation': "I can see this situation is really bothering you.",
            'questions': [
                "What about this situation troubles you the most?",
                "How can we help you feel more comfortable?",
                "Would you like to talk about what's causing these feelings?"
            ]
        },
        'neutral': {
            'validation': "I notice you're feeling quite balanced right now.",
            'questions': [
                "How has your day been going?",
                "Is there anything specific you'd like to discuss?",
                "What's on your mind?"
            ]
        }
    },
    
    'coping_strategies': {
        'general': [
            "Take a few deep breaths - in through your nose, out through your mouth.",
            "Try grounding yourself by naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            "Consider taking a short walk or doing some light stretching."
        ],
        'stress_relief': [
            "Practice progressive muscle relaxation - tense and relax each muscle group.",
            "Take a mindful moment to focus on your breathing.",
            "Write down your thoughts and feelings in a journal."
        ],
        'mood_improvement': [
            "Engage in an activity you enjoy, even if just for a few minutes.",
            "Connect with a friend or family member.",
            "Listen to music that matches or lifts your mood."
        ]
    }
}