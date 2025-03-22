from typing import Dict, List, Optional
from config.prompt_templates import ENGLISH_TEMPLATES
from config.safety_config import CRISIS_RESPONSES, SUPPORT_RESOURCES
import random

class ResponseGenerator:
    def __init__(self):
        self.templates = ENGLISH_TEMPLATES
        self.crisis_responses = CRISIS_RESPONSES
        self.support_resources = SUPPORT_RESOURCES
    
    def generate_response(self, user_input: str, emotion: str,
                         crisis_indicators: List[str],
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate an empathetic response based on user input and emotional state."""
        # Check for crisis indicators first
        if crisis_indicators:
            return self._generate_crisis_response(crisis_indicators)
        
        # Get emotion-specific responses
        emotion_responses = self.templates['emotion_responses'].get(
            emotion,
            self.templates['emotion_responses']['neutral']
        )
        
        # Build response components
        validation = emotion_responses['validation']
        question = random.choice(emotion_responses['questions'])
        coping_strategy = self._get_coping_strategy(emotion)
        
        # Combine components into final response
        response = f"{validation} {question} {coping_strategy}"
        
        return response
    
    def _generate_crisis_response(self, crisis_indicators: List[str]) -> str:
        """Generate a response for crisis situations."""
        # Get the most severe crisis indicator
        priority_order = ['self_harm', 'violence', 'emergency']
        selected_indicator = next(
            (indicator for indicator in priority_order if indicator in crisis_indicators),
            crisis_indicators[0]
        )
        
        # Get crisis response and add support resources
        response = self.crisis_responses[selected_indicator]
        
        # Add relevant helpline information
        if selected_indicator == 'self_harm':
            helpline = self.support_resources['crisis_lines']['national_suicide_prevention']
            response += f"\n\nNational Suicide Prevention Lifeline: {helpline}"
        elif selected_indicator == 'violence':
            helpline = self.support_resources['crisis_lines']['domestic_violence']
            response += f"\n\nDomestic Violence Hotline: {helpline}"
        
        # Add general crisis text line
        response += f"\nCrisis Text Line: {self.support_resources['crisis_lines']['crisis_text_line']}"
        
        return response
    
    def _get_coping_strategy(self, emotion: str) -> str:
        """Get an appropriate coping strategy based on emotional state."""
        strategies = self.templates['coping_strategies']
        
        if emotion in ['sad', 'fear', 'disgust']:
            return random.choice(strategies['stress_relief'])
        elif emotion in ['angry', 'surprise']:
            return random.choice(strategies['general'])
        elif emotion == 'happy':
            return random.choice(strategies['mood_improvement'])
        else:  # neutral or unknown
            return random.choice(strategies['general'])
    
    def format_with_context(self, user_input: str, emotion: str,
                           conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the response template with user input, emotion, and conversation history."""
        template = self.templates['base_template']
        
        # Add conversation history if available
        if conversation_history:
            context = self.templates['context_template'].format(
                history='\n'.join(
                    f"User: {turn['user_input']}\n"
                    f"Emotion: {turn['emotion']}\n"
                    f"Assistant: {turn['response']}"
                    for turn in conversation_history[-3:]
                )
            )
            template += context
        
        # Format template with current interaction
        formatted_template = template.format(
            user_input=user_input,
            emotion=emotion
        )
        
        return formatted_template