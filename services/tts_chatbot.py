# tts_chatbot.py
import os
import sys
from main import MentalHealthChatbot  # Import the original chatbot class from main.py
from services.tts_engine import TTSEngine  # Import the TTS engine with male voice

class TTSMentalHealthChatbot:
    """A wrapper class that adds Text-to-Speech to the MentalHealthChatbot."""
    
    def __init__(self, model_path: str, class_labels_path: str):
        """
        Initialize the TTS-enabled chatbot.
        
        Args:
            model_path (str): Path to the emotion detection model.
            class_labels_path (str): Path to the class labels JSON file.
        """
        # Initialize the original MentalHealthChatbot
        self.chatbot = MentalHealthChatbot(model_path, class_labels_path)
        # Initialize the TTS engine with male voice as specified
        self.tts_engine = TTSEngine(rate=150, volume=1.0)
        self.tts_enabled = True  # Flag to toggle TTS on/off
    
    def run_with_tts(self):
        """Run the chatbot with TTS enabled, wrapping the original run method."""
        print("Starting Mental Health Chatbot with Text-to-Speech...")
        print("Type 'tts off' to disable speech, 'tts on' to enable it.")
        
        # Monkey-patch the process_user_input method to add TTS
        original_process_user_input = self.chatbot.process_user_input
        
        def tts_process_user_input(user_input, visual_emotion_data=None):
            # Handle TTS toggle commands
            if user_input.lower() == 'tts off':
                self.tts_enabled = False
                return "Text-to-Speech disabled."
            elif user_input.lower() == 'tts on':
                self.tts_enabled = True
                return "Text-to-Speech enabled."
            
            # Process the input as usual
            response = original_process_user_input(user_input, visual_emotion_data)
            # Speak the response if TTS is enabled
            if self.tts_enabled:
                self.tts_engine.speak(response)
            return response
        
        # Apply the patched method
        self.chatbot.process_user_input = tts_process_user_input
        
        # Run the original chatbot with the patched method
        try:
            self.chatbot.run()
        finally:
            self.tts_engine.stop()  # Ensure TTS stops cleanly on exit

def main():
    try:
        # Define model and label paths (same as main.py)
        model_path = 'models/best_model.pth'
        class_labels_path = 'models/class_labels.json'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
        
        if not os.path.exists(class_labels_path):
            raise FileNotFoundError(f"Label file '{class_labels_path}' does not exist.")
        
        # Initialize and run the TTS-enabled chatbot
        tts_chatbot = TTSMentalHealthChatbot(
            model_path=model_path,
            class_labels_path=class_labels_path
        )
        tts_chatbot.run_with_tts()
    
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please ensure all required model files are correctly placed in the models directory.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("If the problem persists, check model files or contact support.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nThank you for using the TTS-enabled Mental Health Chatbot!")

if __name__ == "__main__":
    main()