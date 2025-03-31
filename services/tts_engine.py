# services/tts_engine.py
import pyttsx3
from typing import Optional

class TTSEngine:
    """Text-to-Speech engine using pyttsx3 to read text aloud."""
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize the TTS engine with a male voice.
        
        Args:
            rate (int): Speech rate (words per minute). Default is 150.
            volume (float): Volume level (0.0 to 1.0). Default is 1.0.
        """
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        male_voice_index = 1  # Index for male voice (may vary based on system)
        self.engine.setProperty('voice', voices[male_voice_index].id)
        self.engine.setProperty('rate', rate)  # Adjust speech rate
        self.engine.setProperty('volume', volume)  # Adjust volume

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it aloud.
        
        Args:
            text (str): The text to be spoken.
        """
        self.engine.say(text)
        self.engine.runAndWait()
    
    def stop(self) -> None:
        """Stop the speech if currently speaking."""
        self.engine.stop()
    
    def set_rate(self, rate: int) -> None:
        """
        Adjust the speech rate.
        
        Args:
            rate (int): New speech rate (words per minute).
        """
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float) -> None:
        """
        Adjust the volume level.
        
        Args:
            volume (float): New volume level (0.0 to 1.0).
        """
        self.engine.setProperty('volume', volume)

if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello, this is a test of the text-to-speech engine with a male voice!")