from typing import Dict, List, Optional
import time


class AgenticBionicSynthesizer:
    """
    SILENTVOICE AI • DIRECT ARTICULATORY SPEECH SYNTHESIZER
    Delivers pure, direct silent-speech articulation with ZERO canned AI chatter.
    Spoken words are emitted directly as trained by the user without artificial noise or repetition.
    """

    def __init__(self):
        self.patient_name: Optional[str] = None
        self.last_word: Optional[str] = None
        self.last_word_time: float = 0.0
        self.had_intervening_rest: bool = True
        self.continuous_transcript: List[str] = []
        self.conversation_turn: int = 0

    def clear_transcript(self):
        self.continuous_transcript = []
        self.last_word = None
        self.had_intervening_rest = True

    def process_decoded_word(self, word: str, confidence: float) -> Dict:
        now = time.time()

        # If resting baseline is observed, record that the speaker paused
        if word == "RESTING":
            self.had_intervening_rest = True
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": [],
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        # Filter low confidence predictions (< 0.60) to reject transitional artifacts
        if confidence < 0.60:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": [],
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        time_since_last = now - self.last_word_time

        # Global Utterance Debounce:
        # Prevent 1 word from triggering 2-3 words. An articulation requires at least 1.8s
        # unless an explicit pause/rest was observed, in which case 1.3s is sufficient.
        min_cooldown = 1.3 if self.had_intervening_rest else 1.8
        if time_since_last < min_cooldown:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": [],
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        # Prevent duplicate identical word within 2.5 seconds
        if word == self.last_word and time_since_last < 2.5:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": [],
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        # Clean speech: Spoken word is exactly the word or phrase the user trained
        clean_word = word.strip().upper()

        # Prevent duplicate consecutive words in the sentence stream
        if not self.continuous_transcript or self.continuous_transcript[-1] != clean_word:
            self.continuous_transcript.append(clean_word)
            if len(self.continuous_transcript) > 15:
                self.continuous_transcript.pop(0)

        self.last_word = clean_word
        self.last_word_time = now
        self.had_intervening_rest = False
        self.conversation_turn += 1

        return {
            "raw_word": clean_word,
            "confidence": round(confidence, 3),
            "is_active_speech": True,
            "synthesized_sentence": clean_word,
            "ai_dialogue_reply": "",
            "compound_detected": False,
            "suggested_next_words": [],
            "continuous_sentence": " ".join(self.continuous_transcript),
            "patient_name": "User",
            "conversation_turn": self.conversation_turn
        }


# Global synthesizer instance
agent_synthesizer = AgenticBionicSynthesizer()
