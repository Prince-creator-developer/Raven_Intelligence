from typing import Dict, List, Optional
import time


class AgenticBionicSynthesizer:
    """
    Agentic Context & Grammar Reconstruction Layer.
    Transforms raw decoded discrete lip-reading keywords into fluent,
    grammatically complete, and empathetic bionic speech sentences.
    """

    NATURAL_EXPANSIONS = {
        "HELP": "I need immediate assistance, please help me.",
        "WATER": "Could I please have a glass of water?",
        "YES": "Yes, that is correct.",
        "NO": "No, I do not want that.",
        "DOCTOR": "Please call the doctor for an evaluation.",
        "PAIN": "I am experiencing physical pain right now.",
        "THANK YOU": "Thank you very much for your help.",
        "TIRED": "I am feeling very exhausted and need to rest.",
        "NURSE": "Please call the nurse to my bedside.",
        "BREATHE": "I am having difficulty breathing comfortably.",
        "MEDICINE": "It is time for my prescribed medication.",
        "RESTING": ""
    }

    COMPOUND_GRAMMAR = {
        ("HELP", "PAIN"): "I am in severe pain, please help me immediately.",
        ("PAIN", "DOCTOR"): "I am experiencing pain, please have a doctor examine me.",
        ("WATER", "HELP"): "I am thirsty, please help me drink some water.",
        ("TIRED", "BREATHE"): "I am feeling fatigued and short of breath.",
        ("NURSE", "MEDICINE"): "Please alert the nurse that I need my medicine."
    }

    def __init__(self):
        self.recent_history: List[str] = []
        self.last_word_time = 0.0

    def process_decoded_word(self, word: str, confidence: float) -> Dict:
        """
        Processes a newly recognized silent word, applies temporal hysteresis
        and contextual grammar chaining.
        """
        now = time.time()
        
        # Filter resting / low confidence
        if word == "RESTING" or confidence < 0.60:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "compound_detected": False
            }

        # Check for temporal chaining (within 4 seconds of previous word)
        compound_sentence = None
        if len(self.recent_history) > 0 and (now - self.last_word_time) < 4.0:
            prev_word = self.recent_history[-1]
            pair = (prev_word, word)
            rev_pair = (word, prev_word)
            
            if pair in self.COMPOUND_GRAMMAR:
                compound_sentence = self.COMPOUND_GRAMMAR[pair]
            elif rev_pair in self.COMPOUND_GRAMMAR:
                compound_sentence = self.COMPOUND_GRAMMAR[rev_pair]

        self.recent_history.append(word)
        if len(self.recent_history) > 10:
            self.recent_history.pop(0)
        self.last_word_time = now

        final_sentence = compound_sentence or self.NATURAL_EXPANSIONS.get(word, word)

        return {
            "raw_word": word,
            "confidence": round(confidence, 3),
            "is_active_speech": True,
            "synthesized_sentence": final_sentence,
            "compound_detected": compound_sentence is not None,
            "context_history": list(self.recent_history[-4:])
        }


# Global synthesizer instance
agent_synthesizer = AgenticBionicSynthesizer()
