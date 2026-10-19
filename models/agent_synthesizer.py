from typing import Dict, List, Optional
import time


class AgenticBionicSynthesizer:
    """
    SILENTVOICE AI • AGENTIC TWO-WAY CONVERSATIONAL SYNTHESIZER
    Transforms decoded discrete lip movements into fluent patient speech
    and generates real-time empathetic AI Care Partner dialogue responses.
    """

    NATURAL_EXPANSIONS = {
        "PRINCE KUMAR": "My name is Prince Kumar.",
        "HELP": "I need immediate assistance, please help me.",
        "WATER": "Could I please have a glass of water?",
        "YES": "Yes, that is correct.",
        "NO": "No, I do not want that.",
        "DOCTOR": "Please call the doctor for an evaluation.",
        "PAIN": "I am experiencing physical pain right now.",
        "THANK YOU": "Thank you very much for your help and care.",
        "TIRED": "I am feeling very exhausted and need to rest.",
        "NURSE": "Please call the nurse to my bedside.",
        "BREATHE": "I am having difficulty breathing comfortably.",
        "MEDICINE": "It is time for my prescribed medication.",
        "I WANT WATER": "I am very thirsty, I want some water.",
        "RESTING": ""
    }

    AI_CONVERSATIONAL_REPLIES = {
        "PRINCE KUMAR": "Hello Prince Kumar! I am with you and listening closely. How can I assist you right now?",
        "HELP": "Assistance is on the way! I have alerted the medical staff to your room immediately. Please stay calm.",
        "WATER": "Right away. Bringing you a fresh glass of water immediately.",
        "YES": "Understood and confirmed.",
        "NO": "Understood, canceling that request.",
        "DOCTOR": "Paging the on-duty physician to your room right now.",
        "PAIN": "I am so sorry you are in pain. I have flagged this as high priority for the medical team.",
        "THANK YOU": "You are very welcome! I am here whenever you need anything.",
        "TIRED": "Rest comfortably. I will dim the screen and keep watch over your telemetry.",
        "NURSE": "Calling the nurse station to your bedside immediately.",
        "BREATHE": "Please take slow, steady breaths. I have signaled emergency respiratory assistance.",
        "MEDICINE": "Checking your medication schedule now and dispatching the nursing assistant.",
        "I WANT WATER": "Bringing you fresh cold water right now.",
        "RESTING": ""
    }

    SUGGESTED_NEXT_WORDS = {
        "PRINCE KUMAR": ["HELP", "WATER", "DOCTOR", "THANK YOU"],
        "HELP": ["PAIN", "DOCTOR", "WATER", "NURSE"],
        "WATER": ["THANK YOU", "HELP", "YES", "DOCTOR"],
        "YES": ["THANK YOU", "WATER", "HELP"],
        "NO": ["WATER", "HELP", "TIRED"],
        "DOCTOR": ["PAIN", "BREATHE", "MEDICINE", "THANK YOU"],
        "PAIN": ["DOCTOR", "MEDICINE", "HELP", "TIRED"],
        "THANK YOU": ["RESTING", "WATER", "HELP"],
        "TIRED": ["RESTING", "WATER", "BREATHE"],
        "NURSE": ["MEDICINE", "WATER", "HELP", "PAIN"],
        "BREATHE": ["DOCTOR", "HELP", "TIRED", "NURSE"],
        "MEDICINE": ["WATER", "THANK YOU", "YES", "DOCTOR"],
        "I WANT WATER": ["THANK YOU", "HELP", "YES"]
    }

    COMPOUND_GRAMMAR = {
        ("PRINCE KUMAR", "HELP"): "Prince Kumar here. I need immediate assistance, please help me.",
        ("PRINCE KUMAR", "WATER"): "Prince Kumar here. Could I please have a glass of water?",
        ("PRINCE KUMAR", "DOCTOR"): "Prince Kumar here. Please have a doctor come examine me.",
        ("PRINCE KUMAR", "PAIN"): "Prince Kumar here. I am experiencing severe pain.",
        ("HELP", "PAIN"): "I am in severe pain, please help me immediately.",
        ("PAIN", "DOCTOR"): "I am experiencing pain, please have a doctor examine me.",
        ("WATER", "HELP"): "I am thirsty, please help me drink some water.",
        ("TIRED", "BREATHE"): "I am feeling fatigued and short of breath.",
        ("NURSE", "MEDICINE"): "Please alert the nurse that I need my medicine."
    }

    def __init__(self):
        self.patient_name: Optional[str] = None
        self.recent_history: List[str] = []
        self.last_word: Optional[str] = None
        self.last_word_time: float = 0.0
        self.continuous_transcript: List[str] = []
        self.conversation_turn: int = 0

    def process_decoded_word(self, word: str, confidence: float) -> Dict:
        """
        Processes a newly recognized silent word, applies temporal hysteresis,
        contextual grammar chaining, and generates conversational partner replies.
        """
        now = time.time()
        
        # Filter resting / low confidence
        if word == "RESTING" or confidence < 0.60:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": self.SUGGESTED_NEXT_WORDS.get(self.last_word or "HELP", ["HELP", "WATER", "DOCTOR", "THANK YOU"]),
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        # Temporal debounce: Prevent immediate double-triggering within 0.8s for the same word
        if word == self.last_word and (now - self.last_word_time) < 0.8:
            return {
                "raw_word": word,
                "confidence": round(confidence, 3),
                "is_active_speech": False,
                "synthesized_sentence": "",
                "ai_dialogue_reply": "",
                "compound_detected": False,
                "suggested_next_words": self.SUGGESTED_NEXT_WORDS.get(word, ["HELP", "WATER", "DOCTOR"]),
                "continuous_sentence": " ".join(self.continuous_transcript)
            }

        # Check for Patient Identity
        if "PRINCE" in word.upper() or "KUMAR" in word.upper():
            self.patient_name = "Prince Kumar"

        # Check for multi-word temporal chaining (within 6 seconds)
        compound_sentence = None
        if len(self.recent_history) > 0 and (now - self.last_word_time) < 6.0:
            prev_word = self.recent_history[-1]
            pair = (prev_word, word)
            rev_pair = (word, prev_word)
            
            if pair in self.COMPOUND_GRAMMAR:
                compound_sentence = self.COMPOUND_GRAMMAR[pair]
            elif rev_pair in self.COMPOUND_GRAMMAR:
                compound_sentence = self.COMPOUND_GRAMMAR[rev_pair]

        self.recent_history.append(word)
        if len(self.recent_history) > 12:
            self.recent_history.pop(0)

        self.last_word = word
        self.last_word_time = now
        self.conversation_turn += 1

        # Synthesized Patient speech
        if compound_sentence:
            final_sentence = compound_sentence
        else:
            final_sentence = self.NATURAL_EXPANSIONS.get(word, word)

        # Update Continuous Transcript
        # If quiet for > 10 seconds, reset transcript to start fresh sentence
        if len(self.continuous_transcript) > 0 and (now - self.last_word_time) > 10.0:
            self.continuous_transcript = []
        self.continuous_transcript.append(final_sentence)
        if len(self.continuous_transcript) > 6:
            self.continuous_transcript.pop(0)

        # Generate Contextual AI Care Partner Dialogue Reply
        ai_reply = self.AI_CONVERSATIONAL_REPLIES.get(word, "")
        if not ai_reply:
            if self.patient_name:
                ai_reply = f"Understood, {self.patient_name}. I have registered your request for '{word}'."
            else:
                ai_reply = f"Understood. I have registered your request for '{word}'."

        # Personalize reply if patient is identified
        if self.patient_name and "{name}" in ai_reply:
            ai_reply = ai_reply.replace("{name}", self.patient_name)
        elif self.patient_name and "Prince" not in ai_reply and word not in ["PRINCE KUMAR", "RESTING"]:
            ai_reply = ai_reply.replace("Understood", f"Understood, {self.patient_name}")

        suggested = self.SUGGESTED_NEXT_WORDS.get(word, ["HELP", "WATER", "DOCTOR", "THANK YOU"])

        return {
            "raw_word": word,
            "confidence": round(confidence, 3),
            "is_active_speech": True,
            "synthesized_sentence": final_sentence,
            "ai_dialogue_reply": ai_reply,
            "compound_detected": compound_sentence is not None,
            "suggested_next_words": suggested,
            "continuous_sentence": " ".join(self.continuous_transcript),
            "patient_name": self.patient_name or "Patient",
            "conversation_turn": self.conversation_turn
        }


# Global synthesizer instance
agent_synthesizer = AgenticBionicSynthesizer()
