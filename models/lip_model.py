import torch
import torch.nn as nn
import torch.nn.functional as F

VOCABULARY = [
    "RESTING",       # 0: Neutral mouth
    "HELP",          # 1: Bilabial burst /h/ -> open /ɛ/ -> lateral /l/ -> bilabial close /p/
    "WATER",         # 2: Rounded protrusion /w/ -> open /ɔ/ -> flap /t/ -> neutral /ɚ/
    "YES",           # 3: Palatal glide /j/ -> open-mid /ɛ/ -> alveolar fricative /s/
    "NO",            # 4: Alveolar /n/ -> rounded diphthong /oʊ/
    "DOCTOR",        # 5: Alveolar /d/ -> open /ɑ/ -> velar /k/ -> neutral /ɚ/
    "PAIN",          # 6: Bilabial close /p/ -> spread diphthong /eɪ/ -> alveolar /n/
    "THANK YOU",     # 7: Dental /θ/ -> open /æ/ -> velar /ŋ/ + glide /j/ -> rounded /u/
    "TIRED",         # 8: Alveolar /t/ -> open diphthong /aɪ/ -> alveolar /d/
    "NURSE",         # 9: Alveolar /n/ -> retroflex /ɚ/ -> alveolar fricative /s/
    "BREATHE",       # 10: Bilabial /b/ -> retroflex /r/ -> high front /i/ -> dental /ð/
    "MEDICINE"       # 11: Bilabial /m/ -> mid front /ɛ/ -> alveolar /d/ -> high /ɪ/ -> nasal /n/
]

VOCAB_MAP = {word: idx for idx, word in enumerate(VOCABULARY)}


class TemporalAttention(nn.Module):
    """
    Self-Attention layer to dynamically weigh key kinematic articulatory milestones
    (e.g., initial bilabial contact, peak vowel opening, final closure).
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # Temporal attention distribution
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class VisemeTemporalNet(nn.Module):
    """
    Spatio-Temporal Bi-LSTM with 1D-Convolution and Temporal Self-Attention
    for Visual Speech Recognition (Lip-Reading) from 12D geometric kinematic vectors.
    """
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_classes: int = len(VOCABULARY)):
        super(VisemeTemporalNet, self).__init__()
        
        # 1D Temporal Convolutional Feature Extractor
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(0.15)
        
        # Bidirectional Long Short-Term Memory (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        
        # Self-Attention across the 30-frame temporal window
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Classification Head
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len=30, input_dim=12)
        batch_size, seq_len, _ = x.size()
        
        # Reshape for Conv1D: (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = self.dropout_conv(self.relu(self.bn1(self.conv1(x_conv))))
        
        # Reshape back for LSTM: (batch_size, seq_len, channels=32)
        x_lstm = x_conv.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Temporal Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (batch_size, 128)
        
        # Dense layers
        out = self.dropout_fc(self.relu(self.bn_fc(self.fc1(context))))
        logits = self.fc2(out)
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities, attn_weights
