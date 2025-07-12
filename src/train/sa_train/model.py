import torch
import torch.nn as nn

class SentimentFromTextNerLSTM(nn.Module):
    """
    LSTM model that predicts sentiment from text tokens and NER tags.
    Takes text tokens and NER tags as input, and predicts a single sentiment class.
    """
    
    def __init__(self, text_vocab_size, text_embedding_dim,
                 ner_vocab_size, ner_embedding_dim,
                 hidden_dim, sentiment_vocab_size,
                 n_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()

        # Input layers
        self.text_embedding = nn.Embedding(
            text_vocab_size, text_embedding_dim, padding_idx=0
        )
        self.ner_embedding = nn.Embedding(
            ner_vocab_size, ner_embedding_dim, padding_idx=0
        )
        
        # LSTM layer
        self.lstm_input_dim = text_embedding_dim + ner_embedding_dim
        self.lstm = nn.LSTM(
            self.lstm_input_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(self.fc_input_dim, sentiment_vocab_size)

    def forward(self, text, ner_tags):
        """
        Forward pass:
        1. Embed text and NER tags
        2. Concatenate embeddings
        3. Pass through LSTM
        4. Extract final hidden state
        5. Predict sentiment class
        
        Args:
            text: [batch_size, seq_len] - Text token indices
            ner_tags: [batch_size, seq_len] - NER tag indices
            
        Returns:
            output: [batch_size, sentiment_vocab_size] - Scores for each sentiment class
        """
        # Embedding
        embedded_text = self.dropout(self.text_embedding(text))
        embedded_ner = self.dropout(self.ner_embedding(ner_tags))
        
        # Concatenate
        combined_embeddings = torch.cat((embedded_text, embedded_ner), dim=2)
        
        # LSTM
        _, (hidden, _) = self.lstm(combined_embeddings)
        
        # Get final hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden_final = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden_final = self.dropout(hidden[-1,:,:])
        
        # Output
        output = self.fc(hidden_final)
        
        return output
