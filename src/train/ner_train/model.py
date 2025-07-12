import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from src.train.ner_train import config
import logging # Import logging

logger = config.get_logger(__name__) # Get logger instance

class CharCNN(nn.Module):
    """Character-level CNN to capture morphological features."""
    def __init__(self, char_vocab_size, embedding_dim, cnn_filters, kernel_sizes, max_word_len):
        super().__init__()
        self.max_word_len = max_word_len
        # Character embedding layer
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=config.char_vocab.get('<PAD>', 0))
        # List of 2D convolutional layers, one for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, # Input channels (treat char embeddings as grayscale image)
                out_channels=cnn_filters, # Number of filters to learn
                kernel_size=(k, embedding_dim) # Kernel slides along word length, covers full embedding dim
            ) for k in kernel_sizes
        ])
        # The output dimension is the total number of filters across all kernel sizes
        self.output_dim = cnn_filters * len(kernel_sizes)

    def forward(self, char_ids):
        # char_ids: (batch_size, seq_len, max_word_len)
        batch_size, seq_len, max_word_len = char_ids.size()

        # Embed characters: (B, S, W) -> (B, S, W, E_char)
        embedded_chars = self.embedding(char_ids)

        # Prepare for Conv2d: Reshape and add channel dimension
        # (B, S, W, E_char) -> (B*S, W, E_char) -> (B*S, 1, W, E_char)
        embedded_chars = embedded_chars.view(-1, self.max_word_len, config.CHAR_EMBEDDING_DIM)
        embedded_chars = embedded_chars.unsqueeze(1)

        # Apply convolutions, activation, and pooling for each kernel size
        # conv output: (B*S, C_filters, W-k+1, 1) -> Apply ReLU -> squeeze(3) -> (B*S, C_filters, W-k+1)
        conv_outputs = [F.relu(conv(embedded_chars)).squeeze(3) for conv in self.convs]
        # Max-pooling over the resulting length dimension: (B*S, C_filters, W-k+1) -> pool -> (B*S, C_filters, 1) -> squeeze(2) -> (B*S, C_filters)
        pooled_outputs = [F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]

        # Concatenate the outputs from different kernel sizes along the filter dimension
        # List of [(B*S, C_filters)] -> (B*S, C_filters * num_kernels) = (B*S, C_out)
        char_cnn_output = torch.cat(pooled_outputs, dim=1)

        # Reshape back to sequence format: (B*S, C_out) -> (B, S, C_out)
        char_cnn_output = char_cnn_output.view(batch_size, seq_len, self.output_dim)

        return char_cnn_output


class NERModel(nn.Module):
    """
    Main NER model combining contextual embeddings, POS/DEP tags, CharCNN, BiLSTM, and CRF.
    """
    def __init__(self, num_ner_tags, pos_vocab_size, dep_vocab_size, char_vocab_size):
        super().__init__()

        # --- Embedding Layers ---
        # Contextual embeddings (e.g., from spaCy/BERT) are passed as input, not defined here.
        # POS tag embeddings
        self.pos_embedding = nn.Embedding(pos_vocab_size, config.POS_EMBEDDING_DIM, padding_idx=config.pos_vocab.get('<PAD>', 0))
        # Dependency relation embeddings
        self.dep_embedding = nn.Embedding(dep_vocab_size, config.DEP_EMBEDDING_DIM, padding_idx=config.dep_vocab.get('<PAD>', 0))
        # Character-level CNN
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            embedding_dim=config.CHAR_EMBEDDING_DIM,
            cnn_filters=config.CHAR_CNN_FILTERS,
            kernel_sizes=config.CHAR_CNN_KERNELS,
            max_word_len=config.MAX_WORD_LEN
        )

        # --- BiLSTM Layer ---
        # Calculate input dimension for BiLSTM
        lstm_input_dim = (
            config.SPACY_EMBEDDING_DIM +
            config.POS_EMBEDDING_DIM +
            config.DEP_EMBEDDING_DIM +
            self.char_cnn.output_dim
        )
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            bidirectional=True,
            batch_first=True, # Input/output format: (batch, seq, feature)
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0 # Dropout between LSTM layers only
        )

        # --- Output Layers ---
        # Dropout layer applied after BiLSTM
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        # Linear layer to project BiLSTM output to NER tag space (emission scores)
        # Input dimension is BiLSTM hidden dim * 2 (due to bidirectional)
        self.hidden2tag = nn.Linear(config.LSTM_HIDDEN_DIM * 2, num_ner_tags)
        # Conditional Random Field (CRF) layer for sequence decoding
        self.crf = CRF(num_tags=num_ner_tags, batch_first=True)

        # --- Transformer Fine-tuning (Placeholder Info) ---
        # The actual logic for unfreezing layers happens in the optimizer setup (train.py)
        self.unfreeze_transformer_layers = config.UNFREEZE_TRANSFORMER_LAYERS


    def forward(self, embeddings, pos_ids, dep_ids, char_ids, attention_mask, tags=None):
        """
        Forward pass of the NER model.

        Args:
            embeddings (torch.Tensor): Precomputed contextual embeddings (B, S, E_spacy).
            pos_ids (torch.Tensor): POS tag IDs (B, S).
            dep_ids (torch.Tensor): Dependency tag IDs (B, S).
            char_ids (torch.Tensor): Character IDs (B, S, W).
            attention_mask (torch.Tensor): Boolean mask (True for real tokens) (B, S).
            tags (torch.Tensor, optional): True NER tag IDs for loss calculation (B, S). Defaults to None.

        Returns:
            torch.Tensor or list: If tags is provided, returns the negative log-likelihood loss (scalar).
                                  Otherwise, returns a list of predicted tag ID sequences.
        """
        # 1. Get Embeddings for POS, DEP, Chars
        pos_emb = self.pos_embedding(pos_ids)       # (B, S, E_pos)
        dep_emb = self.dep_embedding(dep_ids)       # (B, S, E_dep)
        char_cnn_out = self.char_cnn(char_ids)      # (B, S, C_out)

        # 2. Concatenate all embeddings
        # (B, S, E_spacy + E_pos + E_dep + C_out)
        combined_embeddings = torch.cat([embeddings, pos_emb, dep_emb, char_cnn_out], dim=2)
        combined_embeddings = self.dropout(combined_embeddings) # Apply dropout

        # 3. Pass through BiLSTM
        # Input: (B, S, combined_dim)
        # Output: (B, S, H_lstm * 2)
        # We don't use packing/unpacking here, relying on the CRF mask instead.
        lstm_out, _ = self.bilstm(combined_embeddings)

        # 4. Apply dropout to BiLSTM output
        # Input: (B, S, H_lstm * 2)
        # Output: (B, S, H_lstm * 2)
        features = self.dropout(lstm_out)

        # 5. Project features to tag space (get emission scores)
        # Input: (B, S, H_lstm * 2)
        # Output: (B, S, N_tags)
        emissions = self.hidden2tag(features)

        # 6. Use CRF for loss calculation or decoding
        # Asegurarse de que la máscara sea booleana explícitamente
        bool_mask = attention_mask.bool()
        
        if tags is not None:
            # Training: Calculate negative log-likelihood loss
            # Input emissions: (B, S, N_tags)
            # Input tags: (B, S)
            # Input mask: (B, S) - Boolean mask
            loss = -self.crf(emissions, tags, mask=bool_mask, reduction='mean')
            return loss
        else:
            # Inference: Decode the most likely tag sequence using Viterbi algorithm
            # Input emissions: (B, S, N_tags)
            # Input mask: (B, S) - Boolean mask
            # Output: List (length B) of lists, each inner list contains predicted tag IDs for a sequence
            decoded_tags = self.crf.decode(emissions, mask=bool_mask)
            return decoded_tags
