# NYCU IEE Deep Learning Lab 03: Machine Translation
# BSChen (313510156)
import torch
import torch.nn as nn
import math
from utils import *

PAD_IDX  = 0
BOS_IDX  = 101
EOS_IDX  = 102
MASK_IDX = 103

### ==============================================================================
### (1) Model Definition
### ==============================================================================

class Attention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  torch.Tensor = None
    ) -> torch.Tensor:
        """ Scaled Dot-Product Attention
        Args:
            `query`: (batch_size, n_head, seq_len, key_dim)
            `key`  : (batch_size, n_head, seq_len, key_dim)
            `value`: (batch_size, n_head, seq_len, key_dim)
            `mask` : (batch_size, 1, seq_len, seq_len)
        """
        # Initialization
        d_k = query.size(-1)

        # Compute attention score
        weights = query @ key.transpose(-2, -1) / math.sqrt(d_k)  # (batch_size, n_head, seq_len, seq_len)
        weights = weights.masked_fill(mask == 0, float('-inf')) if mask is not None else weights
        weights = torch.softmax(weights, dim=-1)

        # Retrieve values
        return weights @ value  # (batch_size, n_head, seq_len, key_dim)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # Parameters
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # Layers
        self.attention = Attention()
        self.weight_q  = nn.Linear(d_model, d_model)
        self.weight_k  = nn.Linear(d_model, d_model)
        self.weight_v  = nn.Linear(d_model, d_model)
        self.weight_o  = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  torch.Tensor
    ) -> torch.Tensor:
        """ Multi-Head Attention
        Args:
            `query`: (batch_size, seq_len, d_model)
            `key`  : (batch_size, seq_len, d_model)
            `value`: (batch_size, seq_len, d_model)
            `mask` : (batch_size, 1, seq_len, seq_len)
        """
        # Linear projections
        query = self.weight_q(query)
        key   = self.weight_k(key)
        value = self.weight_v(value)

        # Split into n_head heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch_size, n_head, seq_len, d_k)
        key   = key.view(batch_size,   -1, self.n_head, self.d_k).transpose(1, 2)  # (batch_size, n_head, seq_len, d_k)
        value = value.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch_size, n_head, seq_len, d_k)

        # Apply attention
        out = self.attention(query, key, value, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output linear layer
        return self.weight_o(out)


class TransformerEncoderLayer(nn.Module):
    """ Transformer Encoder Layer """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # Parameters
        self.d_model = d_model
        self.n_head  = n_head
        self.d_ff    = d_ff
        self.dropout = dropout

        # Layers
        self.self_attention = MultiHeadAttention(n_head, d_model)
        self.feed_forward   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.dropout_1   = nn.Dropout(dropout)
        self.dropout_2   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # Multi-head attention
        out = self.self_attention(x, x, x, mask)
        out = self.dropout_1(out)
        x = self.layernorm_1(x + out)  # Add & Norm

        # Feed-forward network
        out = self.feed_forward(x)
        out = self.dropout_2(out)
        x = self.layernorm_2(x + out)  # Add & Norm

        return x


class TransformerDecoderLayer(nn.Module):
    """ Transformer Decoder Layer """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # Parameters
        self.d_model = d_model
        self.n_head  = n_head
        self.d_ff    = d_ff
        self.dropout = dropout

        # Layers
        self.self_attention  = MultiHeadAttention(n_head, d_model)
        self.cross_attention = MultiHeadAttention(n_head, d_model)
        self.feed_forward    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.layernorm_3 = nn.LayerNorm(d_model)
        self.dropout_1   = nn.Dropout(dropout)
        self.dropout_2   = nn.Dropout(dropout)
        self.dropout_3   = nn.Dropout(dropout)

    def forward(
        self, x:  torch.Tensor,
        enc_out:  torch.Tensor,
        src_mask: torch.Tensor=None,
        tgt_mask: torch.Tensor=None
    ) -> torch.Tensor:
        # Masked multi-head self-attention
        out = self.self_attention(x, x, x, tgt_mask)
        out = self.dropout_1(out)
        x = self.layernorm_1(x + out)  # Add & Norm

        # Multi-head cross-attention
        out = self.cross_attention(x, enc_out, enc_out, src_mask)
        out = self.dropout_2(out)
        x = self.layernorm_2(x + out)  # Add & Norm

        # Feed-forward network
        out = self.feed_forward(x)
        out = self.dropout_3(out)
        x = self.layernorm_3(x + out)  # Add & Norm

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout

        # Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """ Transformer Forward Pass
        Args:
            `src`: (batch_size, src_len, d_model)
            `tgt`: (batch_size, tgt_len, d_model)
            `src_mask`: (batch_size, 1, src_len, src_len)
            `tgt_mask`: (batch_size, 1, tgt_len, tgt_len)
        """
        # Encoder layers
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # Decoder layers
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return dec_out


class PositionalEncoding(nn.Module):
    """ Sinusoidal Positional Encoding """
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
        learnable: bool = False
    ):
        super().__init__()

        # Parameters
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable

        # Create positional encoding matrix
        if not learnable:
            freq_max = max_len / (2 * math.pi)
            position = torch.arange(0, max_len).unsqueeze(1)
            denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(freq_max) / d_model))
            pos_enc = torch.zeros(max_len, d_model)
            pos_enc[:, 0::2] = torch.sin(position * denom)
            pos_enc[:, 1::2] = torch.cos(position * denom)
            self.register_buffer('pos_enc', pos_enc)

        # Learnable positional encoding
        else:
            self.pos_enc = nn.Embedding(max_len, d_model)
            self.pos_ids = torch.arange(0, max_len).unsqueeze(0)

        # Embedding dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Add positional encoding to input tensor
        Args:
            `x`: (batch_size, seq_len, d_model)
        """
        # Add positional encoding to the input tensor
        seq_len = x.size(1)

        # Sinusoidal positional encoding
        if not self.learnable:
            return self.dropout(x + self.pos_enc[:seq_len, :])

        # Learnable positional encoding
        else:
            pos_ids = self.pos_ids[:, :seq_len].to(x.device)
            return self.dropout(x + self.pos_enc(pos_ids))


class TokenEmbedding(nn.Module):
    """ Token Embedding Layer """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        # Parameters
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        Args:
            `x`: (batch_size, seq_len)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class Seq2SeqNetwork(nn.Module):
    """ Seq2Seq Network with Transformer (Top-level module) """
    def __init__(
        self, 
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.transformer = Transformer(
            d_model=emb_size,
            num_heads=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, max_len=256, dropout=dropout, learnable=False
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """ Seq2SeqNetwork Forward Pass
        Args:
            `src`: (batch_size, src_len)
            `tgt`: (batch_size, tgt_len)
        """
        # Convert to long tensor
        src = src.long()
        tgt = tgt.long()

        # Embedding and positional encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src))  # (batch_size, seq_len, emb_size)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Causal & padding masking
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Transformer
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(outs)

    def make_src_mask(self, src: torch.Tensor, pad_idx: int=PAD_IDX) -> torch.Tensor:
        """ Padding mask for src sequences """
        batch_size, seq_len = src.size()
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt: torch.Tensor, pad_idx: int=PAD_IDX) -> torch.Tensor:
        """ Causal mask & Padding mask for tgt sequences """
        batch_size, seq_len = tgt.size()
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask = pad_mask & causal_mask  # (batch_size, 1, seq_len, seq_len)
        return mask.to(self.device)


### ==============================================================================
### (2) Inference Function
### ==============================================================================
def translate(
    model: torch.nn.Module,
    src_sentence: str,
    input_tokenizer: BertTokenizer,
    output_tokenizer: BertTokenizer,
    num_beams: int = 3,
    max_length: int = 200,
    min_length: int = 1
) -> str:
    """ Translate a sentence from chinese to english """
    # Initialize
    model.eval()
    sentence = input_tokenizer.encode(src_sentence)
    sentence = torch.tensor(sentence).view(1, -1)  # (1, seq_length)

    model = model.to(DEVICE)
    sentence = sentence.to(DEVICE)

    # Beam search
    max_tokens = max(sentence.shape[1], max_length)
    beams = [[torch.tensor([BOS_IDX], device=DEVICE).view(1, -1), 0]]  # (token list, score)
    for i in range(max_tokens):
        new_beams = []
        for tgt_tokens, acc_score in beams:
            # If EOS is already generated, keep the beam as is
            if tgt_tokens[0, -1].item() == EOS_IDX:
                new_beams.append([tgt_tokens, acc_score])
                continue

            # Inference
            with torch.no_grad():
                logits = model(sentence, tgt_tokens)[:, -1, :]         # (1, vocab_size)
            logits[:, EOS_IDX] = float('-inf') if i < min_length else logits[:, EOS_IDX]
            probs = torch.log_softmax(logits, dim=-1).squeeze(0)       # (vocab_size,)
            top_k_scores, top_k_tokens = torch.topk(probs, num_beams)  # (num_beams,)

            # Expand each beam
            for token, score in zip(top_k_tokens, top_k_scores):
                new_beams.append([
                    torch.cat([tgt_tokens, token.reshape(1, 1)], dim=1), acc_score + score.item()
                ])

        # Prune beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]

        # Early stopping
        if all(tgt_tokens[0, -1].item() == EOS_IDX for tgt_tokens, _ in beams):
            break

    # Return the best beam
    tgt_tokens, _ = beams[0]
    output_sentence = output_tokenizer.decode(tgt_tokens[0].tolist(), skip_special_tokens=True)
    return output_sentence


### ==============================================================================
### (3) Load model
### ==============================================================================
def load_model(MODEL_PATH = None):
    # Configurations
    EMB_SIZE = 256
    NHEAD = 4
    FFN_HID_DIM = 1024  # 2048
    NUM_ENCODER_LAYERS = 3  # 6
    NUM_DECODER_LAYERS = 2  # 6
    SRC_VOCAB_SIZE = tokenizer_chinese().vocab_size
    TGT_VOCAB_SIZE = tokenizer_english().vocab_size

    # Model initialization
    model = Seq2SeqNetwork(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
        device=DEVICE
    )

    # Load model weights
    if MODEL_PATH is not None: 
        model.load_state_dict(torch.load(MODEL_PATH))
    return model
