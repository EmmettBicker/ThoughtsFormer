import torch
import torch.nn as nn

class ThoughtsFormer(nn.Module):
    def __init__(self, vocab_size, num_classes, max_sequence_length=1024, d_model=384, num_heads=6, num_steps=6, dropout=0.1, is_causal=True):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_sequence_length, d_model))
        self.time_embedding = nn.Parameter(torch.randn(num_steps, d_model))

        # Single transformer encoder layer that we'll reuse
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )

        # Output layer for classification
        self.output =  nn.Linear(d_model, num_classes)
        
        self.is_causal = True

    def generate_thoughtsformer_mask(self, sz: int, thoughts_taken: int) -> torch.Tensor:
        t = thoughts_taken + 1
        def create_vertical_lines(size, spacing):
            indices = torch.arange(size)
            pattern = (indices % spacing != 0).float()
            lines = pattern.expand(size, -1)
            return lines

        lines = create_vertical_lines(sz, t).bool()
        blocks = ~torch.block_diag(*torch.ones(sz//t+1, t, t)).bool()[0:sz, 0:sz]
        line_blocks = torch.bitwise_and(lines, blocks)
        mask = line_blocks

        return mask

    def forward(self, x, attention_mask=None):
        B, L = x.shape

        # Handle padding mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None

        # Initial embedding
        x = self.embedding(x)
        x = x + self.pos_embedding[:L].unsqueeze(0)
        E = x.size(2)

        for step in range(self.num_steps):
            causal_mask = self.generate_thoughtsformer_mask(L*(step+1), step).to(x.device) if self.is_causal else None
            # Add time embedding to embeddings that were just generated
            x[:, step::(step+1)] = x[:, step::(step+1)] + self.time_embedding[step].unsqueeze(0).unsqueeze(1)

            next = self.transformer_layer(
                x,
                src_mask=causal_mask,
                # Padding mask is repeated for every thought
                src_key_padding_mask=torch.repeat_interleave(padding_mask, repeats=step+1, dim=-1) if padding_mask is not None else None
            )
            
            # Next sequence
            _x = torch.zeros(B, L * (step+2), E).to(x.device)
            for s in range(step+1):
                _x[:,s::(step+2)] = x[:, s::(step+1)]
            # Add the embeddings that were just generated
            _x[:,step+1::(step+2)] = next[:, step::(step+1)]
            x = _x
        # Return embeddings that were generated at the final step 
        return self.output(x[:,step+1::(step+2)])