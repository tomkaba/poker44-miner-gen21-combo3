import torch
import torch.nn as nn


class Gen20HybridV1(nn.Module):
    def __init__(
        self,
        *,
        action_vocab_size: int = 16,
        street_vocab_size: int = 8,
        seat_vocab_size: int = 12,
        dense_feature_dim: int = 293,
        token_hidden_dim: int = 160,
        dense_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab_size, 24)
        self.street_emb = nn.Embedding(street_vocab_size, 10)
        self.seat_emb = nn.Embedding(seat_vocab_size, 10)

        self.action_mlp = nn.Sequential(
            nn.Linear(24 + 10 + 10 + 12, token_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, token_hidden_dim),
            nn.ReLU(),
        )

        self.dense_norm = nn.LayerNorm(dense_feature_dim)
        self.dense_mlp = nn.Sequential(
            nn.Linear(dense_feature_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim, dense_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = token_hidden_dim * 3 + dense_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        action_type: torch.Tensor,
        street: torch.Tensor,
        actor_seat: torch.Tensor,
        amount: torch.Tensor,
        raise_to: torch.Tensor,
        call_to: torch.Tensor,
        norm_amount_bb: torch.Tensor,
        pot_before: torch.Tensor,
        pot_after: torch.Tensor,
        raise_to_missing: torch.Tensor,
        call_to_missing: torch.Tensor,
        valid_mask: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        act_e = self.action_emb(action_type)
        str_e = self.street_emb(street)
        seat_e = self.seat_emb(actor_seat)

        num_feats = torch.stack(
            [
                amount,
                raise_to,
                call_to,
                norm_amount_bb,
                pot_before,
                pot_after,
                raise_to_missing,
                call_to_missing,
                valid_mask,
                amount * valid_mask,
                pot_before * valid_mask,
                pot_after * valid_mask,
            ],
            dim=-1,
        )

        token_x = torch.cat([act_e, str_e, seat_e, num_feats], dim=-1)
        token_x = self.action_mlp(token_x)

        valid_mask_expanded = valid_mask.unsqueeze(-1)
        token_x = token_x * valid_mask_expanded

        hand_den = valid_mask_expanded.sum(dim=2).clamp_min(1.0)
        hand_repr = token_x.sum(dim=2) / hand_den

        hand_mask = (valid_mask.sum(dim=2) > 0).float().unsqueeze(-1)
        chunk_den = hand_mask.sum(dim=1).clamp_min(1.0)
        chunk_avg = (hand_repr * hand_mask).sum(dim=1) / chunk_den
        chunk_max = (hand_repr * hand_mask + (1.0 - hand_mask) * -1e9).max(dim=1).values
        chunk_min = (hand_repr * hand_mask + (1.0 - hand_mask) * 1e9).min(dim=1).values

        dense_repr = self.dense_mlp(self.dense_norm(dense_features))
        fused = torch.cat([chunk_avg, chunk_max, chunk_min, dense_repr], dim=-1)
        return self.head(fused).squeeze(-1)