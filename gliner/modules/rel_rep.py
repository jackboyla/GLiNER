import torch
from torch import nn
from gliner.modules.span_rep import create_projection_layer, SpanMarkerV0, extract_elements
import torch.nn.functional as F

class EfficientRelationshipModel(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.4):
        super().__init__()
        self.entity_encoder = SpanMarkerV0(hidden_size, max_width=1, dropout=dropout)
        # Assuming the relationship_projector combines two entities' representations
        self.relationship_projector = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)  # Output dimension for relationship representation
        )

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor):
        entity_reps = self.entity_encoder(h, span_idx)  # Shape: [B, N, D]

        # Expand and repeat entity representations to create all pairs
        entity_reps_expanded = entity_reps.unsqueeze(2)  # Shape: [B, N, 1, D]
        entity_reps_tiled = entity_reps_expanded.repeat(1, 1, entity_reps.size(1), 1)  # Shape: [B, N, N, D]
        entity_reps_transposed = entity_reps.unsqueeze(1)  # Shape: [B, 1, N, D]
        entity_reps_tiled_transposed = entity_reps_transposed.repeat(1, entity_reps.size(1), 1, 1)  # Shape: [B, N, N, D]

        # Combine representations of all pairs
        combined_pairs = torch.cat([entity_reps_tiled, entity_reps_tiled_transposed], dim=-1)  # Shape: [B, N, N, 2*D]

        # Flatten combined pairs for batch processing
        combined_pairs_flat = combined_pairs.view(-1, combined_pairs.size(-1))  # Shape: [B*N*N, 2*D]

        # Project combined entity pairs to relationship space
        relationship_reps = self.relationship_projector(combined_pairs_flat)  # Shape: [B*N*N, D]
        relationship_reps = relationship_reps.view(entity_reps.size(0), entity_reps.size(1), entity_reps.size(1), -1)  # Shape: [B, N, N, D]

        return relationship_reps
    

class SpanMarkerV1(nn.Module):
    """
    Marks and projects span endpoints using an MLP.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        # span_idx.shape    ([B, num_possible_spans, 2])

        start_rep = self.project_start(h)  # ([B, L, D])
        end_rep = self.project_end(h)      # ([B, L, D])

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])  # ([B, num_possible_spans, D])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])      # ([B, num_possible_spans, D])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()   # ([B, num_possible_spans, D*2])

        return self.out_project(cat) # ([B, num_possible_spans, D])               #### .view(B, L, self.max_width, D)
    

class RelMarkerv0(nn.Module):
    """
    Efficiently marks and projects representations for all pairs of entities.
    """
    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.span_marker = SpanMarkerV1(hidden_size, max_width, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_indices: torch.Tensor) -> torch.Tensor:
        """
        h: torch.Tensor - The hidden states of the shape [batch_size, seq_len, hidden_size]
        span_indices: torch.Tensor - The span indices of entities of the shape [batch_size, num_entities, 2]
        """
        B, L, D = h.size()
        B, num_entities, _ = span_indices.size()
        entity_reps = self.span_marker(h, span_indices)  #  ([B, num_possible_spans, D])  

        # Create a tensor [B, num_entities, num_entities, D] by repeating entity_reps for pairing
        # Expanding entity_reps to pair each with every other
        entity_reps_expanded = entity_reps.unsqueeze(2).expand(-1, -1, num_entities, -1)
        entity_reps_tiled = entity_reps.unsqueeze(1).expand(-1, num_entities, -1, -1)

        
        # Concatenate the representations of all possible pairs
        # The shape becomes [B, num_entities, num_entities, 2D]
        pair_reps = torch.cat([entity_reps_expanded, entity_reps_tiled], dim=3)  # [B, num_entities, num_entities, 2*hidden_size]

        # Now we have an upper triangular matrix where each [i, j] element is the pair combination
        # of the i-th and j-th entities. We need to remove the diagonal and lower triangular parts.
        triu_mask = torch.triu(torch.ones(num_entities, num_entities), diagonal=1).bool()
        combined_pairs = pair_reps[:, triu_mask]

        # combined_pairs is now a tensor of shape [batch_size, num_pairs, 2*hidden_size]
        # where num_pairs is num_entities * (num_entities - 1) / 2, the number of unique pairs.

        combined_pairs_out = self.out_project(combined_pairs)

        # import ipdb;ipdb.set_trace() 

        return combined_pairs_out

# Example usage:
# h is the hidden states from a transformer model, shape: [batch_size, seq_len, hidden_size]
# span_indices is a tensor containing the start and end indices for each entity, shape: [batch_size, num_entities, 2]
# efficient_entity_pairs_marker = EfficientEntityPairsMarker(hidden_size, max_width)
# efficient_entity_pairs_rep = efficient_entity_pairs_marker(h, span_indices)
# efficient_entity_pairs_rep now contains representations for all possible entity pairs
