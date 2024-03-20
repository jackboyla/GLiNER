import torch
from torch import nn
from span_rep import create_projection_layer

class RelationRepresentation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Define a feed-forward network for processing the combined representation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h: torch.Tensor, span1_idx: torch.Tensor, span2_idx: torch.Tensor) -> torch.Tensor:
        """
        span1_rep: torch.Tensor - The representation of the first span
        span2_rep: torch.Tensor - The representation of the second span
        context_rep: torch.Tensor - The contextual representation between the two spans
        """

        B, L, D = h.size()

        # Get representations for the first and second entity spans
        span1_rep = self.span_marker1(h, span1_idx)
        span2_rep = self.span_marker2(h, span2_idx)

        # Assuming that span1_idx and span2_idx are ordered such that span1_idx < span2_idx
        # Extract context between the two spans
        # This can be done in several ways, for example, by averaging the hidden states between the two entities
        start_context_idx = span1_idx[:, :, 1] + 1
        end_context_idx = span2_idx[:, :, 0]
        context_reps = []
        for b in range(B):
            context_reps.append(torch.mean(h[b, start_context_idx[b]:end_context_idx[b], :], dim=0))
        context_rep = torch.stack(context_reps, dim=0).unsqueeze(1)  # Add an extra dimension to match batch size

        # Concatenate the span representations and context
        combined_rep = torch.cat([span1_rep, context_rep, span2_rep], dim=-1)

        # Project the combined representation for relation classification
        relation_rep = self.relation_project(combined_rep)

        return relation_rep
    


class SpanMarkerV0(nn.Module):
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

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat).view(B, L, self.max_width, D)
