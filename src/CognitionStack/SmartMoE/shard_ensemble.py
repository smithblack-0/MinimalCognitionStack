"""
The shard ensemble is the vector indexing
process designed to prefetch kernels and ensure
that running the feedforward process remains
lightning fast.
"""


import torch
from torch import nn
from typing import Union, Any, Tuple, List
from abc import ABC, abstractmethod

class AbstractShardEnsemble(nn.Module, ABC):
    """
    The abstract class, which defines the prefetch
    interface and the fact that you will need to
    pass in functional kernels
    """
    @abstractmethod
    def prefetch(self,
                 ensembles: torch.Tensor
                 )->Tuple[List[torch.Tensor], torch.Tensor]:
            """
            Prefetches the relevant kernels and keys for the given ensembles.
            This must be compatible with the later forward logic.
            """
    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                weights: torch.Tensor,
                ensembles: torch.Tensor,
                kernels: List[torch.Tensor],
                )->torch.Tensor:
            """
            The resolution process, to actually run the selected
            expert collection. We seek to run, out of a provided
            prefetched collection, only the selected experts
            then combine using the indicated weights.
            x: float Tensor of shape (batch, seq, input_dim)
            weights: float Tensor of shape (batch, num_chosen_ensembles, seq)
            ensembles:  int Tensor of shape (batch, num_chosen_ensembles, seq)
            kernels: List of tensors of the correct shape
            """

class FeedforwardEnsemble(AbstractShardEnsemble):
    """
    A functional feedforward layer designed to separate
    a prefetch invokation followed by a later feedforward
    execution in terms of the relevant experts. Note that
    we use two-layer perceptrons without projections for
    simplicity.
    """

    def __init__(self,
                 ensemble_size: int,
                 input_dim: int,
                 bottleneck_dim: int,
                 dropout_rate: float):
        super().__init__()
        self.num_ensembles = ensemble_size
        self.proj_in = nn.Parameter(torch.randn(ensemble_size, bottleneck_dim, input_dim))
        self.proj_out = nn.Parameter(torch.randn(ensemble_size, input_dim, bottleneck_dim))
        self.keys = nn.Parameter(torch.randn(ensemble_size, input_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.proj_in)
        nn.init.xavier_uniform_(self.proj_out)


    def prefetch(self,
                 ensembles: torch.Tensor
                 )->Tuple[List[torch.Tensor], torch.Tensor]:
            """
            Prefetches the relevant kernels and keys for the given ensembles.
            This must be compatible with the later forward logic.
            """
            # Observe the simple prefetching logic. We are just
            # extracting from the initial ensemble dimension
            # the subset that was deemed relevant
            matrix1 = self.proj_in[ensembles, ...]
            matrix2 = self.proj_out[ensembles, ...]
            keys = self.keys[ensembles, ...]
            return [matrix1, matrix2], keys


    def forward(self,
                x: torch.Tensor,
                weights: torch.Tensor,
                ensembles: torch.Tensor,
                kernels: List[torch.Tensor],
                )->torch.Tensor:
            """
            The resolution process, to actually run the selected
            expert collection. We seek to run, out of a provided
            prefetched collection, only the selected experts
            then combine using the indicated weights.
            x: float Tensor of shape (batch, seq, input_dim)
            weights: float Tensor of shape (batch, num_chosen_ensembles, seq)
            ensembles:  int Tensor of shape (batch, num_chosen_ensembles, seq)
            kernels: List of tensors of the correct shape
            """

            # The vector indexing is happening here. Pay close attention
            # to how your local framework is handling this process. The
            # overall interface is built so you can make an abstract class
            # then drop specific cases in using unpacking logic
            matrix1, matrix2 = kernels


            # The shape is now #(batch, nchosen, seq, bottleneck, in)
            # The shape is now (batch, nchosen, seq, out, bottleneck)
            matrix1 = matrix1[ensembles, ...]
            matrix2 = matrix2[ensembles, ...]


            x = x.unsqueeze(1).unsqueeze(-1) #(batch, 1, seq, in, 1)
            x = torch.matmul(matrix1, x) #(batch, nchosen, seq, bottleneck, 1)
            x = self.activation(x)
            x = self.dropout(x)
            x = torch.matmul(matrix2, x) #(batch, nchosen, seq, out, 1)
            x = x.squeeze(-1) #(batch, nchosen, seq, out)

            while weights.dim() < x.dim():
              weights = weights.unsqueeze(-1)
            x = (x * weights).sum(dim=1)

            return x