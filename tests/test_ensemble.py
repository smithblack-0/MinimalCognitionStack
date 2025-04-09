import torch
from src.CognitionStack.SmartMoE.shard_ensemble import FeedforwardEnsemble



# Define dimensions for the test
batch_size = 2
seq_len = 5
input_dim = 512
num_layer = 10
dropout_rate = 0.05

# Define the traditional feedforward parameters.
# Then show how to derive components with an equivalent
# amount of computation

hidden_dim = 1024
num_experts = 8

num_shards_per_expert = 8
bottleneck_dim = 1024//num_shards_per_expert
num_shards = num_experts*num_shards_per_expert
num_total_shards = num_shards*num_layer

# Initialize input tensor. Note standard shape
x = torch.randn(batch_size, seq_len, input_dim)

# Randomly select ensemble indices for each (batch, timestep, expert).
# Randomly generate some weights.
# This would actually be taken care of by other selection logic
prefetch_index = torch.randint(0, num_total_shards, [num_shards])
ensembles = torch.randint(0, num_shards, (batch_size, num_shards_per_expert,
                                          seq_len))
weights = torch.rand(batch_size, num_shards_per_expert, seq_len)

# Initialize the layer
layer = FeedforwardEnsemble(num_total_shards, input_dim,
                            bottleneck_dim, dropout_rate)

# Execute the layer
# This was now run in an ensemble. Skip using keys
kernels, keys = layer.prefetch(prefetch_index)
output = layer(x, weights, ensembles, kernels)

print(output.shape)
