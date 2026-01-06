import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatingNetwork(nn.Module):
    def __init__(self, N, M, Experts, dtype=torch.float32):
        super().__init__()
        self.conv = nn.Conv1d(N, N, kernel_size=2, padding=0, bias=True, dtype=dtype)
        self.softmax_temp1 = nn.Parameter(torch.tensor([0.1], dtype=dtype))
        self.D = nn.Parameter(torch.zeros(N, M, dtype=dtype))
        self.D.data[:, :N] = torch.eye(N, dtype=dtype)
        self.mlp_layer1 = nn.Linear(M + N, Experts, dtype=dtype)
        self.mlp_layer1.bias = nn.Parameter(
            torch.zeros(Experts, dtype=dtype), requires_grad=False
        )
        self.mlp_layer2 = nn.Linear(Experts, Experts, dtype=dtype)
        self.mlp_layer2.bias = nn.Parameter(
            torch.zeros(Experts, dtype=dtype), requires_grad=False
        )
        self.softmax_temp2 = nn.Parameter(torch.tensor([0.1], dtype=dtype))
        self.sigma = nn.Parameter(torch.ones(N, dtype=dtype) * 0.05, requires_grad=True)

    def forward(self, context, z, precomputed_cnn=None):
        # context: (seq_length, batch_size, N)
        # z: (M, batch_size)
        # precomputed_cnn: Optional precomputed CNN features for inference (seq_length-1, batch_size, N)

        seq_length, batch_size, N = context.shape
        M = z.shape[0]

        # Compute attention weights
        z_obs = self.D @ z.detach()
        # Only add noise during training, skip during inference for speed
        if self.training:
            z_current = z_obs + self.sigma.unsqueeze(1) * torch.randn(
                N, batch_size, dtype=z.dtype, device=z.device
            )
        else:
            z_current = z_obs

        z_current_t = z_current.transpose(0, 1)
        context_frames = context[:-1]

        distances = torch.sum(
            torch.abs(context_frames - z_current_t.unsqueeze(0)), dim=2
        )
        # Use minimum temperature floor to prevent numerical instability
        temp1 = torch.clamp(torch.abs(self.softmax_temp1[0]), min=1e-4)
        attention_weights = F.softmax(-distances / temp1, dim=0)

        # Process context with convolution
        # Use precomputed CNN features if provided, otherwise compute them
        if precomputed_cnn is not None:
            encoded = precomputed_cnn
        else:
            context_for_conv = context.permute(1, 2, 0)
            encoded = self.conv(context_for_conv)
            encoded = encoded.permute(2, 0, 1)

        # Build weighted embedding
        weighted_encoded = encoded * attention_weights.unsqueeze(2)
        embedding = torch.sum(weighted_encoded, dim=0)
        embedding = embedding.transpose(0, 1)

        # Predict expert weights
        combined = torch.cat([embedding, z], dim=0)
        combined_t = combined.transpose(0, 1)
        mlp_output = self.mlp_layer2(F.relu(self.mlp_layer1(combined_t)))
        # Use minimum temperature floor to prevent numerical instability
        temp2 = torch.clamp(torch.abs(self.softmax_temp2[0]), min=1e-4)
        w_exp = F.softmax(-mlp_output.transpose(0, 1) / temp2, dim=0)
        return w_exp

    def gaussian_init(self, M, N, dtype=torch.float32):
        return torch.randn(M, N, dtype=dtype) * 0.01


class ExpertNetwork(nn.Module):
    """Base class for different expert architectures."""

    def __init__(self, M, P=0, probabilistic=False, dtype=torch.float32):
        super().__init__()
        self.M = M
        self.P = P
        self.probabilistic = probabilistic
        self.dtype = dtype

        # Parameter for probabilistic experts
        if probabilistic:
            self.sigma = nn.Parameter(
                torch.ones(1, dtype=dtype) * 0.05, requires_grad=True
            )

    def forward(self, z):
        raise NotImplementedError("Subclasses must implement forward method")

    def add_noise(self, z):
        """Add stochasticity to the latent state if in probabilistic mode.

        Args:
            z: Input tensor
        """
        if self.probabilistic:
            batch_size = z.shape[1]
            noise = torch.randn(self.M, batch_size, dtype=z.dtype, device=z.device)
            return z + self.sigma * noise
        return z

    def gaussian_init(self, M, N):
        return torch.randn(M, N, dtype=self.dtype) * 0.01

    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = R.T @ R / M + np.eye(M)
        lambd = np.max(np.abs(np.linalg.eigvals(K)))
        return K / lambd


class AlmostLinearRNN(ExpertNetwork):
    """Almost linear RNN expert architecture."""

    def __init__(self, M, P, probabilistic=False, dtype=torch.float32):
        super().__init__(M, P, probabilistic, dtype=dtype)
        self.A, self.W, self.h = self.initialize_A_W_h(M)

    def forward(self, z):
        # z: (M, batch_size)
        # Split z into regular and ReLU parts
        z1 = z[: -self.P, :]
        z2 = F.relu(z[-self.P :, :])
        zcat = torch.cat([z1, z2], dim=0)

        output = self.A.unsqueeze(-1) * z + self.W @ zcat + self.h.unsqueeze(-1)

        # Add stochasticity if probabilistic
        if self.probabilistic:
            output = self.add_noise(output)

        return output

    def initialize_A_W_h(self, M):
        A = torch.nn.Parameter(
            torch.diag(
                torch.tensor(self.normalized_positive_definite(M), dtype=self.dtype)
            )
        )
        W = torch.nn.Parameter(self.gaussian_init(M, M))
        h = torch.nn.Parameter(torch.zeros(M, dtype=self.dtype))
        return A, W, h


class BatchedAlmostLinearRNN(nn.Module):
    """Batched version of AlmostLinearRNN that computes all experts in one operation."""

    def __init__(self, M, P, num_experts, probabilistic=False, dtype=torch.float32):
        super().__init__()
        self.M = M
        self.P = P
        self.num_experts = num_experts
        self.probabilistic = probabilistic
        self.dtype = dtype

        # Initialize batched parameters: (num_experts, M) for A and h, (num_experts, M, M) for W
        self.A = nn.Parameter(torch.zeros(num_experts, M, dtype=dtype))
        self.W = nn.Parameter(torch.zeros(num_experts, M, M, dtype=dtype))
        self.h = nn.Parameter(torch.zeros(num_experts, M, dtype=dtype))

        if probabilistic:
            self.sigma = nn.Parameter(torch.ones(1, dtype=dtype) * 0.05)

        # Initialize each expert's parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        for i in range(self.num_experts):
            # Initialize A as diagonal of normalized positive definite matrix
            R = np.random.randn(self.M, self.M).astype(np.float32)
            K = R.T @ R / self.M + np.eye(self.M)
            lambd = np.max(np.abs(np.linalg.eigvals(K)))
            K_normalized = K / lambd
            self.A.data[i] = torch.diag(torch.tensor(K_normalized, dtype=self.dtype))

            # Initialize W with small random values
            self.W.data[i] = torch.randn(self.M, self.M, dtype=self.dtype) * 0.01

            # h is already zeros

    def forward(self, z):
        """
        Compute all experts in a single batched operation.

        Args:
            z: Latent state of shape (M, batch_size)

        Returns:
            All expert outputs stacked: (num_experts, M, batch_size)
        """
        M, batch_size = z.shape

        # Split z into regular and ReLU parts
        z1 = z[: -self.P, :]  # (M-P, batch_size)
        z2 = F.relu(z[-self.P :, :])  # (P, batch_size)
        zcat = torch.cat([z1, z2], dim=0)  # (M, batch_size)

        # Compute A * z for all experts: (num_experts, M, batch_size)
        # A: (num_experts, M), z: (M, batch_size)
        Az = self.A.unsqueeze(-1) * z.unsqueeze(0)  # (num_experts, M, batch_size)

        # Compute W @ zcat for all experts using batched matmul
        # W: (num_experts, M, M), zcat: (M, batch_size)
        # Result: (num_experts, M, batch_size)
        Wz = torch.bmm(self.W, zcat.unsqueeze(0).expand(self.num_experts, -1, -1))

        # Add bias: h: (num_experts, M) -> (num_experts, M, 1)
        output = Az + Wz + self.h.unsqueeze(-1)

        # Add noise if probabilistic
        if self.probabilistic:
            noise = torch.randn_like(output) * self.sigma
            output = output + noise

        return output


class ClippedShallowPLRNN(ExpertNetwork):
    """Clipped shallow PLRNN expert architecture."""

    def __init__(self, M, hidden_dim=50, probabilistic=False, dtype=torch.float32):
        super().__init__(M, hidden_dim, probabilistic, dtype=dtype)
        self.A = torch.nn.Parameter(
            torch.diag(
                torch.tensor(self.normalized_positive_definite(M), dtype=self.dtype)
            )
        )
        self.W1 = torch.nn.Parameter(self.gaussian_init(M, hidden_dim))
        self.W2 = torch.nn.Parameter(self.gaussian_init(hidden_dim, M))
        self.h1 = torch.nn.Parameter(torch.zeros(M, dtype=self.dtype))
        self.h2 = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=self.dtype))

    def forward(self, z):
        # z: (M, batch_size)
        W2z = self.W2 @ z
        output = (
            self.A.unsqueeze(-1) * z
            + self.W1 @ (F.relu(W2z + self.h2.unsqueeze(-1)) - F.relu(W2z))
            + self.h1.unsqueeze(-1)
        )

        # Add stochasticity if probabilistic
        if self.probabilistic:
            output = self.add_noise(output)

        return output


class DynaMix(nn.Module):
    def __init__(
        self,
        M,
        N,
        Experts,
        P=2,
        hidden_dim=50,
        expert_type="almost_linear_rnn",
        probabilistic_expert=False,
        dtype=torch.float32,
    ):
        """
        Initialize a DynaMix model.

        Args:
            M: Dimension of latent state
            N: Dimension of observation space
            Experts: Number of experts
            P: Number of ReLU dimensions
            hidden_dim: Hidden dimension for clipped shallow PLRNN
            expert_type: Type of expert to use ("almost_linear_rnn" or "clipped_shallow_plrnn")
            probabilistic_expert: Whether to use probabilistic experts
            dtype: Data type for model parameters (default: torch.float32)
        """
        super().__init__()

        self.expert_type = expert_type
        self.probabilistic_expert = probabilistic_expert
        self.dtype = dtype
        self._use_batched_experts = False

        # Use batched experts for almost_linear_rnn (much faster)
        if expert_type == "almost_linear_rnn":
            self._use_batched_experts = True
            self.batched_experts = BatchedAlmostLinearRNN(
                M, P, Experts, probabilistic=probabilistic_expert, dtype=dtype
            )
            # Keep empty ModuleList for compatibility
            self.experts = nn.ModuleList()
        elif expert_type == "clipped_shallow_plrnn":
            # Fall back to individual experts for clipped_shallow_plrnn
            self.experts = nn.ModuleList()
            for _ in range(Experts):
                self.experts.append(
                    ClippedShallowPLRNN(
                        M, hidden_dim, probabilistic=probabilistic_expert, dtype=dtype
                    )
                )
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

        self.gating_network = GatingNetwork(N, M, Experts, dtype=dtype)
        self.B = nn.Parameter(self.uniform_init((N, M), dtype=dtype))
        self.N = N
        self.Experts = Experts
        self.P = P
        self.hidden_dim = hidden_dim
        self.M = M

    def step(self, z, context, precomputed_cnn=None):
        # z: (M, batch_size)
        # context: (seq_length, batch_size, N)
        # precomputed_cnn: Optional precomputed CNN features

        # Compute expert weights
        w_exp = self.gating_network(
            context, z, precomputed_cnn=precomputed_cnn
        )  # (Experts, batch_size)

        # Compute all expert outputs
        if self._use_batched_experts:
            # Fast path: single batched operation for all experts
            # Shape: (Experts, M, batch_size)
            expert_outputs = self.batched_experts(z)
        else:
            # Slow path: iterate over experts (for clipped_shallow_plrnn)
            expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=0)

        # Apply weights: w_exp is (Experts, batch_size), need to broadcast to (Experts, M, batch_size)
        weighted_outputs = expert_outputs * w_exp.unsqueeze(1)

        # Sum across experts: (M, batch_size)
        return weighted_outputs.sum(dim=0)

    def forward(self, z, context, precomputed_cnn=None):
        """
        Forward pass through the DynaMix model.

        Args:
            z: Latent state of shape (M, batch_size)
            context: Context data of shape (seq_length, batch_size, N)
            precomputed_cnn: Optional precomputed CNN features to avoid redundant computation for inference

        Returns:
            Updated latent state
        """
        return self.step(z, context, precomputed_cnn=precomputed_cnn)

    def precompute_cnn(self, context):
        """
        Precompute CNN features for more efficient inference.

        Args:
            context: Context data of shape (seq_length, batch_size, N)

        Returns:
            Precomputed CNN features
        """
        # Process context with convolution
        context_for_conv = context.permute(1, 2, 0)
        encoded = self.gating_network.conv(context_for_conv)

        return encoded.permute(2, 0, 1)

    def uniform_init(self, shape, dtype=torch.float32):
        din = shape[-1]
        r = 1 / np.sqrt(din)
        return (torch.rand(shape, dtype=dtype) * 2 - 1) * r

    def gaussian_init(self, M, N):
        return torch.randn(M, N, dtype=self.dtype) * 0.01


def print_model_parameters(model):
    """Print simplified breakdown of model parameters by component."""
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "-" * 60)
    print("Model Parameter Summary:")
    print(f"  Architecture: DynaMix with {model.expert_type} experts")
    if model.expert_type == "almost_linear_rnn":
        print(
            f"  Dimensions: M={model.M}, N={model.N}, Experts={model.Experts}, P={model.P}"
        )
    else:
        print(
            f"  Dimensions: M={model.M}, N={model.N}, Experts={model.Experts}, Hidden dim={model.hidden_dim}"
        )
    print(f"  Probabilistic experts: {model.probabilistic_expert}")

    # Count parameters
    gating_params = sum(p.numel() for p in model.gating_network.parameters())
    # Count expert params from either batched or individual experts
    if hasattr(model, "batched_experts") and model._use_batched_experts:
        expert_params = sum(p.numel() for p in model.batched_experts.parameters())
    else:
        expert_params = sum(
            p.numel() for expert in model.experts for p in expert.parameters()
        )
    b_params = model.B.numel()

    # Print parameter counts
    print("\nParameter counts:")
    print(f"  Gating Network: {gating_params:,} ({gating_params / total_params:.1%})")
    print(f"  Experts: {expert_params:,} ({expert_params / total_params:.1%})")
    print(f"  Observation matrix: {b_params:,} ({b_params / total_params:.1%})")
    print(f"  Total: {total_params:,} parameters")
    print("-" * 60)
