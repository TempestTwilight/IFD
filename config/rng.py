"""RNG management for deterministic reproducibility."""

import random
from typing import Any

import numpy as np
import torch


class RNGManager:
    """Manages deterministic RNG streams derived from master seed.

    Uses numpy.random.SeedSequence for principled stream derivation.
    Each stream is independent and deterministic.
    """

    def __init__(self, master_seed: int):
        """Initialize RNG manager with master seed.

        Args:
            master_seed: Single seed for all experiment randomness.
        """
        self.master_seed = master_seed
        self._seq = np.random.SeedSequence(master_seed)

        # Derive child seeds for different streams
        children = self._seq.spawn(10)
        self._seeds = {
            "global": children[0].generate_state(1)[0],
            "partition": children[1].generate_state(1)[0],
            "model_init": children[2].generate_state(1)[0],
            "training": children[3].generate_state(1)[0],
            "client_sampling": children[4].generate_state(1)[0],
            "attack": children[5].generate_state(1)[0],
            "eval": children[6].generate_state(1)[0],
            "reserved1": children[7].generate_state(1)[0],
            "reserved2": children[8].generate_state(1)[0],
            "reserved3": children[9].generate_state(1)[0],
        }

        # Initialize global RNG state
        self._init_global()

    def _init_global(self) -> None:
        """Initialize process-global RNG states for compatibility."""
        seed = int(self._seeds["global"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_seed(self, stream: str) -> int:
        """Get seed for a named stream.

        Args:
            stream: Stream name (partition, model_init, etc.)

        Returns:
            Deterministic seed for the stream.
        """
        if stream not in self._seeds:
            raise ValueError(f"Unknown stream: {stream}. Available: {list(self._seeds.keys())}")
        return int(self._seeds[stream])

    def get_numpy_rng(self, stream: str) -> np.random.Generator:
        """Get numpy Generator for a stream.

        Args:
            stream: Stream name.

        Returns:
            Independent numpy Generator.
        """
        return np.random.default_rng(self.get_seed(stream))

    def get_torch_generator(self, stream: str, device: str = "cpu") -> torch.Generator:
        """Get torch Generator for a stream.

        Args:
            stream: Stream name.
            device: 'cpu' or 'cuda'.

        Returns:
            Independent torch Generator.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.get_seed(stream))
        return gen

    def get_rng_state(self) -> dict[str, Any]:
        """Capture current RNG state for checkpointing.

        Returns:
            Dictionary with Python/NumPy/PyTorch RNG states.
        """
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "master_seed": self.master_seed,
        }

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore RNG state from checkpoint.

        Args:
            state: Dictionary returned by get_rng_state().
        """
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
