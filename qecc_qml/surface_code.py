"""Simplified surface code error correction stub using 3-qubit majority vote."""

import random
from typing import List


class SurfaceCodeStub:
    """3-qubit repetition code as a simplified surface code stub.

    Encodes a single logical qubit into 3 physical qubits using a repetition
    code. Decoding uses majority vote. This is a classical simulation of the
    error correction concept, not a true quantum surface code.
    """

    def encode(self, bit: int) -> List[int]:
        """Encode a single bit into a 3-qubit repetition code.

        Args:
            bit: The logical bit (0 or 1) to encode.

        Returns:
            List of 3 physical bits [bit, bit, bit].
        """
        return [bit, bit, bit]

    def decode(self, bits: List[int]) -> int:
        """Decode 3 physical bits using majority vote.

        Args:
            bits: List of 3 physical bits.

        Returns:
            The majority vote (0 or 1).
        """
        return 1 if sum(bits) >= 2 else 0

    def correct(self, noisy_bits: List[int], flip_prob: float = 0.1) -> int:
        """Apply random bit flips then decode via majority vote.

        Args:
            noisy_bits: List of 3 physical bits that may contain errors.
            flip_prob: Probability of flipping each bit independently.

        Returns:
            The decoded logical bit after error correction.
        """
        corrected = []
        for bit in noisy_bits:
            if random.random() < flip_prob:
                corrected.append(bit ^ 1)  # flip the bit
            else:
                corrected.append(bit)
        return self.decode(corrected)
