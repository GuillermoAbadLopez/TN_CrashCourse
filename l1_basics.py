"""Module for the first set of exercises in the course."""

import numpy as np


def zeros_mps(d: int, chi: int, n_sites: int) -> list[np.array]:
    """Create an MPS filled with zeros.

    Args:
        d (int): bond dimension of the physical indices.
        chi (int): maximum bond dimension of the virtual indices.
        n_sites (int): number of sites in the MPS.

    Returns:
        list[np.array]: list of tensors representing the MPS.
    """
    # First we create a list for each qubit
    tensors = [0 for _ in range(n_sites)]

    # Then we create the tensors for the first qubit
    tensors[0] = np.zeros((1, d, min(chi, d)))

    # Then we create the tensors for the rest of the qubits
    for i in range(1, n_sites):
        # Dimension for the left and right legs of the tensor
        d_l_leg = min(chi, d**i, d ** (n_sites - i))  # Pau does: tensors[k - 1].shape[2]
        d_r_leg = min(chi, d ** (i + 1), d ** (n_sites - i - 1))  # Pau does: min(chi, tensors[k - 1].shape[2] * 2 ...
        # Create the tensor with the dimensions calculated
        tensors[i] = np.zeros((d_l_leg, d, d_r_leg))

    return tensors


def random_mps(d: int, chi: int, n_sites: int) -> list[np.array]:
    """Create an MPS filled with random complex numbers.

    Args:
        d (int): bond dimension of the physical indices.
        chi (int): maximum bond dimension of the virtual indices.
        n_sites (int): number of sites in the MPS.

    Returns:
        list[np.array]: list of tensors representing the MPS.
    """
    # First we create a list for each qubit
    tensors = [0 for _ in range(n_sites)]

    # Then we create the tensors for the first qubit
    tensors[0] = (np.random.rand(1, d, min(chi, d)) + 1j * np.random.rand(1, d, min(chi, d))) / np.sqrt(2)

    # Then we create the tensors for the rest of the qubits
    for i in range(1, n_sites):
        # Dimension for the left and right legs of the tensor
        d_l_leg = min(chi, d**i, d ** (n_sites - i))  # Pau does: tensors[k - 1].shape[2]
        d_r_leg = min(chi, d ** (i + 1), d ** (n_sites - i - 1))  # Pau does: min(chi, tensors[k - 1].shape[2] * 2 ...
        # Create the tensor with the dimensions calculated
        tensors[i] = (np.random.rand(d_l_leg, d, d_r_leg) + 1j * np.random.rand(d_l_leg, d, d_r_leg)) / np.sqrt(2)

    return tensors


def product_mps(d: int, chi: int, n_sites: int) -> list[np.array]:
    """Create an MPS representing the state |000...>.

    Args:
        d (int): bond dimension of the physical indices.
        chi (int): maximum bond dimension of the virtual indices.
        n_sites (int): number of sites in the MPS.

    Returns:
        list[np.array]: list of tensors representing the MPS.
    """
    # Create the empty MPS
    tensors = zeros_mps(d, chi, n_sites)

    # Set each tensor to |0>
    for i in range(n_sites):
        tensors[i][0][0][0] = 1.0

    return tensors
