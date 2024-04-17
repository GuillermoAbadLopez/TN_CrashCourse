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

    tensors = [0 for _ in range(n_sites)]

    tensors[0] = np.zeros((1, d, min(chi, d)))
    for i in range(1, n_sites):
        d_l_leg = min(chi, d**i, d ** (n_sites - i))  # Pau does: tensors[k - 1].shape[2]
        d_r_leg = min(chi, d ** (i + 1), d ** (n_sites - i - 1))  # Pau does: min(chi, tensors[k - 1].shape[2] * 2 ...
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

    tensors = [0 for _ in range(n_sites)]

    init_site_shape = (1, d, min(chi, d))
    tensors[0] = (np.random.rand(init_site_shape) + 1j * np.random.rand(init_site_shape)) / np.sqrt(2)

    for i in range(1, n_sites):
        d_l_leg = min(chi, d**i, d ** (n_sites - i))  # Pau does: tensors[k - 1].shape[2]
        d_r_leg = min(chi, d ** (i + 1), d ** (n_sites - i - 1))  # Pau does: min(chi, tensors[k - 1].shape[2] * 2 ...
        site_shape = (d_l_leg, d, d_r_leg)
        tensors[i] = (np.random.rand(site_shape) + 1j * np.random.rand(site_shape)) / np.sqrt(2)

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

    tensors = zeros_mps(d, chi, n_sites)

    for i in range(n_sites):
        tensors[i][0][0][0] = 1.0

    return tensors
