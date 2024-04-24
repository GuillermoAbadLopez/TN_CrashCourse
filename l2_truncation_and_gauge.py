"""Module for the second set of exercises in the course."""

import numpy as np
import pytest
from ncon import ncon

from l1_basics import product_mps, random_mps


### 1.a)
def coefficient_MPS(bit_string, mps):
    """Compute the coefficient of a bit string in an MPS.

    Args:
        bit_string (str): bit string for which we want to compute the coefficient.
        mps (list[np.array]): list of tensors representing the MPS.

    Returns:
        complex: coefficient of the bit string in the MPS.
    """
    if len(bit_string) != len(mps):
        raise ValueError("The length of the bit string and the MPS do not match.")

    # Initialize the projection operator

    # Loop over all the qubits
    for q, state_projection in enumerate(bit_string):
        # Get the projected state tensor for that qubit:
        tensor = mps[q]
        projected_tensor = tensor[:, int(state_projection), :]
        # Multiply the coefficient by the value of the tensor
        P = P @ projected_tensor if q != 0 else projected_tensor
    return P[0, 0]


### 1.b)

# TESTING IT AGAINST THE OLD FUNCTION product_mps:
mps = product_mps(d=2, chi=10, n_sites=3)

# Test the coefficient of the |000> state is 1:
bit_string = "000"
assert 1 == coefficient_MPS(bit_string, mps)

# Test the rest of coefficient are 0:
for bit_string in ["001", "010", "011", "100", "101", "110", "111"]:
    assert 0 == coefficient_MPS(bit_string, mps)

# Test different shapes of MPS and strings:
for bit_string in ["01", "0100"]:
    with pytest.raises(ValueError) as excinfo:
        coefficient_MPS(bit_string, mps)
        assert str(excinfo.value) == "The length of the bit string and the MPS do not match."


### 2.a)
def left_orthogonalize_tensor(tensor: np.array, dtol=1e-12) -> tuple[np.array, np.array]:
    """Left orthogonalizes a given tensor from an MPS.

    Args:
        tensor (np.ndarray): The input tensor to be left orthogonalized.
        dtol (float, optional): The tolerance for determining non-zero eigenvalues. Defaults to 1e-12.

    Returns:
        np.ndarray: The left orthonormalized tensor.
        np.ndarray: Orthogonalization matrix to be plugged into the tensor on the right of the input tensor in the MPS.
    """
    # Calculate the SVD of the tensor
    U, S, V = np.linalg.svd(tensor.reshape(-1, tensor.shape[-1]), full_matrices=False)
    # Calculate the rank of the tensor
    rank = np.sum(S > dtol)
    # Calculate the left orthonormalized tensor
    left_tensor = U[:, :rank].reshape(tensor.shape[0], tensor.shape[1], rank)
    # Calculate the matrix for orthogonalization
    orth_matrix = np.diag(S[:rank]) @ V[:rank]
    return left_tensor, orth_matrix


### 2.b)
def left_orthogonalize(mps: list[np.array], n_stop: int = None, dtol: float = 1e-12) -> list[np.array]:
    """Bring the MPS into left orthogonal form by left orthogonalizing the tensors, starting from the leftmost tensor
    and moving right sites positions.

    Args:
        tensors (list[np.array]): The list of tensors of the mps
        n_sites (int, optional): Number of sites of the MPS to bring to left orthogonal form. If not specified, all
            tensors up to the second to last one are left orthogonalized.
        dtol (float, optional):  The tolerance for determining non-zero eigenvalues in the diagonalization process.
            Defaults to 1e-12.

    Raises:
        - ValueError: If sites is an integer bigger than the number of sites minus one.

    Returns:
        list[np.array]: The list of tensors in left orthogonal form.
    """
    # No stop provided
    if n_stop is None:
        n_stop = len(mps) - 1

    # Errors check
    if not isinstance(n_stop, int):
        raise ValueError("Number of sites must be an integer.")
    if n_stop > len(mps) - 1:
        raise ValueError("Number of sites is bigger than the number of sites minus one.")

    # Loop over the tensors from left to right
    for i in range(n_stop):
        # Left orthogonalize the tensor
        left_tensor, orth_matrix = left_orthogonalize_tensor(mps[i], dtol)
        # Update the tensor and the next tensor in the MPS
        mps[i] = left_tensor
        mps[i + 1] = ncon([orth_matrix, mps[i + 1]], [[-1, 2], [2, -3, -4]])
        # mps[i + 1] = np.einsum("ijk,kl->ijl", orth_matrix, mps[i + 1])  # Without ncon
    return mps


### 3.a)
def norm_MPS(mps: list):
    """Compute the norm of an MPS.

    Args:
        mps (list[np.array]): list of tensors representing the MPS.

    Returns:
        float: norm of the MPS.
    """
    # Initialize the norm tensor with a 1 dimensional Identity tensor
    norm = np.array([[1]])
    for tensor in mps:
        # Conjugate the tensor
        tensor_conj = np.conj(tensor)
        # Contract the "norm tensor" on the left, with the next pair of tensor-conjugated until the end
        norm = ncon([norm, tensor, tensor_conj], [[1, 2], [1, 3, -4], [2, 3, -5]])

    # Contract the last two legs of the norm tensor
    norm = ncon([norm], [[1, 1]])
    return norm


### 3.b)
def norm_most_right_tensor(mps):
    """Compute the norm of the most left tensor of an MPS.

    Args:
        mps (list[np.array]): list of tensors representing the MPS.

    Returns:
        float: norm of the most left tensor.
    """
    # Get the most right tensor
    most_left_tensor = mps[-1]
    # Conjugate the tensor
    conjugate_tensor = np.conjugate(most_left_tensor)
    # Contract the tensor with its conjugate
    norm = ncon([most_left_tensor, conjugate_tensor], [[1, 2, 3], [1, 2, 3]])
    return norm


# TESTING BOTH NORMS GIVE THE SAME!
random_mps = random_mps(d=2, chi=10, n_sites=5)

first_norm = norm_most_right_tensor(left_orthogonalize(random_mps))
second_norm = norm_MPS(random_mps)

print(first_norm, "\n", second_norm)
assert np.isclose(first_norm, second_norm)
