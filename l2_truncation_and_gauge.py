"""Module for the second set of exercises in the course."""

import pytest

from l1_basics import product_mps


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
