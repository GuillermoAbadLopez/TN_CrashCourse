"""Module for the testing the first set of exercises in the course."""

import numpy as np

from l1_basics import product_mps, random_mps, zeros_mps


# PAU ANSWERS:
# 1)
def Zeros_MPS(d, chi, n_sites):
    """Write a function that creates an MPS with open boundary conditions where all the tensor elements are 0."""
    tensors = [0 for x in range(n_sites)]
    tensors[0] = np.zeros((1, d, min(chi, d)))
    for k in range(1, n_sites):
        tensors[k] = np.zeros(
            (tensors[k - 1].shape[2], d, min(chi, tensors[k - 1].shape[2] * 2, d ** (n_sites - k - 1)))
        )
    return tensors


# 2.a)
def Random_MPS(d, chi, n_sites):
    """Write a function Random_MPS(d, chi, n_sites) that returns an MPS with open boundary conditions with all its
    tensor elements as random complex numbers, with the same conditions as in exercise 1."""
    tensors = [0 for x in range(n_sites)]
    tensors[0] = np.random.rand(1, d, min(chi, d)) + 1j * np.random.rand(1, d, min(chi, d))
    for k in range(1, n_sites):
        tensors[k] = np.random.rand(
            tensors[k - 1].shape[2], d, min(chi, tensors[k - 1].shape[2] * 2, d ** (n_sites - k - 1))
        ) + 1j * np.random.rand(
            tensors[k - 1].shape[2], d, min(chi, tensors[k - 1].shape[2] * 2, d ** (n_sites - k - 1))
        )
    return tensors


# 2.b)
def Product_MPS(d, chi, n_sites):
    """Write a function Product_MPS(d, chi, n_sites) that returns an MPS with open boundary conditions that
    represent the quantum state |000…>, using the same conditions as exercise 1."""
    tensors = Zeros_MPS(d, chi, n_sites)
    for k in range(0, n_sites):
        (tensors[k])[0][0][0] = 1.0
    return tensors


# TESTING 1):
for j in range(1, 4):
    for p in range(1, 5):
        # print("NEW COMPARISON \n")
        # print(zeros_mps(j * 2, j, p))
        # print("\n")
        # print(Zeros_MPS(j * 2, j, p))

        solution = Zeros_MPS(j * 2, j, p)
        answer = zeros_mps(j * 2, j, p)

        for tensor in answer:
            assert tensor.all() == solution[0].all()


# TESTING 2.a):
for j in range(1, 4):
    for p in range(1, 5):
        solution = Random_MPS(j * 2, j, p)
        answer = random_mps(j * 2, j, p)

        # CANNOT TEST, BECAUSE THE RANDOM NUMBERS ARE DIFFERENT (AND SQRT(2) factor is missing in the solution)
        # for tensor in answer:
        #     assert tensor.all() == solution[0].all()


# TESTING 2.b):
# TESTING 1):
for j in range(1, 4):
    for p in range(1, 5):
        solution = Product_MPS(j * 2, j, p)
        answer = product_mps(j * 2, j, p)

        for tensor in answer:
            assert tensor.all() == solution[0].all()
