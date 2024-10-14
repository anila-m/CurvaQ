from matplotlib import pyplot as plt
import numpy as np
import torch

from qnns.cuda_qnn import CudaPennylane


def get_zero_state(num_qubits):
    """calculates the zero vector for a given number of qubits

    Args:
        num_qubits (int): the number of qubits the zero vector should have

    Returns:
        vector: a vector with a leading 1 and 2^num_qubits -1 zeroes behind it
    """
    n = 2 ** num_qubits
    state = [[1]]
    for _ in range(n - 1):
        state.append([0])
    return torch.tensor(np.array([state]), dtype=torch.complex128)


def get_bin_index(num_bins, fidelity):
    """calculates the index of the bin corresponding to the fidelity level

    Args:
        num_bins (int): how many bins are wanted
        fidelity (float): the fidelity for which to calculate the corresponding bin

    Returns:
        int: bin index of fidelity
    """
    return int(np.floor(fidelity * num_bins))


def sample_params(num_params):
    """generates num_params random numbers between 0 and 2pi

    Args:
        num_params (int): how many parameters you need

    Returns:
        array: an array filled with uniformly random numbers between 0 and 2pi
    """
    return np.random.rand(num_params, 1) * np.pi * 2


def calc_fidelity(U1, U2, num_qubits):
    """calculates the fidelity between two unitaries on a zero vector qubit input

    Args:
        U1 (tensor): first unitary
        U2 (tensor): second unitary
        num_qubits (int): number of qubits

    Returns:
        float: fidelity between the unitaries
    """
    X = get_zero_state(num_qubits)
    y_1 = torch.matmul(U1, X)
    y_2 = torch.matmul(U2, X).conj()
    dot_products = torch.sum(torch.mul(y_1, y_2), dim=[1, 2])
    return torch.abs(dot_products) ** 2


def uniform_haar_distribution(num_qubits, num_bins):
    """calculates the values of the uniform haar distribution for num_bins bins and num_qubits qubits, 
       which corresponds to the fidelity distribution of haar random unitaries

    Args:
        num_qubits (int): number of qubits
        num_bins (int): number of bins for the distribution

    Returns:
        float: expected fidelity for haar random unitaries
    """
    distribution = []
    N = 2 ** num_qubits
    for i in range(num_bins):
        F = i / num_bins
        haar_prob = (N - 1) * (1 - F) ** (N - 2)
        distribution.append(haar_prob)
    return np.divide(distribution, np.sum(distribution))


def KL_divergence(p, q):
    """calculates the Kullback Leibler Divergence

    Args:
        p (array): first distribution
        q (array): second distribution

    Returns:
        float: divergence between the two distribution according to KL-D
    """
    return np.sum(np.where((p * q) != 0, p * np.log(p / q), 0))


def expressibility(num_tries, num_bins, num_qubits):
    """calculates the expressibility of an ansatz 
       by comparing the distribution of the ansatzes' fidelity with that 
       of haar random unitaries with the KL-Divergence

    Args:
        num_tries (int): how many sampled tries
        num_bins (int): number of bins of distributions
        num_qubits (int): number of qubits of unitaries
    """
    haar_dist = uniform_haar_distribution(num_qubits, num_bins)
    pl_dist = [0] * num_bins
    qnn1 = CudaPennylane(num_wires=num_qubits, num_layers=1, device="cpu")
    qnn2 = CudaPennylane(num_wires=num_qubits, num_layers=1, device="cpu")
    for _ in range(num_tries):
        qnn1.params = torch.tensor(
            sample_params(3 * num_qubits), dtype=torch.float64, requires_grad=True
        ).reshape(qnn1.params.shape)
        qnn2.params = torch.tensor(
            sample_params(3 * num_qubits), dtype=torch.float64, requires_grad=True
        ).reshape(qnn2.params.shape)
        V_1 = qnn1.get_tensor_V()
        V_2 = qnn2.get_tensor_V()
        fid = calc_fidelity(V_1, V_2, num_qubits).item()
        i = get_bin_index(num_bins, fid)
        pl_dist[i] = pl_dist[i] + 1
    pl_dist = np.divide(pl_dist, np.sum(pl_dist))
    expressibility = KL_divergence(pl_dist, haar_dist)
    plt.suptitle(
        f"Expressibility: {np.round(expressibility, 5)}\nDistributions for {num_bins} bins, {num_qubits} qubits and {num_tries} samples")
    x_vals = range(num_bins)
    plt.bar(x_vals, haar_dist, label="Haar", alpha=.4)
    plt.bar(x_vals, pl_dist, label="PennyLane Ansatz", alpha=.4)
    num_ticks = 11
    tick_locations = np.linspace(0, num_bins - 1, num_ticks)
    tick_labels = np.round(np.linspace(0, 1, num_ticks), 1)
    plt.ylabel("Probability")
    plt.xlabel("Fidelity")
    plt.xticks(tick_locations, tick_labels)
    plt.tight_layout()
    plt.legend()
    plt.show()


expressibility(num_tries=100000, num_bins=50, num_qubits=2)
