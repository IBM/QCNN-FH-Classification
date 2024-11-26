# (C) Copyright IBM 2017, 2020, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

import quimb as qu

from typing import List, Tuple, Any

from quimb.experimental.operatorbuilder import *
from quimb.experimental.operatorbuilder.operatorbuilder import parse_edges_to_unique


def fermi_hubbard_NN_from_edges(
    num_sites: int,
    edges_t1: List[Tuple[int, int]],
    edges_t2: List[Tuple[int, int]],
    t1: float = 1.0,
    t2: float = 0.0,
    U: float = 1.0,
    mu: float = 0.0,
) -> SparseOperatorBuilder:
    """Constructs a sparse op representation of the Fermi-Hubbard Hamiltonian.

    Args:
        num_sites (int): number of sites of hte lattice.
        edges_t1 (List[Tuple[int, int]]): edges connected by a t1 term.
        edges_t2 (List[Tuple[int, int]]): edges connected by a t2 term.
        t1 (float, optional): t1 term of the Hamiltonian. Defaults to 1.0.
        t2 (_type_, optional): t2 term of the Hamiltonian. Defaults to 0.0.
        U (float, optional): potential energy term. Defaults to 1.0.
        mu (float, optional): chemical potential. Defaults to 0.0.

    Returns:
        SparseOperatorBuilder: sparse representation of the Hamiltonian.
    """
    H = SparseOperatorBuilder()
    _, edges_t1 = parse_edges_to_unique(edges_t1)
    _, edges_t2 = parse_edges_to_unique(edges_t2)

    if t1 != 0.0:
        for cooa, coob in edges_t1:
            # hopping
            for s in "↑↓":
                H += -t1, ("+", (s, cooa)), ("-", (s, coob))
                H += -t1, ("+", (s, coob)), ("-", (s, cooa))

    if t2 != 0.0:
        for cooa, coob in edges_t2:
            # hopping
            for s in "↑↓":
                H += -t2, ("+", (s, cooa)), ("-", (s, coob))
                H += -t2, ("+", (s, coob)), ("-", (s, cooa))

    for coo in np.arange(0, num_sites):
        # interaction
        H += U, ("n", ("↑", coo)), ("n", ("↓", coo))

        # chemical potential
        H += mu, ("sn", ("↑", coo))
        H += mu, ("sn", ("↓", coo))

        # # correct the energy shift due to the chemical potential
        # H += (-mu*num_sites), ("n", ("↑", coo))
        # H += (-mu*num_sites), ("n", ("↓", coo))

    H.jordan_wigner_transform()
    return H


def build_num_particles_operator(
    num_sites: int, ham: SparseOperatorBuilder
) -> SparseOperatorBuilder:
    """Constructs the particle number operator.

    An Hamiltonian has to be provided because it is taken as a reference to extract
    the Hilbert space on which the operator is defined.

    Args:
        num_sites (int): number of sites of the lattice.
        ham (SparseOperatorBuilder): sparse representation of the Hamiltonian.

    Returns:
        SparseOperatorBuilder: sparse representation of the number operator.
    """
    terms = []
    for site in np.arange(0, num_sites):
        terms.append(
            (
                1.0,
                ("n", ("↑", site)),
            )
        )
        terms.append(
            (
                1.0,
                ("n", ("↓", site)),
            )
        )
    O_num_part = SparseOperatorBuilder(
        terms=terms,
        hilbert_space=ham.hilbert_space,  # need to embed into larger Hilbert space of Hamiltonian
    )

    # ensure fermionic
    O_num_part.jordan_wigner_transform()
    sO = O_num_part.build_sparse_matrix()
    return sO


def calculate_num_particles(num_sites: int, ham: Any, psi: Any) -> int:
    """Calculate the number of particles associated with a given wave functon.

    Args:
        num_sites (int): number of lattice sites.
        ham (Any): Hamiltonian.
        psi (Any): wave function/

    Returns:
        int: <N> (rounded to the closest integer)
    """
    sO = build_num_particles_operator(num_sites, ham)
    return np.round(qu.expec(sO, psi), 0)


def FH_Hamiltonian_NN(
    num_sites: int, t1: float, t2: float, U: float, mu: float = 0.0, pbc: int = 0
) -> Tuple[SparseOperatorBuilder, Any]:
    """Constructs a nearest-neighbour Fermi-Hubbard Hamiltonian.

    Args:
        num_sites (int): number of lattice sites.
        t1 (float): nearest-neighbour scaling factor.
        t2 (float): next-nearest-neighbour scaling factor.
        U (float): potential energy term.
        mu (float, optional): chemical potential. Defaults to 0.0.
        pbc (int, optional): whether to apply periodic boundary conditions.
            Defaults to 0 (no PBC).

    Returns:
        Tuple[SparseOperatorBuilder, Any]: symbolic and sparse representation of H.
    """
    # define the FH Hamiltonian with NN terms
    interactions_t1 = []
    for site_i in np.arange(0, num_sites - 1):
        interactions_t1.append([site_i, site_i + 1])
        # if site_i == 1:
        # elif site_i == num_sites:
    if pbc == 1:
        interactions_t1.append([num_sites - 1, 0])
    interactions_t2 = []
    for site_j in np.arange(0, num_sites - 1):
        if site_j + 2 < num_sites:
            interactions_t2.append([site_j, site_j + 2])
    if pbc == 1:
        interactions_t2.append([num_sites - 2, 0])
        interactions_t2.append([num_sites - 1, 1])
    # print(interactions_t1)
    # print(interactions_t2)
    # interactions_t1 = [[1,2],[2,3],[3,4]]
    # need to choose the chemical potential in order to be at half-filling
    ham = fermi_hubbard_NN_from_edges(
        num_sites, interactions_t1, interactions_t2, t1=t1, t2=t2, U=U, mu=mu
    )
    sH = ham.build_sparse_matrix()
    return ham, sH


def solve_ground_state_half_filling(
    num_sites: int, ham: SparseOperatorBuilder, sH: Any, show: bool = True
) -> Tuple[float, float, Any]:
    """Calculates the ground state of the Fermi-Hubbard Hamiltonian at half filling.

    The half-filling condition is enforced by properly tuning the
    chemical potential.

    Args:
        num_sites (int): number of lattice sites.
        ham (SparseOperatorBuilder): symbolic representation of the Hamiltonian.
        sH (Any): sparse matrix representation of the Hamiltonian.
        show (bool, optional): whether to print info. Defaults to True.

    Returns:
        Tuple[float, float, any]: ground-state energy, average of the
            number of particle operator, eigenvectors of the Hamiltonian
            matrix.
    """

    # SELECT THE GROUND-STATE AT HALF FILLING
    # calculate all the eigenstates of the Hamiltonian
    ek, vk = qu.eigh(sH, k=2 ** (2 * 4), backend="AUTO")
    vk = vk.transpose()
    eigs_num_part = [
        calculate_num_particles(num_sites, ham, np.array([state]).transpose())
        for state in vk
    ]

    # energies are sorted in growing order, so the first state with N/2 particles is the right one
    for i, n in enumerate(eigs_num_part):
        if n == num_sites:
            gs = np.array([vk[i]]).transpose()
            break

    en_gs = qu.expec(sH, gs)
    n_gs = calculate_num_particles(num_sites, ham, gs)

    if show:
        print("energy = ", en_gs)
        print("number of particles = ", n_gs)

    return en_gs, n_gs, gs


def solve_ground_state(
    num_sites: int, ham: SparseOperatorBuilder, sH: Any, show: bool = True
) -> Tuple[float, float, Any]:
    """Exact diagonalization of the FH Hamiltonian.

    Args:
        num_sites (int): number of lattice sites.
        ham (SparseOperatorBuilder): symbolic representation of the Hamiltonian.
        sH (Any): sparse representation of the Hamiltonian.
        show (bool, optional): whether to print the results. Defaults to True.

    Returns:
        Tuple[float, float, any]: ground-state energy, average of the
            number of particle operator, eigenvectors of the Hamiltonian
            matrix.
    """

    en, v = qu.eigh(sH, k=1, backend="AUTO")
    # v = v.transpose()
    # n = calculate_num_particles(num_sites, ham, np.array([v]).transpose())
    n = calculate_num_particles(num_sites, ham, v)
    if show:
        print("energy = ", en[0])
        print("number of particles = ", n)
    return en, n, v


def FH_Hamiltonian_NN_half_filling(
    num_sites: int, t1: float, t2: float, U: float, pbc: int = 0
) -> Tuple[SparseOperatorBuilder, Any]:
    """Calculates the Fermi-Hubbard Hamiltonian with half-filling as ground state.

    The fact that the half-filling configuration is the absolute ground state of the Hamiltonian
    is ensured by properly tuning the chemical potential.

    Args:
        num_sites (int): number of lattice sites.
        t1 (float): nearest-neighbour term.
        t2 (float): next-to-nearest-neighbour term.
        U (float): potential energy term.
        pbc (int, optional): whether to apply periodic boundary conditions.
            Defaults to 0.

    Returns:
        Tuple[SparseOperatorBuilder, Any]: symbolic and sparse representation of the Hamiltonian.
    """

    # find the ground state at half-filling via ED
    print("Reference ED at half-filling:")
    ham_hf, sham_hf = FH_Hamiltonian_NN(num_sites, t1, t2, U, mu=0.0)
    en_hf, _, _ = solve_ground_state_half_filling(num_sites, ham_hf, sham_hf)
    print("\n")
    print("New Hamiltonian at half-filling:")
    mu_hf = None
    if t2 == 0.0:
        mu_hf = -U / 2
    # elif t2==0.3: # previously computed, to speed up testing
    #     mu_hf = -2.288
    else:
        mu_vec = []
        for mu in np.arange(-U, U, 0.001):

            ham_mu, sham_mu = FH_Hamiltonian_NN(
                num_sites, t1=t1, t2=t2, U=U, mu=mu, pbc=pbc
            )
            en_mu, n_mu, _ = solve_ground_state(num_sites, ham_mu, sham_mu, show=False)

            if int(n_mu) == num_sites:  # np.isclose(en_hf, en_mu, atol=1.e-3) and
                mu_vec.append(mu)

        mu_hf = mu_vec[int(len(mu_vec) / 2)]

    if mu_hf is not None:
        print("chemical potential at half-filling = ", np.round(mu_hf, 3))
        ham_mu, sham_mu = FH_Hamiltonian_NN(
            num_sites, t1=t1, t2=t2, U=U, mu=mu_hf, pbc=pbc
        )
    else:
        print("Could not find mu to be at half-filling!")

    return ham_mu, sham_mu
