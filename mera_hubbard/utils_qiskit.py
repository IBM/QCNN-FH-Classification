# (C) Copyright IBM 2017, 2020, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions needed to implement the qiskit solution of FH Hamiltonian
"""


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Any

import qiskit_nature

qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
    SquareLattice,
    Lattice,
    LatticeDrawStyle,
)
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel, LatticeModel
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.properties import ParticleNumber, Magnetization

from qiskit_nature.second_q.operators import FermionicOp

from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver


def create_model_qiskit(
    lattice_type: Tuple[int],
    t1: float,
    U: float,
    mu: float,
    t2: float,
    draw: bool = False,
) -> Tuple[FermiHubbardModel, FermionicOp]:
    """Generates the Fermi-Hubbard model in qiskit format

    Args:
        lattice_type (Tuple[int]): number of sites per dimension.
        t1 (float): nearest-neighbour scaling factor.
        U (float): potential energy scaling factor.
        mu (float): chemical potential/
        t2 (float): next-to-nearest-neighbour scaling factor
        draw (bool, optional): whether to print results.
            Defaults to False.

    Returns:
        Tuple[FermiHubbardModel, FermionicOp]: qiskit model and operator
            associated with the Fermi-Hubbard Hamiltonian.
    """
    positions = None

    # I add here the minus sign for the hopping terms!!!
    t1 *= -1.0
    t2 *= -1.0

    model_dim = len(lattice_type)
    if model_dim == 1:
        num_sites_per_col = 1
        num_sites_per_row = lattice_type[0]
    else:
        num_sites_per_col = lattice_type[0]
        num_sites_per_row = lattice_type[1]
    num_sites = num_sites_per_col * num_sites_per_row

    # 1D line lattice
    if model_dim == 1:
        figsize = (num_sites_per_row, 1)

        if t2 == 0.0:
            lattice = LineLattice(
                num_sites_per_row, boundary_condition=BoundaryCondition.OPEN
            )
            model = FermiHubbardModel(lattice.uniform_parameters(t1, mu), U)
        else:
            # hopping terms
            adj_matrix = LineLattice(
                num_sites_per_row, boundary_condition=BoundaryCondition.OPEN
            ).to_adjacency_matrix()
            adj_matrix *= t1
            # on-site
            for i in np.arange(num_sites_per_row):
                adj_matrix[i][i] = mu
            # add hopping terms
            for i in np.arange(num_sites_per_row - 2):
                adj_matrix[i][2 + i] = t2
                adj_matrix[2 + i][i] = t2

            lattice = Lattice.from_adjacency_matrix(interaction_matrix=adj_matrix)
            model = FermiHubbardModel(lattice, onsite_interaction=U)
            positions = []
            for i in np.arange(num_sites_per_row):
                positions.append([i, -(i % 2) / 5.0])

    # 2D lattice
    elif model_dim == 2:
        figsize = (num_sites_per_row, 1.3)
        # Attention! rows and cols are switched!!!
        if t2 == 0.0:
            lattice = SquareLattice(
                rows=num_sites_per_row,
                cols=num_sites_per_col,
                boundary_condition=BoundaryCondition.OPEN,
            )
            model = FermiHubbardModel(lattice.uniform_parameters(t1, mu), U)

        # 2D square lattice with next-nearest-neighbors hopping
        else:
            adj_matrix = SquareLattice(
                rows=num_sites_per_row,
                cols=num_sites_per_col,
                boundary_condition=BoundaryCondition.OPEN,
            ).to_adjacency_matrix()
            # first hopping terms
            adj_matrix *= t1
            # on-site potential mu
            for i in np.arange(num_sites):
                adj_matrix[i][i] = mu
            # second hopping terms
            half_ind = int(num_sites / 2)
            for i in np.arange(int(num_sites / 2 - 1)):
                adj_matrix[half_ind + i][i + 1] = t2
                adj_matrix[half_ind + i + 1][i] = t2
                adj_matrix[i][half_ind + i + 1] = t2
                adj_matrix[i + 1][half_ind + i] = t2

            lattice = Lattice.from_adjacency_matrix(interaction_matrix=adj_matrix)
            model = FermiHubbardModel(lattice, onsite_interaction=U)
            # if draw:
            #     print(adj_matrix)
            #     print(lattice.weighted_edge_list)

            positions = []
            for j in np.arange(num_sites_per_col):
                for i in np.arange(num_sites_per_row):
                    positions.append([i, j])
            # print(positions)

    # define the hamiltonian
    ham = model.second_q_op().simplify()

    # draw
    if draw:
        print("Adjacency matrix:")
        print(lattice.to_adjacency_matrix())  # weighted=True
        print("\n")
        plt.figure(figsize=figsize)
        plt.tight_layout()
        if positions:
            lattice.draw(
                style=LatticeDrawStyle(pos=positions, with_labels=True)
            )  # , self_loop=True)
        else:
            lattice.draw(style=LatticeDrawStyle(with_labels=True))
        print("Hamiltonian:")
        print(ham)

    return model, ham


def solve_ground_state_qiskit(
    model: LatticeModel, print_res: bool = False
) -> Tuple[float, float, Any]:
    """Find the ground state at half-filling via Exact Diagonalization.

    Always returns eigenstate as a vector, needs to be converted into a Statevector after if needed.

    Args:
        model (LatticeModel): representation of the Hamiltonian.
        print_res (bool, optional): whether to print results. Defaults to False.

    Returns:
        Tuple[float, float, Any]: energy, particle number, eigenvector.
    """
    num_sites = model.lattice.num_nodes
    lmp = LatticeModelProblem(model)
    lmp.properties.add(ParticleNumber(num_sites))
    # lmp.properties.add(AngularMomentum(num_sites))
    lmp.properties.add(Magnetization(num_sites))
    qubit_mapper = JordanWignerMapper()
    #### solution ground state

    def filter_criterion(eigenstate, eigenvalue, aux_values):
        return np.isclose(aux_values["ParticleNumber"][0], num_sites)

    numpy_solver_gs = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
    solver_gs = GroundStateEigensolver(qubit_mapper, numpy_solver_gs)
    res_gs = solver_gs.solve(lmp)
    exact_gs = res_gs.groundenergy
    num_part_gs = np.round(res_gs.aux_operators_evaluated[0]["ParticleNumber"].real, 0)

    if print_res:
        print("Energy gs = ", res_gs.groundenergy)
        print(
            "Number of particles =",
            np.round(res_gs.aux_operators_evaluated[0]["ParticleNumber"].real, 0),
        )
        # print('Angular Momentum =', res_gs.aux_operators_evaluated[0]['AngularMomentum'])
        print("Magnetization =", res_gs.aux_operators_evaluated[0]["Magnetization"])
        # print(res_gs.groundstate[0]) # this is a quantum circuit!
        # res_gs.groundstate[0].draw()
        # res_gs.raw_result.eigenstate # print the eigenstate as a StateVector

    return exact_gs, num_part_gs, res_gs.raw_result.eigenstate  # res_gs.groundstate[0]
