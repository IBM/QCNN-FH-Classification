[![License](https://img.shields.io/github/license/qiskit-community/quantum-prototype-template?label=License)](./LICENSE.txt)

# MERA-Hubbard

This code relies on the `quimb` library for the implementation of the MERA of the Fermi-Hubbard model with NN terms, and on `qiskit` for translating the optimized structure to a quantum circuit and implementing the respective QCNN.

The following notebooks show the capability of the code:

`hamiltonian.ipynb`:

- Check that, given an arbitrary FH Hamiltonian (with or without NN terms), it has the same energy and number of particles as obtained from qiskit.

`dmrg_energy.ipynb`:

- Check consistency on the energy obtained via DMRG implementation in `quimb`.

`optimization_mera_hubbard.ipynb`:

- Creates a randomly initialized MERA and optimizes it with respect to the energy of the chosen FH Hamiltonian
- Optimizes it.
- Saves the optimized MERA in a file.

`conversion_tensors_matrices.ipynb`:

- To initialize gates in a quantum circuit we need matrices, not tensors, so the tensors which compose the MERA need to be reshaped first. The idea would be to do it before saving the MERA in a file and import it into qiskit, indeed quimb has the function `.fuse()` which allows to contract two indices. This notebook is then used to test this pipeline by using only MERA with small bond dimension (only max_bond = 2), so they are not necessarily optimized to give the correct energy. However, for 1x2 lattices usually the MERA gives already the correct energy and it is possible to perform a check on the fidelity of the state
- save the new matrices in a file: layer by layer, first the unitaries of one layer, then the isometries of that layer

`initialization_qiskit_circuit.ipynb`:

- read the saved file and implement the circuit
- to check the conversion is correct the idea is to compare the energy obtained from the final state of the qiskit circuit $E = \langle\psi\|H\|\psi\rangle$ to the eigenvalue of the Hamiltonian $H$ implemented in qiskit. However the qiskit implementation uses the class `FermionicOp` which makes the computation more difficult
- we start then by checking the fidelity of the two states (the one obtained at the end of the circuit and the eigenstate)

## WARNING:

- The code does not support not-cyclic MERA in the library (the attribute *cyclic* is there but not used... fixable with one line in `tensor_mera.py`). However, after some tests it seems that cyclic MERA are easier to optimize even for Hamiltonian with open bc

## NEXT STEPS:

- Find the problem in the conversion of the tensors to matrices (more precisely completing the matrices of the isometries to make them unitary) to obtain then fidelity equal to 1 (now it is around 0.3). A first test is to check the correct conversion of the unitaries and the indices convention of quimb is to initialize a MERA with random unitaries and isometries equal to identities, DO NOT optimize it but calculate the energy it gives wrt the hamiltonian, save it and convert it into qiskit, calculate the energy with the same hamiltonian in qiskit. The problem in the implementation here is that the qiskit ham is a FermionicOp and it needs to be a generic Operator to calculate the expectation value with a Statevector. The old module `qiskit.chemistry` contained FermionicTransformation (https://docs.quantum.ibm.com/api/qiskit/0.28/qiskit.chemistry.transformations.FermionicTransformation) which allowed to do this. However, I don't know where it has been moved in the new qiskit version.
- Understand how to treat/convert tensors with higher bond dimension than 2 to matrices to define quantum gates
- Reverse the circuit to obtain a simil-QCNN structure
- Add QEC gated before or after the isometries to allow for correct phase recognition (maybe add parameterized gates here and shortly train those)
- Try QPR on the metallic phase

## License

This project uses the [Apache License 2.0](https://github.ibm.com/Alberto-Baiardi/MERA-Hubbard/blob/main/LICENSE.md).
