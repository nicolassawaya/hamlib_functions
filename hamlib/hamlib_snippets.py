# Snippets of code aiding HamLib project

import numpy as np
import networkx as nx
import h5py
import mat2qubit
from openfermion import SymbolicOperator, QubitOperator, FermionOperator
import copy
import re
from qiskit.quantum_info import SparsePauliOp

# The functions below are for "hierarchical" hdf5 files that make use of groups
# and subgroups to divide the datasets


def _parse_hdf5_recursive(func):
    """Decorator that recursively iterates through HDF5 file and performs
    some action that can be specified by `func` on the internal and leaf
    nodes in the file."""
    def wrapper(obj, path='/', key=None):
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for ky in obj.keys():
                func(obj, path, key=ky, leaf=False)
                wrapper(obj=obj[ky], path=path + ky + '/', key=ky)
        elif type(obj) is h5py._hl.dataset.Dataset:
            func(obj, path, key=None, leaf=True)
    return wrapper


def get_hierarchical_hdf5_keys(fname_hdf5):
    """Get list of full path keys in hdf5 file. (Applicable to any
    "hierarchical" HamLib hdf5 file)."""
    all_keys = []

    @_parse_hdf5_recursive
    def action(obj, path='/', key=None, leaf=False):
        if leaf is True:
            all_keys.append(path)

    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])

    return all_keys


def print_hdf5_structure(fname_hdf5):
    """Print the hierarchical structure in the hdf5 file. (Applicable to any
    "hierarchical" HamLib hdf5 file)"""

    @_parse_hdf5_recursive
    def action(obj, path='/', key=None, leaf=False):
        if key is not None:
            print((path.count('/')-1)*'\t', '-', key, ':', path + key + '/')
        if leaf:
            print((path.count('/')-1)*'\t', '[^^DATASET^^]')

    print(f'start of file: {fname_hdf5}\n')
    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])
    print(f'\nend of file: {fname_hdf5}\n')


# The functions below are for "flat" hdf5 files that store all the datasets


def get_hdf5_keys(fname_hdf5):
    """Get list of keys in hdf5 file. (Applicable to any "flat" HamLib hdf5
    file)"""

    with h5py.File(fname_hdf5, 'r') as f:
        keys = list(f.keys())

    return keys


def save_graph_hdf5(G, fname_hdf5, str_key, overwrite=True, grid_pos=None):
    """Save networkx graph to the appropriate hdf5 file.
    Note: This function uses 'a'==APPEND mode. This means that if a particular
            key is already present in the hdf5 file, it will throw an error.
            Hence if you're running the same code a second time, you will need
            to delete the hdf5 file first.
    """
    es = list(G.edges)

    with h5py.File(fname_hdf5, 'a') as f:

        if str_key in f:
            if overwrite:
                del f[str_key]
                f[str_key] = np.array(es)
        else:
            f[str_key] = np.array(es)

        if grid_pos is None:
            pass
        else:
            # Store dict{nodes: grid positions} as attribute of each graph
            for k, v in grid_pos.items():
                f[str_key].attrs[str(k)] = v


def read_graph_hdf5(fname_hdf5, str_key):
    """Read networkx graphs from appropriate hdf5 file.
    Returns a single graph, with specified str_key
    """
    with h5py.File(fname_hdf5, 'r') as f:
        G = nx.Graph(list(np.array(f[str_key])))

    return G


def read_gridpositions_hdf5(fname_hdf5, str_key):
    """Read grid positions, stored as attribute of each networkx graph from
    appropriate hdf5 file.
    Returns grid positions of nodes associated with a single graph, with
    specified str_key
    """
    with h5py.File(fname_hdf5, 'r') as f:
        dataset = f[str_key]
        gridpositions_dict = dict(dataset.attrs.items())

    return gridpositions_dict


def save_openfermion_hdf5(qubop, fname_hdf5, str_key, overwrite=True):
    """Save any openfermion operator object to hdf5 file.
    Intended for QubitOperator and FermionOperator."""
    with h5py.File(fname_hdf5, 'a', libver='latest') as f:

        if str_key in f:
            if overwrite:
                del f[str_key]
                f[str_key] = str(qubop)
        else:
            f[str_key] = str(qubop)


def read_openfermion_hdf5(fname_hdf5, str_key, optype=QubitOperator):
    """Read any openfermion operator object from hdf5 file.
    'optype' is the op class. Can be QubitOperator or FermionOperator.
    """
    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        op = optype(f[str_key][()].decode("utf-8"))

    return op


def read_qiskit_hdf5(fname_hdf5: str, key: str):
    """
    Read the operator object from HDF5 at specified key to qiskit SparsePauliOp
    format.
    """
    def _generate_string(term):
        # change X0 Z3 to XIIZ
        indices = [
            (m.group(1), int(m.group(2)))
            for m in re.finditer(r'([A-Z])(\d+)', term)
        ]
        return ''.join(
            [next((char for char, idx in indices if idx == i), 'I')
             for i in range(max(idx for _, idx in indices) + 1)]
        )

    def _append_ids(pstrings):
        # append Ids to strings
        return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]

    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        pattern = r'([\d.]+) \[([^\]]+)\]'
        matches = re.findall(pattern, f[key][()].decode("utf-8"))

        labels = [_generate_string(m[1]) for m in matches]
        coeffs = [float(match[0]) for match in matches]
        op = SparsePauliOp(_append_ids(labels), coeffs)
    return op


def save_mat2qubit_hdf5(symbop, fname_hdf5, str_key, overwrite=True):
    """Save mat2qubit.qSymbOp operator to hdf5 file"""
    with h5py.File(fname_hdf5, 'a') as f:

        if str_key in f:
            if overwrite:
                del f[str_key]
                f[str_key] = str(symbop)
        else:
            f[str_key] = str(symbop)

    return


def read_mat2qubit_hdf5(fname_hdf5, str_key):
    """Returns mat2qubit.qSymbOp operator from hdf5 file"""
    with h5py.File(fname_hdf5, 'r') as f:
        op = mat2qubit.qSymbOp(f[str_key][()].decode("utf-8"))

    return op


def save_clause_list_hdf5(clause_list, fname_hdf5, str_key):
    """Save clause list to appropriate hdf5 file
    Expects clause list to be in DIMACS format
    """
    dt = h5py.vlen_dtype(np.dtype('int32'))
    with h5py.File(fname_hdf5, 'a') as f:
        dset = f.create_dataset(str_key, (len(clause_list), ), dtype=dt)
        for i, clause in enumerate(clause_list):
            dset[i] = np.array(clause)
        # f[str_key] = [np.array(clause) for clause in clause_list]


def read_clause_list_hdf5(fname_hdf5, str_key):
    """read clause list from appropriate hdf5 file

    Returns clause list in DIMACS format
    """
    clause_list = []
    with h5py.File(fname_hdf5, 'r') as f:
        for clause in list(np.array(f[str_key])):
            clause_list.append([v for v in clause])

    return clause_list


def num_terms(op):
    """Returns number of terms in object"""

    if isinstance(op, QubitOperator):
        return len(op.terms)
    elif isinstance(op, FermionOperator):
        return len(op.terms)
    elif isinstance(op, mat2qubit.qSymbOp):
        return len(op.fullOperator)
    else:
        raise TypeError("Object is not a supported operator type.")


def getPauliStringLengths(qubop):
    """Return list of Pauli string lengths in op"""
    return [len(term) for term in qubop.terms.keys()]


def pweight_distr(qubop):
    """Returns probability weight distribution of terms in openfermion op"""

    if isinstance(qubop, SymbolicOperator):

        # Get histogram of pweights
        pweight_list_all = getPauliStringLengths(qubop)
        bins = np.arange(0, max(pweight_list_all)+2, 1, dtype=int)
        hist, bin_edges = np.histogram(
            pweight_list_all, bins=bins, density=False
        )
        return dict(zip(bin_edges, hist))

    else:
        raise TypeError(
            f"Type {type(qubop)} is not a supported operator type."
        )


def remove_qindices(op, inds_to_remove):
    """Removes indices (i.e. removes qubits or fermionic modes) from operator.
    Any term containing the given indices are removed.

    Args:
        op (inhereting from SymbolicOperator): Operator with indices to be
        removed

    Returns:
        new_op (inhereting from SymbolicOperator): New operator with indices
        removed
    """

    assert isinstance(op, SymbolicOperator)
    # assert inds_to_remove is iterable
    assert hasattr(inds_to_remove, '__iter__'), "Input is not iterable"

    new_op = copy.deepcopy(op)

    for term in op.terms:
        for factor in term:
            if factor[0] in inds_to_remove:
                new_op.terms.pop(term)
                break

    return new_op


def filter_dict():
    raise NotImplementedError()
    """
    key_to_object
    # key_to_dict = lambda key: dict(
    # [ a.split('-') for a in key.split('_')[1:] ])
    subdict = {k:v for k,v in key_to_object.items() if v['Lx']=='4'}
    """


def process_keystring(inpkey):
    """Process keystring into dictionary

    HamLib format is:
    {problemclass}_{varname1}-{var1}_{varnam2}-{var2}…,
        where ‘problem class’ may include hyphens, but ‘varX’ may not.
    """

    key_to_dict = {}
    pieces = inpkey.split('_')
    key_to_dict['probclass'] = pieces[0]
    for piece in pieces[1:]:
        spl = piece.split('-')
        val = spl[1]
        if val.isnumeric():
            key_to_dict[spl[0]] = float(val)
        else:
            key_to_dict[spl[0]] = val

    return key_to_dict


def convert_to_dimacs(clause_list, clause_signs):
    dimacs_list = []
    for clause, signs in zip(clause_list, clause_signs):
        dimacs_list.append(
            tuple(
                [var + 1 if sign == 0 else -(var + 1)
                 for var, sign in zip(clause, signs)]
            )
        )

    return dimacs_list


def convert_to_hamlib(dimacs_list):
    clause_list = []
    clause_signs = []
    for d_clause in dimacs_list:
        clause = []
        signs = []
        for var in d_clause:
            clause.append(abs(var) - 1)
            signs.append(0 if var > 0 else 1)
        clause_list.append(clause)
        clause_signs.append(signs)

    return clause_list, clause_signs


def remove_smaller_values(op, thresh):
    """Removes absolute values below given threshold"""

    assert isinstance(op, SymbolicOperator)

    new_op = copy.deepcopy(op)

    for term in op.terms:
        if abs(op.terms[term]) < thresh:
            new_op.terms.pop(term)

    return new_op
