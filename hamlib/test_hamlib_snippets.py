# Unit tests for hamlib_snippets
# 
# USAGE: pytest -s test_hamlib_snippets.py

from . import hamlib_snippets
import pytest
import networkx as nwx

from openfermion import QubitOperator, FermionOperator, count_qubits
from qiskit.quantum_info import SparsePauliOp
#from mat2qubit import qSymbOp

import os

fname = "test_test_hamlib_snip.hdf5"


def test_io_hierarchical_hdf5():
    # Define
    op1 = QubitOperator("1 [] + 2 [X0 X2] + 3 [Z0] + 4 [Y1]")
    op2 = QubitOperator("2 [] + 4 [X0 X2] + 6 [Z0] + 8 [Y1]")

    # Save at hierarchical paths
    strkey1 = "/toplevel/midlevel/lowlevel/op1"
    hamlib_snippets.save_openfermion_hdf5(op1, fname, strkey1)
    strkey2 = "/toplevel/midlevel/op2"
    hamlib_snippets.save_openfermion_hdf5(op2, fname, strkey2)

    # # Print structure of file keys
    # hamlib_snippets.print_hierarchical_hdf5_structure(fname)

    # Get flat list of keys
    keys = hamlib_snippets.get_hierarchical_hdf5_keys(fname)

    # Load
    op_loaded1 = hamlib_snippets.read_openfermion_hdf5(fname, keys[0])
    op_loaded2 = hamlib_snippets.read_openfermion_hdf5(fname, keys[1])

    # Compare
    assert len(keys) == 2
    assert op1 == op_loaded1
    assert op2 == op_loaded2

    # Delete file
    os.remove(fname)


def test_io_qubitoperator():

    # Define
    op = QubitOperator("1 [] + 2 [X0 X2] + 3 [Z0] + 4 [Y1]")

    # Save
    strkey = "testkey"
    hamlib_snippets.save_openfermion_hdf5(op,fname,strkey)
    # Save again to test duplicate-saving
    hamlib_snippets.save_openfermion_hdf5(op,fname,strkey)

    # Load
    op_loaded = hamlib_snippets.read_openfermion_hdf5(fname,strkey)

    # Compare
    assert op == op_loaded

    # Load as Qiskit SparsePauliOp
    op_qiskit = hamlib_snippets.read_qiskit_hdf5(fname, strkey)
    assert isinstance(op_qiskit, SparsePauliOp)


    # Delete file
    os.remove(fname)


def test_io_fermionoperator():

    # Define
    op = FermionOperator("1 [] + 2 [0^ 2] + 3 [0^ 0] + 4 [4^ 3 9 3^]")

    # Save
    strkey = "testkey"
    hamlib_snippets.save_openfermion_hdf5(op,fname,strkey)
    # Save again to test duplicate-saving
    hamlib_snippets.save_openfermion_hdf5(op,fname,strkey)

    # Load
    operator_class = FermionOperator
    op_loaded = hamlib_snippets.read_openfermion_hdf5(fname,strkey,operator_class)

    # Compare
    assert op == op_loaded

    # Delete file
    os.remove(fname)


"""
def test_io_mat2qubit():
    
    # Define
    op = qSymbOp(" k [ n_A ] ++ j [ p_B ] ++ r [] ")
    # Save
    strkey = "testkey"
    hamlib_snippets.save_mat2qubit_hdf5(op,fname,strkey)
    # Save again to test duplicate-saving
    hamlib_snippets.save_mat2qubit_hdf5(op,fname,strkey)
    # Load
    op_loaded = hamlib_snippets.read_mat2qubit_hdf5(fname,strkey)
    # Compare
    assert str(op) == str(op_loaded)
    # Delete file
    os.remove(fname)
"""



def test_io_graph():

    # create graph G 
    G_grid = nwx.grid_graph(dim=(3,))
    G = nwx.convert_node_labels_to_integers(G_grid)

    # save graph G 
    strkey = "testkey"
    hamlib_snippets.save_graph_hdf5(G, fname, strkey)

    # get the keys and confirm correct
    keys = hamlib_snippets.get_hdf5_keys(fname)
    assert len(keys) == 1
    assert strkey == keys[0]

    # read graph and confirm correct 
    G_out = hamlib_snippets.read_graph_hdf5(fname, strkey)
    assert nwx.is_isomorphic(G_out, G)

    # attempt to read grid_pos from hdf5 file and confirm there is no grid_pos attribute
    grid_pos_out = hamlib_snippets.read_gridpositions_hdf5(fname, strkey)
    assert grid_pos_out == {}

    # delete file
    os.remove(fname)
    
    
    
def test_io_graph_wgridpos():

    # create graph G and get grid positions grid_pos
    G_grid = nwx.grid_graph(dim=(3,))
    G = nwx.convert_node_labels_to_integers(G_grid)
    nodes = list(G.nodes)
    gridnodes = list(G_grid.nodes)
    grid_pos = {str(nodes[i]):gridnodes[i] for i in range(0,3)}

    # save graph G and save grid_pos dict as attribute
    strkey = "testkey"
    hamlib_snippets.save_graph_hdf5(G, fname, strkey, grid_pos=grid_pos)

    # get the keys and confirm correct
    keys = hamlib_snippets.get_hdf5_keys(fname)
    assert len(keys) == 1
    assert strkey == keys[0]

    # read graph and confirm correct 
    G_out = hamlib_snippets.read_graph_hdf5(fname, strkey)
    assert nwx.is_isomorphic(G_out, G)

    # read grid_pos and confirm correct
    grid_pos_out = hamlib_snippets.read_gridpositions_hdf5(fname, strkey)
    assert grid_pos_out == grid_pos

    # delete files
    os.remove(fname)

    
    
def test_cleanup():
    # Clean up--will be required if any of above tests fail
    if os.path.exists(fname):
        os.remove(fname)


def test_num_terms():
    # Test number of terms
    op = QubitOperator("1 [] + 2 [X0 X2] + 3 [Z0] + 4 [Y1]")
    assert hamlib_snippets.num_terms(op) == 4

    op = FermionOperator("1 [] + 2 [0^ 2] + 3 [0^ 0] + 4 [4^ 3 9 3^]")
    assert hamlib_snippets.num_terms(op) == 4

    #op = qSymbOp(" k [ n_A ] ++ j [ p_B ] ++ r [] ")
    #assert hamlib_snippets.num_terms(op) == 3


def test_pweight_distr():
    # Test pweight_distr
    op = QubitOperator("1 [] + 2 [X0 X2] + 3 [Z0] + 4 [Y1]")
    assert hamlib_snippets.pweight_distr(op) == {0: 1, 2: 1, 1: 2}


def test_process_keystring():
    # Test process_keystring()
    keystr = "graph-2D-triag-pbc-qubitnodes_Lx-4_Ly-198"
    gold = {'probclass': 'graph-2D-triag-pbc-qubitnodes', 'Lx': 4, 'Ly': 198}
    res = hamlib_snippets.process_keystring(keystr)
    assert res == gold


def test_clause_list():

    # Delete file if it exists
    if os.path.exists("test_clause_list_snippets.hdf5"):
        os.remove("test_clause_list_snippets.hdf5")
        
    # Example clause list with sign list
    clause_list = [
            (1,2,3),
            (2,4,5),
            (1,3,6),
            (2,3,4)
            ]

    sign_list = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            ]

    # Save clause+sign list in dimacs form
    dimacs_format = hamlib_snippets.convert_to_dimacs(clause_list, sign_list)
    hamlib_snippets.save_clause_list_hdf5(dimacs_format, "test_clause_list_snippets.hdf5", "example_clause")

    # Read in clause list (dimacs form)
    test_clause_list = hamlib_snippets.read_clause_list_hdf5("test_clause_list_snippets.hdf5", "example_clause")

    # Convert to sets for easy checking
    base_clause_set = set([tuple(c) for c in dimacs_format])
    test_clause_set = set([tuple(c) for c in test_clause_list])

    # Check if the clause lists are the same
    assert base_clause_set == test_clause_set

    # Delete file
    os.remove("test_clause_list_snippets.hdf5")


def test_convert_to_dimacs():
    # Example clause list with sign list
    clause_list = [
            (1,2,3),
            (2,4,5),
            (1,3,6),
            (2,3,4)
            ]

    sign_list = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            ]

    # Save clause+sign list in dimacs form
    dimacs_format = hamlib_snippets.convert_to_dimacs(clause_list, sign_list)

    expected = set()
    # Note: these clauses are in a different order from the original
    expected.add((2, 3, -4))
    expected.add((3, -5, 6))
    expected.add((-2, 4, -7))
    expected.add((3, -4, -5))

    dimacs_set = set([c for c in dimacs_format])

    assert expected == dimacs_set


def test_convert_from_dimacs():
    dimacs = [
        (1, 2, -3),
        (2, -4, 5),
        (-1, 3, -6),
        (2, -3, -4),
        ]
    
    # Note: these clauses are in a different order from the original
    expected_clause_set = set([
        (0, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
        (0, 2, 5)
        ])

    expected_sign_set = set([
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            ])

    converted_clause_list, converted_sign_list = hamlib_snippets.convert_to_hamlib(dimacs)

    converted_clause_set = set([tuple(c) for c in converted_clause_list])
    converted_sign_set = set([tuple(s) for s in converted_sign_list])

    assert expected_clause_set == converted_clause_set
    assert expected_sign_set == converted_sign_set


def test_remove_qindices():

    qop  = QubitOperator("-1 [] + -2 [X0 X2] + -3 [Z0] + -4 [Y1]")
    res  = hamlib_snippets.remove_qindices(qop,(2,))
    gold = QubitOperator("-1 [] + -3 [Z0] + -4 [Y1]")
    assert gold==res, (gold,res)
    assert 2 == count_qubits(gold)

    res = hamlib_snippets.remove_qindices(qop,(1,))
    gold = QubitOperator("-1 [] + -2 [X0 X2] + -3 [Z0]")
    assert gold==res, (gold,res)
    assert 3 == count_qubits(gold)




def test_remove_smaller_values():
    # Test remove_smaller_values
    op = QubitOperator("-1 [] + -2 [X0 X2] + -3 [Z0] + -4 [Y1]")
    op2 = hamlib_snippets.remove_smaller_values(op, 3)
    assert op2 == QubitOperator("-3 [Z0] + -4 [Y1]")








