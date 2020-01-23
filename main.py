import tensornetwork as tn
import numpy as np

a = tn.Node(np.identity(2), name='a')
hadamard_op = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
hadamard_node = tn.Node(hadamard_op)
e = a.get_edge(0) ^ hadamard_node.get_edge(1)
a = tn.contract(e)
print(a.get_edge(0).get_nodes())
print(a.get_edge(1).get_nodes())
# c = tn.outer_product(a, a)
# print(c.get_tensor())