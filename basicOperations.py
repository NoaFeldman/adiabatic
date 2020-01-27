import tensornetwork as tn
import tensornetwork.backends.base_backend as be
import numpy as np
import math

def getStartupState(n):
    psi = [None] * n
    baseLeftTensor = np.zeros((1, 2, 2))
    baseLeftTensor[0, 0, 0] = 1
    psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['vL0', 's0', 'vR1*'], \
                backend = None)
    baseMiddleTensorOdd = np.zeros((2, 2, 2))
    baseMiddleTensorOdd[0, 1, 0] = 1
    baseMiddleTensorEven = np.zeros((2, 2, 2))
    baseMiddleTensorEven[0, 0, 0] = 1
    for i in range(1, int(n/2)):
        psi[2*i - 1] = tn.Node(baseMiddleTensorOdd, name=('site' + str(i + 1)), \
                               axis_names=['vL' + str(i + 1), 's' + str(i + 1), 'vR' + str(i+2) + '*'], \
                               backend = None)
        psi[2*i] = tn.Node(baseMiddleTensorEven, name=('site' + str(i + 2)), \
                             axis_names=['vL' + str(i + 2), 's' + str(i + 2), 'vR' + str(i + 3) + '*'], \
                             backend = None)
    baseRightTensor = np.zeros((2, 2, 1))
    baseRightTensor[0, 1, 0] = 1
    psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)), \
                               axis_names=['vL' + str(n - 1), 's' + str(n - 1), 'vR' + str(n)], \
                               backend = None)
    return psi

# Assuming psi1, psi2 have the same length, Hilbert space etc.
# assuming psi2 is conjugated
def getOverlap(psi1, psi2):
    psi1[0][0] ^ psi2[0][0]
    psi1[0][1] ^ psi2[0][1]
    contracted = tn.contract_between(psi1[0], psi2[0])
    for i in range(1, len(psi1) - 1):
        contracted = tn.contract(contracted[0] ^ psi1[i][0])
        contracted[0] ^ psi2[i][0]
        contracted[1] ^ psi2[i][1]
        contracted = tn.contract_between(contracted, psi2[i])
    contracted = tn.contract(contracted[0] ^ psi1[len(psi1) - 1][0])
    contracted[0] ^ psi2[len(psi1) - 1][0]
    contracted[1] ^ psi2[len(psi1) - 1][1]
    contracted[2] ^ psi2[len(psi1) - 1][2]
    contracted = tn.contract_between(contracted, psi2[len(psi1) - 1])
    return contracted.tensor

def printNode(node):
    if node == None:
        print('None')
        return
    print('node ' + node.name + ':')
    edgesNames = ''
    for edge in node.edges:
        edgesNames += edge.name + ', '
    print(edgesNames)
    print(node.tensor.shape)


def copyState(psi, conj=False):
    result = list(tn.copy(psi, conjugate=conj)[0].values())
    if conj:
        for node in result:
            for edge in node.edges:
                if edge.name[len(edge.name) - 1] == '*':
                    edge.name = edge.name[0:len(edge.name) - 1]
                else:
                    edge.name = edge.name + '*'
    return result

def addNodes(node1, node2):
    # TODO asserts
    if node1 is None:
        if node2 is None:
            return None
        else:
            return node2
    else:
        if node2 is None:
            return node1
        else:
            result = copyState([node1])[0]
            result.set_tensor(result.get_tensor() + node2.get_tensor())
            return result

def multNode(node, c):
    result = copyState([node])[0]
    result.set_tensor(result.get_tensor() * c)
    return result


def getNodeNorm(node):
    copy = copyState([node])[0]
    copyConj = copyState([node], conj=True)[0]
    for i in range(node.get_rank()):
        copy[i] ^ copyConj[i]
    return math.sqrt(tn.contract_between(copy, copyConj).get_tensor())


def multiContraction(node1, node2, edges1, edges2, nodeName=None):
    if node1 is None or node2 is None:
        return None
    if edges1[len(edges1) - 1] == '*':
        copy1 = copyState([node1], conj=True)[0]
        edges1 = edges1[0:len(edges1) - 1]
    else:
        copy1 = copyState([node1])[0]
    if edges2[len(edges2) - 1] == '*':
        copy2 = copyState([node2], conj=True)[0]
        edges2 = edges2[0:len(edges2) - 1]
    else:
        copy2 = copyState([node2])[0]
    for i in range(len(edges1)):
        copy1[int(edges1[i])] ^ copy2[int(edges2[i])]
    return tn.contract_between(copy1, copy2, name=nodeName)


def permute(node, permutation):
    if node is None:
        return None
    axisNames = []
    for i in range(len(permutation)):
        axisNames.append(node.edges[permutation[i]].name)
    result = tn.transpose(node, permutation, axis_names=axisNames)
    result.add_axis_names(axisNames)
    for i in range(len(axisNames)):
        result.get_edge(i).set_name(axisNames[i])
    return result

