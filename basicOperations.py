import tensornetwork as tn
import tensornetwork.backends.base_backend as be
import numpy as np
import math
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union, \
    Sequence, Iterable, Type

def getStartupState(n):
    psi = [None] * n
    baseLeftTensor = np.zeros((1, 2, 2))
    baseLeftTensor[0, 1, 0] = -1
    baseLeftTensor[0, 0, 1] = 1
    psi[0] = tn.Node(baseLeftTensor, name='site0', axis_names=['v0', 's0', 'v1'], \
                backend = None)
    baseMiddleTensor = np.zeros((2, 2, 2))
    baseMiddleTensor[0, 1, 0] = -1
    baseMiddleTensor[1, 0, 1] = 1
    baseMiddleTensor[1, 1, 0] = -1
    baseMiddleTensor[0, 0, 1] = 1
    for i in range(1, n-1):
        psi[i] = tn.Node(baseMiddleTensor, name=('site' + str(i)), \
                               axis_names=['v' + str(i), 's' + str(i), 'v' + str(i+1)], \
                               backend = None)
    baseRightTensor = np.zeros((2, 2, 1))
    baseRightTensor[0, 1, 0] = -1 / math.sqrt(2)
    baseRightTensor[1, 0, 0] = 1 / math.sqrt(2)
    psi[n - 1] = tn.Node(baseRightTensor, name=('site' + str(n - 1)), \
                               axis_names=['v' + str(n - 1), 's' + str(n - 1), 'v' + str(n)], \
                               backend = None)
    norm = getOverlap(psi, psi)
    psi[0] = multNode(psi[0], 1/math.sqrt(norm))
    return psi

# Assuming psi1, psi2 have the same length, Hilbert space etc.
# assuming psi2 is conjugated
def getOverlap(psi1Orig: List[tn.Node], psi2Orig: List[tn.Node]):
    psi1 = copyState(psi1Orig)
    psi2 = copyState(psi2Orig, conj=True)
    psi1[0][0] ^ psi2[0][0]
    psi1[0][1] ^ psi2[0][1]
    contracted = tn.contract_between(psi1[0], psi2[0], name='contracted')
    for i in range(1, len(psi1) - 1):
        psi1[i][1] ^ psi2[i][1]
        contracted[0] ^ psi1[i][0]
        contracted[1] ^ psi2[i][0]
        contracted = tn.contract_between(tn.contract_between(contracted, psi1[i]), psi2[i])
    psi1[len(psi1) - 1][1] ^ psi2[len(psi1) - 1][1]
    psi1[len(psi1) - 1][2] ^ psi2[len(psi1) - 1][2]
    contracted[0] ^ psi1[len(psi1) - 1][0]
    contracted[1] ^ psi2[len(psi1) - 1][0]
    contracted = tn.contract_between(tn.contract_between(contracted, psi1[len(psi1) - 1]), psi2[len(psi1) - 1])

    result = contracted.tensor
    tn.remove_node(contracted)
    removeState(psi1)
    removeState(psi2)
    return result

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


def multiContraction(node1: tn.Node, node2: tn.Node, edges1, edges2, nodeName=None):
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


def permute(node: tn.Node, permutation):
    if node is None:
        return None
    axisNames = []
    for i in range(len(permutation)):
        axisNames.append(node.edges[permutation[i]].name)
    result = tn.transpose(node, permutation, axis_names=axisNames)
    result.add_axis_names(axisNames)
    for i in range(len(axisNames)):
        result.get_edge(i).set_name(axisNames[i])
    result.set_name(node.name)
    return result


def svdTruncation(node: tn.Node, leftEdges: List[tn.Edge], rightEdges: List[tn.Edge], \
                  dir: str, maxBondDim=1024, leftName='U', rightName='V',  edgeName=None):
    maxBondDim = getAppropriateMaxBondDim(maxBondDim, leftEdges, rightEdges)
    if dir == '>>':
        leftEdgeName = edgeName
        rightEdgeName = None
    else:
        leftEdgeName = None
        rightEdgeName = edgeName

    [U, S, V, truncErr] = tn.split_node_full_svd(node, leftEdges, rightEdges, max_singular_values=maxBondDim, \
                                       left_name=leftName, right_name=rightName, \
                                       left_edge_name=leftEdgeName, right_edge_name=rightEdgeName)
    if dir == '>>':
        l = copyState([U])[0]
        r = copyState([tn.contract_between(S, V, name=V.name)])[0]
    else:
        l = copyState([tn.contract_between(U, S, name=U.name)])[0]
        r = copyState([V])[0]
    tn.remove_node(U)
    tn.remove_node(S)
    tn.remove_node(V)
    return [l, r, truncErr]


# Apparently the truncation method doesn'tlike it if max_singular_values is larger than the size of S.
def getAppropriateMaxBondDim(maxBondDim, leftEdges, rightEdges):
    uDim = 1
    for e in leftEdges:
        uDim *= e.dimension
    vDim = 1
    for e in rightEdges:
        vDim *= e.dimension
    if maxBondDim > min(uDim, vDim):
        return min(uDim, vDim)
    else:
        return maxBondDim


# Split M into 2 3-rank tensors for sites k, k+1
def assignNewSiteTensors(psi, k, M, dir, getOrthogonal=False):
    [sitek, sitekPlus1, truncErr] = svdTruncation(M, [M[0], M[1]], [M[2], M[3]], dir, \
            leftName=('site' + str(k)), rightName=('site' + str(k+1)), edgeName = ('v' + str(k+1)))
    tn.remove_node(psi[k])
    psi[k] = sitek
    # if k > 0:
    #     psi[k-1][2] ^ psi[k]
    tn.remove_node(psi[k+1])
    psi[k+1] = sitekPlus1
    # if k+2 < len(psi):
    #     psi[k+1][2] ^ psi[k+2][0]
    return [psi, truncErr]


def getEdgeNames(node: tn.Node):
    result = []
    for edge in node.edges:
        result.append(edge.name)
    return result


# k is curr working site, shift it by one in dir direction.
def shiftWorkingSite(psi: List[tn.Node], k, dir):
    if dir == '<<':
        pair = tn.contract(psi[k-1][2] ^ psi[k][0], axis_names=getEdgeNames(psi[k-1])[:2] + getEdgeNames(psi[k])[1:])
        [psi, truncErr] = assignNewSiteTensors(psi, k-1, pair, dir)
    else:
        pair = tn.contract(psi[k][2] ^ psi[k+1][0], axis_names=getEdgeNames(psi[k])[:2] + getEdgeNames(psi[k+1])[1:])
        [psi, truncErr] = assignNewSiteTensors(psi, k, pair, dir)
    return psi


def removeState(psi):
    for i in range(len(psi)):
        tn.remove_node(psi[i])


def addStates(psi1: List[tn.Node], psi2: List[tn.Node]):
    result = copyState(psi1)
    resultTensor = np.zeros((1, psi1[0].shape[1], psi1[0].shape[2] + psi2[0].shape[2]))
    resultTensor[0, :, :psi1[0].shape[2]] = psi1[0].tensor
    resultTensor[0, :, psi1[0].shape[2]:] = psi2[0].tensor
    result[0].set_tensor(resultTensor)
    for i in range(1, len(psi1)-1):
        resultTensor = \
            np.zeros((psi1[i].shape[0] + psi2[i].shape[0], psi1[i].shape[1], psi1[i].shape[2] + psi2[i].shape[2]))
        resultTensor[:psi1[i].shape[0], :, :psi1[i].shape[2]] = psi1[i].tensor
        resultTensor[psi1[i].shape[0]:, :, psi1[i].shape[2]:] = psi2[i].tensor
        result[i].set_tensor(resultTensor)
    resultTensor = np.zeros((psi1[len(psi1)-1].shape[0] + psi2[len(psi1)-1].shape[0], psi1[len(psi1)-1].shape[1], 1))
    resultTensor[:psi1[len(psi1)-1].shape[0], :, :] = psi1[len(psi1)-1].tensor
    resultTensor[psi1[len(psi1)-1].shape[0]:, :, :] = psi2[len(psi1)-1].tensor
    result[len(psi1)-1].set_tensor(resultTensor)
    return result


def getOrthogonalState(psi: List[tn.Node], psiInitial=None):
    psiCopy = copyState(psi)
    if psiInitial is None:
        psiInitial = getStartupState(len(psi))
    overlap = getOverlap(psiCopy, psiInitial)
    psiCopy[0] = multNode(psiCopy[0], -overlap)
    result = addStates(psiInitial, psiCopy)
    result[0] = multNode(result[0], 1/math.sqrt(getOverlap(result, result)))
    removeState(psiCopy)
    return result
