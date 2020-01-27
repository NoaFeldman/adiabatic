import tensornetwork as tn
import numpy as np
import basicOperations as bops
import math

class HOp:
    def __init__(self, singles, r2l, l2r):
        self.singles = singles
        self.r2l = r2l
        self.l2r = l2r

# onsightTerm as a 2*2 matrix 0, 1
# neighborTerm as a 4*4 matrix 00, 01, 10, 11
def getDMRGH(N, onsiteTerm, neighborTerm):
    neighborTerm = np.reshape(neighborTerm, (2, 2, 2, 2))

    hSingles = [None] * N
    for i in range(N):
        hSingles[i] = tn.Node(onsiteTerm, name=('single' + str(i)), axis_names=['s' + str(i) + '*', 's' + str(i)])
    hr2l = [None] * (N)
    hl2r = [None] * (N)
    for i in range(N-1):
        pairOp = tn.Node(neighborTerm, \
                         axis_names=['s' + str(i) + '*', 's' + str(i+1) + '*', 's' + str(i), 's' + str(i+1)])
        splitted = tn.split_node(pairOp, [pairOp[0], pairOp[2]], [pairOp[1], pairOp[3]], \
                                          left_name=('l2r' + str(i)), right_name=('r2l' + str(i) + '*'), edge_name='m')
        hr2l[i+1] = bops.permute(splitted[1], [1, 2, 0])
        hl2r[i] = splitted[0]
    return HOp(hSingles, hr2l, hl2r)

class HExpValMid:
    def __init__(self, identityChain, opSum, openOp):
    # HLR.identityChain is
    # I x I x I...
    # for all sites contracted (two degree tensor).
    # HLR.opSum is
    # H(1).single x I x I... + I x H(2).single x I... + H.l2r(1) x H.r2l(2) x I...(two degree tensor)
    # HLR.openOp is
    # I x I x...x H(l).l2r(three degree tensor)
        self.identityChain = identityChain
        self.opSum = opSum
        self.openOp = openOp


# Returns <H> for the lth site:
#  If l is in the begining of the chain, returns
#   _
#  | |--
#  | |
#  | |-- for dir='>>', and the miror QSpace for dir = '<<'
#  | |
#  |_|--
#
#  else, performs
#   _        _
#  | |--  --| |--
#  | |      | |
#  | |--  --| |--
#  | |      | |
#  |_|--  --|_|--
def getHLR(psi, l, H, dir, HLRold):
    if dir == '>>':
        if l == -1:
            identityChain = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0))))
            opSum = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0))))
            openOp = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0))))
            return HExpValMid(identityChain, opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            psil[0] ^ psilConj[0]
            psil[1] ^ psilConj[1]
            identityChain = tn.contract_between(psil, psilConj, name='identity-chain')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            singlel = bops.copyState([H.singles[l]], conj=False)[0]
            psil[0] ^ psilConj[0]
            psil[1] ^ singlel[0]
            psilConj[1] ^ singlel[1]
            opSum1 = tn.contract_between(psil, \
                     tn.contract_between(singlel, psilConj), name='operator-sum')
            if l > 0:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.openOp])[0]
                r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
                psil[0] ^ HLRoldCopy[0]
                psilConj[0] ^ HLRoldCopy[1]
                psil[1] ^ r2l_l[0]
                psilConj[1] ^ r2l_l[1]
                r2l_l[2] ^ HLRoldCopy[2]
                opSum2 = tn.contract_between(psil, tn.contract_between(psilConj, tn.contract_between(r2l_l, HLRoldCopy)))
                opSum1 = bops.addNodes(opSum1, opSum2)

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[0] ^ HLRoldCopy[0]
            psilConj[0] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1 = bops.addNodes(opSum1, opSum3)

            if l < len(psi) - 1:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
                psil[0] ^ psilConj[0]
                psil[1] ^ l2r_l[0]
                psilConj[1] ^ l2r_l[1]
                openOp =  tn.contract_between(psil, tn.contract_between(psilConj, l2r_l), name='open-operator')
            else:
                openOp = None
            return HExpValMid(identityChain, opSum1, openOp)
    if dir == '<<':
        if l == len(psi):
            identityChain = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2))))
            opSum = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2))))
            openOp = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2))))
            return HExpValMid(identityChain, opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ psilConj[1]
            identityChain = tn.contract_between(psil, psilConj, name='identity-chain')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            single_l = bops.copyState([H.singles[l]], conj=False)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ single_l[0]
            psilConj[1] ^ single_l[1]
            opSum1 = tn.contract_between(psil, \
                                         tn.contract_between(single_l, psilConj), name='operator-sum')

            if l < len(psi) -1:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.openOp])[0]
                l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
                psil[2] ^ HLRoldCopy[0]
                psilConj[2] ^ HLRoldCopy[1]
                psil[1] ^ l2r_l[0]
                psilConj[1] ^ l2r_l[1]
                l2r_l[2] ^ HLRoldCopy[2]
                opSum2 = tn.contract_between(psil, tn.contract_between(psilConj, tn.contract_between(l2r_l, HLRoldCopy)))
                opSum1 = bops.addNodes(opSum1, opSum2)

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1 = bops.addNodes(opSum1, opSum3)

            if l > 0:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
                psil[2] ^ psilConj[2]
                psil[1] ^ r2l_l[0]
                psilConj[1] ^ r2l_l[1]
                openOp = tn.contract_between(psil, tn.contract_between(psilConj, r2l_l), name='open-operator')
            else:
                openOp = None
            return HExpValMid(identityChain, opSum1, openOp)


# k is the working site
def lanczos(HR, HL, H, k, psi):
    [T, base] = getTridiagonal(HR, HL, H, k, psi)
    [Es, Vs] = np.linalg.eig(T)
    minIndex = np.argmin(Es)
    E0 = Es[minIndex]
    M = None
    for i in range(len(Es)):
        M = bops.addNodes(M, bops.multNode(base[i], Vs[i][minIndex]))

    M = bops.multNode(M, 1/bops.getNodeNorm(M))
    return [M, E0]


def getTridiagonal(HR, HL, H, k, psi):
    accuracy = 1e-10 # 1e-12

    v = bops.multiContraction(psi[k], psi[k+1], '2', '0')
    # Small innaccuracies ruin everything!
    v.set_tensor(v.get_tensor() / bops.getNodeNorm(v))

    base = []
    base.append(v)
    Hv = applyHToM(HR, HL, H, v, k)
    alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()

    w = bops.addNodes(Hv, bops.multNode(v, -alpha))
    beta = bops.getNodeNorm(w)

    # Start with T as an array and turn into tridiagonal matrix at the end.
    Tarr = [[0, 0, 0]]
    Tarr[0][1] = alpha
    counter = 0
    formBeta = 2 * beta # This is just some value to init formBeta > beta.
    while (beta > accuracy) and (counter <= 50) and (beta < formBeta):
        Tarr[counter][2] = beta
        Tarr.append([0, 0, 0])
        Tarr[counter + 1][0] = beta
        counter += 1

        v = bops.multNode(w, 1 / beta)
        base.append(v)
        Hv = applyHToM(HR, HL, H, v, k)
        alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()
        Tarr[counter][1] = alpha
        w = bops.addNodes(bops.addNodes(Hv, bops.multNode(v, -alpha)), \
                          bops.multNode(base[counter-1], -beta))
        formBeta = beta
        beta = bops.getNodeNorm(w)
    T = np.zeros((len(Tarr), len(Tarr)))
    T[0][0] = Tarr[0][1]
    T[0][1] = Tarr[0][2]
    for i in range(1, len(Tarr)-1):
        T[i][i-1] = T[i][0]
        T[i][i] = T[i][1]
        T[i][i+1] = T[i][2]
    T[len(Tarr)-1][len(Tarr)-2] = T[len(Tarr)-1][0]
    T[len(Tarr) - 1][len(Tarr) - 1] = T[len(Tarr) - 1][1]
    return [T, base]


def applyHToM(HR, HL, H, M, k):
    k1 = k
    k2 = k + 1

    # Add HL.opSum x h.identity(k1) x h.identity(k2) x I(Right)
    # and I(Left) x h.identity(k1) x h.identity(k2) x HR.opSum
    Hv = bops.multiContraction(HL.opSum, M, '0', '0')
    Hv = bops.addNodes(Hv, bops.multiContraction(M, HR.opSum, '3', '0'))

    # Add I(Left) x h.single(k1) x h.identity(k2) x I(Right)
    # And I(Left) x h.identity(k1) x h.single(k2) x I(Right)
    Hv = bops.addNodes(Hv, \
                       tn.transpose(bops.multiContraction(M, H.singles[k1], '1', '0'), [0, 3, 1, 2]))
    Hv = bops.addNodes(Hv, \
                       tn.transpose(bops.multiContraction(M, H.singles[k2], '2', '0'), [0, 1, 3, 2]))

    # Add HL.openOp x h.r2l(k1) x h.identity(k2) x I(Right)
    # And I(Left) x h.identity(k1) x h.l2r(k2) x HR.openOp
    HK1R2L = bops.permute(bops.multiContraction(H.r2l[k1], M, '0', '1'), [2, 1, 0, 3, 4])
    Hv = bops.addNodes(Hv, \
                      bops.multiContraction(HL.openOp, HK1R2L, '02', '01'))
    HK2L2R = bops.permute(bops.multiContraction(H.l2r[k2], M, '0', '2'), [2, 3, 0, 1, 4])
    Hv = bops.addNodes(Hv, \
                       bops.multiContraction(HK2L2R, HR.openOp, '34', '02'))

    # Add I(Left) x h.l2r(k1) x h.r2l(k2) x I(Right)
    HK1K2 = bops.multiContraction(M, H.l2r[k1], '1', '0')
    HK1K2 = bops.multiContraction(HK1K2, H.r2l[k2], '14', '02')
    HK1K2 = bops.permute(HK1K2, [0, 2, 3, 1])
    Hv = bops.addNodes(Hv, HK1K2)

    return Hv


# def decomposeAndTruncate(M, k, dir, opts)
#     # perform an SVD decomposition on M.
#
#     [psi(k), psi(k+1), I] = myOrthoQS(M, [1, 2], dir, opts);
#     truncErr = I.svd2tr;
#     psi(k).info.itags(length(psi(k).info.itags)) = ...
#         strcat(int2str(k), 'a', psi(k).info.itags(length(psi(k).info.itags)));
#     psi(k+1).info.itags(1) = strcat(int2str(k), 'a', psi(k+1).info.itags(1));


N=8
psi = bops.getStartupState(N)
t = 0.5
onsiteTerm = np.zeros((2, 2))
onsiteTerm[1][1] = 1
onsiteTerm[0][0] = 1
neighborTerm = np.zeros((4, 4))
neighborTerm[1][2] = 1
neighborTerm[2][1] = 1
neighborTerm[0][0] = 1
neighborTerm[3][3] = 1
H = getDMRGH(N, onsiteTerm, neighborTerm)
HLs = [None] * (N+1)
HLs[0] = getHLR(psi, -1, H, '>>', 0)
for l in range(N):
    HLs[l+1] = getHLR(psi, l, H, '>>', HLs[l])
HRs = [None] * (N+1)
HRs[N] = getHLR(psi, N,  H, '<<', 0)
k = N-2
[T, base] = getTridiagonal(HRs[k+2], HLs[k], H, k, psi)
[M, E0] = lanczos(HRs[k+2], HLs[k], H, k, psi)
# TODO this gives s5, s7 for some reason
bops.printNode(M)
# TODO this is buggy
[l, r] = bops.svdTruncation(M, M.get_all_edges()[:2], M.get_all_edges()[2:], '<<')
bops.printNode(l)
bops.printNode(r)