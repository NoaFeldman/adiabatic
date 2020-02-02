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
    def __init__(self, opSum, openOp):
    # HLR.opSum is
    # H(1).single x I x I... + I x H(2).single x I... + H.l2r(1) x H.r2l(2) x I...(two degree tensor)
    # HLR.openOp is
    # I x I x...x H(l).l2r(three degree tensor)
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
            opSum = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0))))
            openOp = tn.Node(np.zeros((psi[0].get_dimension(0), psi[0].get_dimension(0))))
            return HExpValMid(opSum, openOp)
        else:
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
                openOp = tn.contract_between(psil, tn.contract_between(psilConj, l2r_l), name='open-operator')
            else:
                openOp = None
            return HExpValMid(opSum1, openOp)
    if dir == '<<':
        if l == len(psi):
            opSum = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2))))
            openOp = tn.Node(np.zeros((psi[l-1].get_dimension(2), psi[l-1].get_dimension(2))))
            return HExpValMid(opSum, openOp)
        else:
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
            return HExpValMid(opSum1, openOp)


# k is the working site
def lanczos(HL, HR, H, k, psi, psiCompare):
    [T, base] = getTridiagonal(HL, HR, H, k, psi, psiCompare)
    [Es, Vs] = np.linalg.eig(T)
    minIndex = np.argmin(Es)
    E0 = Es[minIndex]
    M = None
    for i in range(len(Es)):
        M = bops.addNodes(M, bops.multNode(base[i], Vs[i][minIndex]))

    M = bops.multNode(M, 1/bops.getNodeNorm(M))
    return [M, E0]

def getIdentity(psi, k, dir):
    psil = bops.copyState([psi[k]])[0]
    psilCopy = bops.copyState([psi[k]], conj=True)[0]
    if dir == '>>':
        result = bops.multiContraction(psil, psilCopy, '01', '01').tensor
    else:
        result = bops.multiContraction(psil, psilCopy, '12', '12').tensor
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = round(result[i][j], 2)
    return result


def getTridiagonal(HL, HR, H, k, psi, psiCompare=None):
    accuracy = 1e-10 # 1e-12

    v = bops.multiContraction(psi[k], psi[k+1], '2', '0')
    # Small innaccuracies ruin everything!
    v.set_tensor(v.get_tensor() / bops.getNodeNorm(v))

    psiCopy = bops.copyState(psi)

    base = []
    base.append(v)
    Hv = applyHToM(HL, HR, H, v, k)
    alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()

    if psiCompare is not None:
        copyV = bops.copyState([v])[0]
        psiCopy = bops.assignNewSiteTensors(psiCopy, k, copyV, '>>')[0]
        print('line 196, k = ' + str(k) + ', overlap = ' + str(bops.getOverlap(psiCopy, psiCompare)))

    E = stateEnergy(psi, H)

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

        if psiCompare is not None:
            copyV = bops.copyState([v])[0]
            psiCopy = bops.assignNewSiteTensors(psiCopy, k, copyV, '>>')[0]
            print('line 219, k = ' + str(k) + ', counter = ' + str(counter) + ', overlap = ' + str(bops.getOverlap(psiCopy, psiCompare)))
        Hv = applyHToM(HL, HR, H, v, k)

        alpha = bops.multiContraction(v, Hv, '0123', '0123*').get_tensor()
        Tarr[counter][1] = alpha
        w = bops.addNodes(bops.addNodes(Hv, bops.multNode(v, -alpha)), \
                          bops.multNode(base[counter-1], -beta))
        formBeta = beta
        beta = bops.getNodeNorm(w)
    T = np.zeros((len(Tarr), len(Tarr)))
    T[0][0] = Tarr[0][1]
    if len(Tarr) > 1:
        T[0][1] = Tarr[0][2]
    for i in range(1, len(Tarr)-1):
        T[i][i-1] = Tarr[i][0]
        T[i][i] = Tarr[i][1]
        T[i][i+1] = Tarr[i][2]
    T[len(Tarr)-1][len(Tarr)-2] = Tarr[len(Tarr)-1][0]
    T[len(Tarr) - 1][len(Tarr) - 1] = Tarr[len(Tarr) - 1][1]
    return [T, base]


def applyHToM(HL, HR, H, M, k):
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
    HK1R2L = bops.permute(bops.multiContraction(M, H.r2l[k1], '1', '0'), [0, 4, 3, 1, 2])
    Hv = bops.addNodes(Hv, \
                      bops.multiContraction(HL.openOp, HK1R2L, '02', '01'))
    HK2L2R = bops.permute(bops.multiContraction(M, H.l2r[k2], '2', '0'), [0, 1, 3, 4, 2])
    Hv = bops.addNodes(Hv, \
                       bops.multiContraction(HK2L2R, HR.openOp, '43', '02'))

    # Add I(Left) x h.l2r(k1) x h.r2l(k2) x I(Right)
    HK1K2 = bops.multiContraction(M, H.l2r[k1], '1', '0')
    HK1K2 = bops.multiContraction(HK1K2, H.r2l[k2], '14', '02')
    HK1K2 = bops.permute(HK1K2, [0, 2, 3, 1])
    Hv = bops.addNodes(Hv, HK1K2)

    return Hv


def dmrgStep(HL, HR, H, psi, k, dir, psiCompare=None, opts=None):
    # Perform a single DMRG step:
    # 1. Contracts psi(k) and psi(k + dir) to get M.
    # 2. Performs lancsoz and get a new contracted M.
    # 3. Performs an SVD in order to get the new working site, at k + dir.
    # 4. Calculates HL(k) / HR(k) (according to dir)
    k1 = k
    k2 = k + 1
    [M, E0] = lanczos(HL, HR, H, k1, psi, psiCompare)
    [psi, truncErr] = bops.assignNewSiteTensors(psi, k, M, dir)
    if dir == '>>':
        if psiCompare is not None:
            psiCompare = bops.shiftWorkingSite(psiCompare, k, '>>')
            psi = bops.getOrthogonalState(psiCompare, psiInitial=psi)
        newHL = getHLR(psi, k, H, dir, HL)
        return psi, newHL, E0, truncErr
    else:
        if psiCompare is not None:
            psiCompare = bops.shiftWorkingSite(psiCompare, k, '<<')
            psi = bops.getOrthogonalState(psiCompare, psiInitial=psi)
        newHR = getHLR(psi, k+1, H, dir, HR)
        return psi, newHR, E0, truncErr


# Assume the OC is at the last (rightmost) site. sweeps all the way left and back right again.
def dmrgSweep(psi, H, HLs, HRs, psiCompare=None):
    k = len(psi) - 2
    maxTruncErr = 0
    while k > 0:
        [psi, newHR, E0, truncErr] = dmrgStep(HLs[k], HRs[k+2], H, psi, k, '<<', psiCompare)
        # if HRs[k+1] is not None:
        # TODO remove all nodes in HLR
            # tn.remove_node(HRs[k+1])
        HRs[k+1] = newHR
        if len(truncErr) > 0 and maxTruncErr < max(truncErr):
            maxTruncErr = max(truncErr)
        k -= 1
    for k in range(len(psi) - 2):
        E0Old = E0
        [psi, newHL, E0, truncErr] = dmrgStep(HLs[k], HRs[k + 2], H, psi, k, '>>', psiCompare)
        if E0 > E0Old:
            print('E0 > E0Old, k = ' + str(k) + ', E0Old = ' + str(E0Old) + ', E0 = ' + str(E0))
        # if HLs[k+1] is not None:
        # TODO remove all nodes in HLR
        #     tn.remove_node(HLs[k+1])
        HLs[k + 1] = newHL
        if len(truncErr) > 0 and maxTruncErr < max(truncErr):
            maxTruncErr = truncErr
    return psi, E0, truncErr


def getH(N, onsiteTerm, neighborTerm, psi):
    H = getDMRGH(N, onsiteTerm, neighborTerm)
    HLs = [None] * (N + 1)
    HLs[0] = getHLR(psi, -1, H, '>>', 0)
    for l in range(N):
        HLs[l + 1] = getHLR(psi, l, H, '>>', HLs[l])
    HRs = [None] * (N + 1)
    HRs[N] = getHLR(psi, N, H, '<<', 0)
    return H, HLs, HRs


def getGroundState(H, HLs, HRs, N, psi, psiCompare=None, accuration=10**(-8)):
    truncErrs = []
    [psi, E0, truncErr] = dmrgSweep(psi, H, HLs, HRs, psiCompare)
    truncErrs.append(truncErr)
    while True:
        [psi, E0Curr, truncErr] = dmrgSweep(psi, H, HLs, HRs)
        truncErrs.append(truncErr)
        if math.fabs((E0Curr-E0)/E0) < accuration:
            return psi, E0Curr, truncErrs
        E0 = E0Curr


def stateEnergy(psi: tn.Node, H: HOp):
    E = 0
    for i in range(len(psi)):
        psiCopy = bops.copyState(psi)
        single_i = bops.copyState([H.singles[i]])[0]
        psiCopy[i] = bops.permute(tn.contract(psiCopy[i][1] ^ single_i[0], name=('site' + str(i))), [0, 2, 1])
        E += bops.getOverlap(psiCopy, psi)
        bops.removeState(psiCopy)
        tn.remove_node(single_i)
    for i in range(len(psi) - 1):
        psiCopy = bops.copyState(psi)
        r2l = bops.copyState([H.r2l[i+1]])[0]
        l2r = bops.copyState([H.l2r[i]])[0]
        psiCopy[i][2] ^ psiCopy[i+1][0]
        psiCopy[i][1] ^ l2r[0]
        r2l[0] ^ psiCopy[i+1][1]
        l2r[2] ^ r2l[2]
        M = tn.contract_between(psiCopy[i], \
                                tn.contract_between(l2r, tn.contract_between(r2l, psiCopy[i+1])))
        [psiCopy, te] = bops.assignNewSiteTensors(psiCopy, i, M, '>>')
        E += bops.getOverlap(psiCopy, psi)
        bops.removeState(psiCopy)
        tn.remove_node(r2l)
        tn.remove_node(l2r)
    return E




N = 8
t = 0.5
onsiteTerm = np.zeros((2, 2))
onsiteTerm[1][1] = 0
onsiteTerm[0][0] = 0
neighborTerm = np.zeros((4, 4))
neighborTerm[1][2] = 1
neighborTerm[2][1] = 1
neighborTerm[0][0] = 0
neighborTerm[3][3] = 0
psi1 = bops.getStartupState(N)
H, HLs, HRs = getH(N, onsiteTerm, neighborTerm, psi1)
psi1, E0, truncErrs = getGroundState(H, HLs, HRs, N, psi1, None)
print(stateEnergy(psi1, H))
print(len(truncErrs))
psi2 = bops.getOrthogonalState(psi1)
print(bops.getOverlap(psi1, psi2))
H, HLs, HRs = getH(N, onsiteTerm, neighborTerm, psi2)
psi2, E0, truncErrs = getGroundState(H, HLs, HRs, N, psi2, psi1)
print(stateEnergy(psi2, H))
print(bops.getOverlap(psi1, psi2))
print(len(truncErrs))
# psi3 = bops.addStates(psi2, psi)
# print(bops.getOverlap(psi, psi3))
# print(bops.getOverlap(psi2, psi3))
# psi3[0] = bops.multNode(psi3[0], 1 / math.sqrt(bops.getOverlap(psi3, psi3)))
# H, HLs, HRs = getH(N, onsiteTerm, neighborTerm, psi3)
# psi3, E0, truncErrs = getGroundState(H, HLs, HRs, N, psi3)
# print(stateEnergy(psi3, H))
# print(bops.getOverlap(psi, psi3))
# print(bops.getOverlap(psi2, psi3))