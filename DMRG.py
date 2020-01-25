import tensornetwork as tn
import numpy as np
import basicOperations as bops

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
    hl2r = [None] * (N-1)
    for i in range(N-1):
        pairOp = tn.Node(neighborTerm, \
                         axis_names=['s' + str(i) + '*', 's' + str(i+1) + '*', 's' + str(i), 's' + str(i+1)])
        splitted = tn.split_node(pairOp, [pairOp[0], pairOp[2]], [pairOp[1], pairOp[3]], \
                                          left_name=('l2r' + str(i)), right_name=('r2l' + str(i) + '*'), edge_name='m')
        hr2l[i+1] = splitted[0]
        hl2r[i] = splitted[1]
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
        if l == 0:
            psi0 = bops.copyState([psi[l]], conj=False)[0]
            psi0Conj = bops.copyState([psi[l]], conj=True)[0]
            psi0[0] ^ psi0Conj[0]
            psi0[1] ^ psi0Conj[1]
            identityChain = tn.contract_between(psi0, psi0Conj, name='identity-chain')
            psi0 = bops.copyState([psi[l]], conj=False)[0]
            psi0Conj = bops.copyState([psi[l]], conj=True)[0]
            single0 = bops.copyState([H.singles[l]], conj=False)[0]
            psi0[0] ^ psi0Conj[0]
            psi0[1] ^ single0[0]
            single0[1] ^ psi0Conj[1]
            opSum = tn.contract_between(psi0, tn.contract_between(psi0Conj, single0), name = 'operator-sum')
            psi0 = bops.copyState(psi, conj=False)[0]
            psi0Conj = bops.copyState(psi, conj=True)[0]
            l2r0 = bops.copyState(H.l2r, conj=False)[0]
            psi0[0] ^ psi0Conj[0]
            psi0[1] ^ l2r0[1]
            l2r0[2] ^ psi0Conj[1]
            openOp = tn.contract_between(psi0, tn.contract_between(psi0Conj, l2r0), name='open-operator')
            return HExpValMid(identityChain, opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
            psil[0] ^ HLRoldCopy[0]
            psilConj[0] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            identityChain = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy), name='identity-chain')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
            singlel = bops.copyState([H.singles[l]], conj=False)[0]
            psil[0]^HLRoldCopy[0]
            psilConj[0] ^ HLRoldCopy[1]
            psil[1] ^ singlel[0]
            psilConj[1] ^ singlel[1]
            opSum1 = tn.contract_between(HLRoldCopy, \
                     tn.contract_between(psil, \
                     tn.contract_between(singlel, psilConj)), name='operator-sum')

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
            opSum1.set_tensor(opSum1.get_tensor() + opSum2.get_tensor())

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[0] ^ HLRoldCopy[0]
            psilConj[0] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1.set_tensor(opSum1.get_tensor() + opSum3.get_tensor())

            if l < len(psi) - 1:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
                l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
                psil[0] ^ HLRoldCopy[0]
                psilConj[0] ^ HLRoldCopy[1]
                psil[1] ^ l2r_l[1]
                psilConj[1] ^ l2r_l[2]
                openOp =  tn.contract_between(HLRoldCopy, \
                    tn.contract_between(psil, tn.contract_between(psilConj, l2r_l)), name='open-operator')
            else:
                openOp = None
            return HExpValMid(identityChain, opSum1, openOp)
    if dir == '<<':
        if l == len(psi)-1:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ psilConj[1]
            identityChain = tn.contract_between(psil, psilConj, name='identity-chain')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            singlel = bops.copyState([H.singles[l]], conj=False)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ singlel[0]
            singlel[1] ^ psilConj[1]
            opSum = tn.contract_between(psil, tn.contract_between(psilConj, singlel), name='operator-sum')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
            psil[2] ^ psilConj[2]
            psil[1] ^ r2l_l[0]
            r2l_l[1] ^ psilConj[1]
            openOp = tn.contract_between(psil, tn.contract_between(psilConj, r2l_l), name='open-operator')
            return HExpValMid(identityChain, opSum, openOp)
        else:
            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            identityChain = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy), name='identity-chain')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
            single_l = bops.copyState([H.singles[l]], conj=False)[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ single_l[0]
            psilConj[1] ^ single_l[1]
            opSum1 = tn.contract_between(HLRoldCopy, \
                                         tn.contract_between(psil, \
                                         tn.contract_between(single_l, psilConj)), name='operator-sum')

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.openOp])[0]
            l2r_l = bops.copyState([H.l2r[l]], conj=False)[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ l2r_l[1]
            psilConj[1] ^ l2r_l[2]
            l2r_l[0] ^ HLRoldCopy[2]
            opSum2 = tn.contract_between(psil, tn.contract_between(psilConj, tn.contract_between(l2r_l, HLRoldCopy)))
            opSum1.set_tensor(opSum1.get_tensor() + opSum2.get_tensor())

            psil = bops.copyState([psi[l]], conj=False)[0]
            psilConj = bops.copyState([psi[l]], conj=True)[0]
            HLRoldCopy = bops.copyState([HLRold.opSum])[0]
            psil[2] ^ HLRoldCopy[0]
            psilConj[2] ^ HLRoldCopy[1]
            psil[1] ^ psilConj[1]
            opSum3 = tn.contract_between(psil, tn.contract_between(psilConj, HLRoldCopy))
            opSum1.set_tensor(opSum1.get_tensor() + opSum3.get_tensor())

            if l > 0:
                psil = bops.copyState([psi[l]], conj=False)[0]
                psilConj = bops.copyState([psi[l]], conj=True)[0]
                HLRoldCopy = bops.copyState([HLRold.identityChain])[0]
                r2l_l = bops.copyState([H.r2l[l]], conj=False)[0]
                psil[2] ^ HLRoldCopy[0]
                psilConj[2] ^ HLRoldCopy[1]
                psil[1] ^ r2l_l[0]
                psilConj[1] ^ r2l_l[1]
                openOp = tn.contract_between(HLRoldCopy, \
                                 tn.contract_between(psil, tn.contract_between(psilConj, r2l_l)), name='open-operator')
            else:
                openOp = None
            return HExpValMid(identityChain, opSum1, openOp)


N=4
psi = bops.getStartupState(N)
t = 0.5
onsiteTerm = np.zeros((2, 2))
onsiteTerm[1][1] = 1
neighborTerm = np.zeros((4, 4))
neighborTerm[1][2] = 1
neighborTerm[2][1] = 1
H = getDMRGH(N, onsiteTerm, neighborTerm)
HL = getHLR(psi, 0, H, '>>', 0)
for l in range(1, N):
    HL = getHLR(psi, l, H, '>>', HL)
HR = getHLR(psi, N-1,  H, '<<', 0)
l = N-2
while l >= 0:
    HR = getHLR(psi, l, H, '<<', HR)
    bops.printNode(HR.identityChain)
    bops.printNode(HR.openOp)
    bops.printNode(HR.opSum)
    l -= 1
