import numpy as np
from mgw.ne_oracle import oracleNE
from mgw.sinkhorn_oracle import oracleSinkhorn
from mgw.mgw_utils import factorized_square_Euclidean

"""
Overall scheme:
initialize values
Initialize matrices
while cond:
    for i=1...k:
        (Ai,Bi,Gi,Wi) = oracle.step(Ai,Bi,Gi,Wi)
        oracle.update_plan(A1...Ak)
    t += 1
return (A1..Ak),(B1..Bk)
"""

class entropicMGW():
    def __init__(self, params, data):
        self.params = params
        self.k = params.k
        self.L = 64
        self.delta = 1e-6
        self.delta_sinkhorn = 1e-9
        self.numiter_sinkhorn = 100
        self.dims = params.dims
        self.calc_M(data)
        self.initialize_matrices()
        if params.alg == 'sinkhorn_gw':
            self.oracle = oracleSinkhorn(params, data)
        elif params.alg == 'ne_gw':
            self.oracle = oracleNE(params, data)

    def calc_M(self, data):
        self.M = []
        X,MU = data
        for i in range(self.k):
            x = X[:,:,i]
            y = X[:,:, (i+1) % self.k ]
            wtx = MU[i,:]
            wty = MU[(i+1) % self.k ,:]
            self.M.append((np.dot(np.sum(x ** 2, axis=1), wtx) * np.dot(np.sum(y ** 2, axis=1), wty)) ** (0.5) + 1e-5)


    def initialize_matrices(self):
        self.G = [1]*self.k
        self.W = [0]*self.k
        self.A = [np.zeros(self.dims[i],self.dims[(i + 1) % self.k]) for i in range(self.k)]


    def train(self, data):
        """
        training routine of the entropic MGW alg with oracle
        """
        t=0
        Pi = [0]*self.k
        # generalization of the bimargin condition
        while all([np.linalg.norm(self.G[i]) > self.delta for i in range(self.k)]):
            # start with Sinkhorn:
            self.oracle.update(data)
            Pi = self.oracle.calc_plan()
            for i in range(self.k):
                self.update_matrices(Pi[i], i, t)
            t += 1

        # calc MGW cost:
        # option 1 - calculate the OT cost for each pair and take the sum
        c1 = 0
        c2 = 0
        X, MU = data
        for i in range(self.k):
            x = X[:, :, i]
            y = X[:, :, (i + 1) % self.k]

            A_1, A_2 = factorized_square_Euclidean(x, x)
            B_1, B_2 = factorized_square_Euclidean(y, y)

            wtx = MU[i, :]
            wty = MU[(i + 1) % self.k, :]

            c2 += -2 * np.trace(np.dot(np.dot(np.dot(B_2, Pi.T), A_1), np.dot(np.dot(A_2, Pi), B_1)))
            c1 += np.dot(np.dot(wtx, np.dot(A_1, A_2) ** 2), wtx) + np.dot(wty, np.dot(np.dot(B_1, B_2) ** 2, wty))

        # option 2 - calculate the OT cost and add frobenious norms
        c1 = sum([np.linalg.norm(A) for A in self.A])
        c2 = self.oracle.calc_ot_cost()

        return c1+c2


    def update_matrices(self, Pi, i, t, data):
        X, _ = data
        x = X[:, :, i]
        y = X[:, :, (i + 1) % self.k]
        # update G using gradient:
        if self.params.cost == 'quad_gw':
            self.G[i] = 64*self.A[i] - 32 * np.dot(np.dot(x.T, Pi), y)
        elif self.params.cost == 'ip_gw':
            self.G[i] = 16*self.A[i] - 8 * np.dot(np.dot(x.T, Pi), y)

        # update W:
        self.W[i] = self.W[i] + (t + 1) / 2 * self.G[i]

        # update A:
        px1 = self.A[i] - self.G[i] / self.L
        nm1 = np.linalg.norm(px1)
        Yk = min(1, self.M[i] / (2 * nm1)) * px1

        nm2 = np.linalg.norm(self.W[i]) / self.L
        Zk = -1 / self.L * min(1, self.M[i] / (2 * nm2)) * self.W[i]

        self.A[i] = 2 / (t + 3) * Zk + (t + 1) / (t + 3) * Yk

