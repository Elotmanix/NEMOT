from utils.config import PreprocessMeta
from sinkhorn.sinkhorn_utils import MOT_Sinkhorn
from utils.data_fns import gen_data
import numpy as np



def main():
    # TD:
    np.random.seed(42)
    params = PreprocessMeta()
    params['alg'] = 'sinkhorn_mot'
    params['cost_graph'] = 'circle'
    params.n = 101
    params.k = 5
    params.eps = 0.5
    params.dims = [4]*params.k

    # params = gen_params()

    X, MU = gen_data(params)

    # for i in range(1,params.k):
    #     X[:,:,i] = X[:,:,0]

    MOT_agent = MOT_Sinkhorn(params, X, MU)
    MOT_agent.solve_sinkhorn()
    # CALC OT COST
    ot_cost = MOT_agent.calc_ot_cost()
    print('OT cost:', ot_cost)
    if params.save_results:
        MOT_agent.save_results()



#OLD:
# class MOT_Sinkhorn():
#     def __init__(self, params, X, MU):
#         self.eps = params.eps
#         self.tol = 1e-10
#         self.k = params['k']
#         self.calc_kernel(params['cost_graph'],X)
#         self.MU = MU
#
#     def train(self):
#         """
#         alog flow:
#         1. initialize phi vectors
#         2. initialize beta vectors (for circle or tree structure)
#         3. loop
#             a. update phi vectors according to cost graph structure
#             b. update beta vectors (for circle or tree structure)
#             c. calculate error
#         :return:
#         """
#         # initialize error list
#         S = [0]
#         err = [np.inf]
#         # initialize phis (as zeros):
#         prev_phi = [np.zeros(self.n) for i in range(self.k)]
#         # initialize betas
#         beta = self.update_beta(prev_phi)
#         iter = 0
#         while err[-1] > self.tol:
#             phi = self.update_phi(prev_phi, beta)
#             beta = self.update_beta(phi)
#
#             S.append(self.calc_val(phi))
#
#             prev_phi=phi
#             iter += 1
#             err.append( np.abs(S[i]-S[i-1]) )
#         err.pop(0)
#         pass
#
#     def update_phi(self, prev_phi, prev_beta):
#         return 0
#
#     def calc_val(self, phi):
#         return 0
#
#     def update_beta(self, phi):
#         #TD - 'init' is for initialization phase, o
#         beta = []
#         for k in range(self.k-1, -1, -1):
#             if k == self.k-1:
#                 b = self.kernel[k]
#             else:
#                 b = (phi[k+1]*b)
#             beta = [b] + beta
#         return 0
#
#     def calc_kernel(self, cost_graph, X):
#         """
#         calculate the kernel.
#         X has shape (n,d,k)
#         Kernel shape depends on cost graphical structure
#         """
#         if cost_graph == 'circle':
#             # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
#             # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
#             K = []
#             for i in range(self.k):
#                 C = QuadCost(data=X, mod='circle')
#                 K = [torch.exp(-c/self.eps) for c in C]
#
#         elif cost_graph == 'tree':
#             # TD
#             K=0
#
#         elif cost_graph == 'full':
#             # TD - figure out how to generate cost tensor in QuadCost
#             C = QuadCost(data=X, mod='full')
#             K = torch.exp(-C/self.eps)
#
#         self.kernel = K






if __name__ == '__main__':
    main()