import numpy as np
import torch.nn as nn
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
from utils.data_fns import QuadCost, QuadCostGW, MultiTensorDataset
from utils.tree_fns import create_tree
import pickle
import os


class MOT_NE_alg():
    def __init__(self, params, device):
        self.models = []
        self.params = params
        self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        for i in range(params['k']):
            # model = NE_mot_model(dim=params['dims'][i], hidden_dim=params['hidden_dim'])
            # model.to(device)
            ### Tao's
            d = params['dims'][i]
            model = Net_EOT(dim=d, K=min(6*d,80), deeper=True).to(device)
            ###
            self.models.append(model)
        self.cost = params['cost']
        self.opt = [torch.optim.Adam(list(self.models[i].parameters()), lr=params['lr']) for i in range(self.k)]

        if params.schedule:
            self.scheduler = [torch.optim.lr_scheduler.StepLR(opt, step_size=params.schedule_step, gamma=params.schedule_gamma) for opt in self.opt]

        if self.params.cost_graph == 'tree':
            self.tree_root = create_tree(self.params)
        else:
            self.tree_root = None

        self.device = device
        self.cost_graph = params['cost_graph']
        self.using_wandb = params['using_wandb']
        self.params = params


    def train_mot(self, X):
        x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        self.tot_loss = []
        self.times = []
        for epoch in range(self.num_epochs):
            l = []
            t0 = timer()
            for i, data in enumerate(x_b):
                self.zero_grad_models()
                # data.to(self.device)
                for k_ind in range(self.k):
                    # k-margin loss
                    phi = [self.models[i](data[:,:,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    e_term = self.calc_exp_term(phi, data)
                    loss = -(sum(phi).mean() - self.eps*e_term)
                    if self.params.regularize_pariwise_coupling and self.params.cost_graph != 'full':
                        # print('regularize_pariwise_coupling')
                        pairwise_reg = self.calc_pairwise_coupling_regularizer(phi, data)
                        reg_loss = loss + self.params.regularize_pariwise_coupling_reg * pairwise_reg
                        reg_loss.backward()
                    ####
                    # bimargin loss (for debugging)
                    # cost = self.calc_cost(data)
                    # loss = dual_loss(pred_x = phi[0], pred_y = phi[1], cost=cost, eps=self.eps)
                    ####
                    else:
                        loss.backward()

                    if self.params.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(), self.params.max_grad_norm)

                    self.opt[k_ind].step()
                    l.append(loss.item())

            print_plan_overall = False
            if print_plan_overall:
                phi_all = [self.models[i](X[:,:,i]) for i in range(self.k)]
                print(self.calc_exp_term(phi_all, X))

            l = np.mean(l)
            self.tot_loss.append(-l+ self.eps)
            epoch_time = timer()-t0
            self.times.append(epoch_time)
            print(f'finished epoch {epoch}, loss={-l+ self.eps:.5f}, took {epoch_time:.2f} seconds')

            print_debug = True
            if epoch%10==0 and print_debug and self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
                P = self.calc_plan(X)
                ot_cost = self.calc_ot_cost(P,X)
                print(f'ot_cost {ot_cost}')

            if self.params.schedule:
                for sched in self.scheduler:
                    sched.step()
                    lr = sched.get_last_lr()[0]
                    # print(f'updated learning rate {lr}')

            # print(f'finished epoch {epoch}, loss={l / i}')
        self.models_to_eval

    def save_results(self,X=None):
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params,
            # 'plan': plan,
        }
        if self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
            plan = self.calc_plan(X)
            ###
            ot_cost = self.calc_ot_cost(plan,X)
            ###
            data_to_save['ot_cost'] = ot_cost
        else:
            ot_cost = 0
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss,
                       'avg_time': avg_time
                       })
            if self.params.cost_graph != 'full':
                wandb.log(({
                       'ot_cost': ot_cost
                       }))

        if self.params.cost_graph != 'full':
            print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds, ot_cost {ot_cost:.5f}')
        else:
            print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')

    def calc_ot_cost(self, P, X):
        C = self.calc_cost(X)
        # phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
        # e_term = torch.eye(phi[0].shape[0]).to(self.device)
        # for i in range(self.k):
        #     L = torch.exp((0.5 * (phi[i] + phi[(i + 1) % self.k].T) - C[i]) / self.eps)
        #     e_term = (e_term @ L)
        # normal = torch.trace(e_term)
        ot_cost = sum([torch.sum(c * p) for (c, p) in zip(C, P)])
        return ot_cost
        # return ot_cost/normal

    def calc_exp_term(self, phi, x):
        # TD - IMPLEMENT MORE MEMORY EFFICIENT CALCULATION!!! DIVIDE LOSS!!

        # calc loss tensor

        if self.params.cost_implement == 'simplified':
            reduced_phi = torch.sum(torch.concatenate(phi, axis=1), axis=1)
            if self.cost_graph == 'full':
                # calculate the simplified loss
                # calc reduced phi term:

                # cal reduced cost term:
                diffs = x.unsqueeze(-1) - x.unsqueeze(-2)
                # calc cost. this method counts all cost terms twice, se we divide the cost by 2.
                cost = 0.5*torch.norm(diffs, dim=1) ** 2
                e_term = torch.exp((reduced_phi - cost.sum(axis=(1,2)))/self.eps)
            elif self.cost_graph == 'circle':
                shifted_x = torch.roll(x, shifts=1, dims=-1)
                diffs = x - shifted_x
                cost = torch.norm(diffs, dim=1) ** 2
                e_term = torch.exp((reduced_phi - cost.sum(axis=(-1))) / self.eps)
            return e_term.mean()

        # COMBINATORIAL IMPLEMENTATION
        c = self.calc_cost(x)
        if self.cost_graph == 'circle':
            n = phi[0].shape[0]
            e_term = torch.eye(n).to(self.device)
            for i in range(self.k):
                L = torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps)
                e_term = (e_term @ L)*(1/n)
            return torch.trace(e_term)
            # # calc mapping:
            # reshaped_term = []
            # reshaped_c = []
            # for index, vec in enumerate(phi):
            #     # Create a shape of length k with 1s except at the index position
            #     shape = [1] * self.k
            #     shape[index] = -1
            #     reshaped_c.append(c[index].reshape(shape))
            #     reshaped_term.append(vec.reshape(shape))
            # reshaped_term = sum(reshaped_term)
            # c = sum(reshaped_c)
        elif self.cost_graph == 'full':
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            return torch.mean(torch.exp((reshaped_term-c)/self.eps))
        elif self.cost_graph == 'tree':
            n = phi[0].shape[0]
            e_term = self.calc_exp_term_tree(n,phi,c)
            return e_term



    def calc_exp_term_tree(self, n, phi, c):
        """
        We traverse the tree and aggregate the multiplications.
        """

        def traverse_and_calculate(node):
            # Aggregate children node calculation into V
            V = torch.ones(size=(n,1)).cuda()
            for child in node.children:
                V = V*traverse_and_calculate(child)

            # If we're at the root then we need to calculate
            if node.is_root_flag:
                # there is a vector and the beginning
                L = 1 / n * torch.exp((phi[node.index] ) / self.eps).t()
                return L @ V

            ones = torch.ones(size=(n, 1)).cuda()
            L = 1/n*torch.exp( ( ones@phi[node.index].t() - c[node.index] )/self.eps )

            return L @ V

        # Traverse the tree starting from the root and calculate the matrices
        return traverse_and_calculate(self.tree_root).squeeze()


    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            if self.cost_graph == 'circle' and self.params.euler == 1:
                cost = QuadCost(data, mod='euler')
            else:
                cost = QuadCost(data, mod=self.cost_graph, root=self.tree_root)
        elif self.cost == 'quad_gw':
            # IMPLEMENT - cost = QuadCostGW(data, self.matrices)
            pass
        elif self.cost == 'ip_gw':
            # IMPLEMENT - cost = IPCostGW(data, self.matrices)
            pass

        # NOW - BROADCAST!!
        return cost

    def zero_grad_models(self):
        for opt in self.opt:
            opt.zero_grad()

    def models_to_eval(self):
        for model in self.models:
            model.eval()

    def calc_plan(self, X):
        if self.cost_graph == 'circle':
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            c = self.calc_cost(X)
            exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
            ot_plan = []
            for i in range(self.k):
                P = self.calc_pairwise_plan(exp_terms,i, (i+1)%self.k )
                if self.params.normalize_plan:
                    P = P/torch.sum(P)
                ot_plan.append(P)
                if self.params.check_P_sum and self.params.using_wandb:
                    print('h')
                    wandb.log({f'ot_plan_{i}_sum': torch.sum(ot_plan[i]).item()})
        else:
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            c = self.calc_exp_term(phi, X)
            ot_plan = torch.exp((reshaped_term - c) / self.eps)
        return ot_plan

    def calc_pairwise_plan_old(self, L,i,j):
        '''
        calculate the pairwise plan from margin i to margin j
        TD (from pic) - start from (i,i+1) and then (k,1) and then general (i,j)?
        Pi_{i,j} = \prod_{l=i}^{j-1}L_{l,l+1} \odot (\prod_{l=1}^{i-1}L_{l,l+1}) @ (\prod_{l=j+1}^{k}L_{l,l+1}).T
        currently implemented for same number of samples.
        '''
        k = len(L)
        n = L[0].shape[0]

        if i == 0:
            A = torch.eye(L[0].shape[0]).to(self.device)
            B = torch.eye(L[0].shape[0]).to(self.device)

            for l in range(k):
                if l<j:
                    A = A@L[l]
                else:
                    B = B@L[l]
            C = A*B
            print(f'adjacent with zero, P adds up to {torch.sum(C)}')
            return C/(n**k)
        elif j == (i+1)%k:
            if j == i+1:
                # two adjacent idx, i !=0, j=i+1
                A = torch.eye(n).to(self.device)
                B = torch.eye(n).to(self.device)
                for l in range(k):
                    if l <= i:
                        A = A @ L[l]
                    else:
                        B = B @ L[l]
                C = (A.T * B.T)
                print(f'adjacent, P adds up to {torch.sum(C)/(n**k)}')
                return C/(n**k)
            else:
                # cycle case or
                # two adjacent idx, i !=0, j=i+1
                A = torch.eye(n).to(self.device)
                B = torch.eye(n).to(self.device)
                for l in range(k):
                    if l <= i:
                        A = A @ L[l]
                    else:
                        B = B @ L[l]
                C = (A * B.T)
                print(f'adjacent, P adds up to {torch.sum(C) / (L[0].shape[0] ** k)}')
                return C/(n**k)
        else:
            # GENERAL CASE
            return

    def calc_pairwise_plan_(self, L,i,j,verbose=True):
        ### NEW IMPLEMENTATION
        '''
        calculate the pairwise plan from margin i to margin j
        TD (from pic) - start from (i,i+1) and then (k,1) and then general (i,j)?
        Pi_{i,j} = \prod_{l=i}^{j-1}L_{l,l+1} \odot (\prod_{l=1}^{i-1}L_{l,l+1}) @ (\prod_{l=j+1}^{k}L_{l,l+1}).T
        currently implemented for same number of samples.
        '''
        k = len(L)
        n = L[0].shape[0]

        if i == 0:
            A = torch.eye(L[0].shape[0]).to(self.device)
            B = torch.eye(L[0].shape[0]).to(self.device)

            for l in range(k):
                if l<j:
                    A = A@L[l]
                    # A = A@L[l]/n
                else:
                    B = B@L[l]
                    # B = B @ L[l] / n
            C = A.T*B.T
            if verbose:
                print(f'adjacent, P adds up to {torch.sum(C)}')
            return C
        elif j == (i+1)%k:
            if j == i+1:
                # two adjacent idx, i !=0, j=i+1
                A = torch.eye(n).to(self.device)
                B = torch.eye(n).to(self.device)
                for l in range(k):
                    if l <= i:
                        A = A @ L[l]
                        # A = A @ L[l]/n
                        # A = A @ L[l]/(n*L[l].sum(0))
                    else:
                        B = B @ L[l]
                        # B = B @ L[l]/n
                        # B = B @ L[l]/(n*L[l].sum(0))
                C = (A.T * B.T)
                if verbose:
                    print(f'adjacent, P adds up to {torch.sum(C)}')
                return C
            else:
                # cycle case or
                # two adjacent idx, i !=0, j=i+1
                A = torch.eye(n).to(self.device)
                B = torch.eye(n).to(self.device)
                for l in range(k):
                    if l <= i:
                        A = A @ L[l]
                        # A = A @ L[l]/n
                    else:
                        B = B @ L[l]
                        # B = B @ L[l] / n
                C = (A.T * B.T)
                if verbose:
                    print(f'adjacent, P adds up to {torch.sum(C)}')
                return C
        else:
            # GENERAL CASE
            return

    def calc_pairwise_plan(self, L, i, verbose=False):
        # Compute A: Product of matrices up to index i-1
        if i > 0:
            A = L[0]
            for k in range(1, i):
                A = A @ L[k]
        else:
            # Use identity matrix if no matrices before index i
            size = L[i].shape[0]
            A = torch.eye(size, dtype=L[i].dtype, device=L[i].device)

        # Compute B: Product of matrices from index i+1 to end
        if i + 1 < len(L):
            B = L[i + 1]
            for k in range(i + 2, len(L)):
                B = B @ L[k]
        else:
            # Use identity matrix if no matrices after index i
            size = L[i].shape[1]
            B = torch.eye(size, dtype=L[i].dtype, device=L[i].device)

        # Calculate A.T @ B.T
        C = A.T @ B.T

        # Element-wise multiplication with L[i]
        output = C * L[i]

        # if verbose:
        #     print(output.sum())

        return output/output.sum()

    def calc_pairwise_coupling_regularizer(self, phi, x):
        c = self.calc_cost(x)
        exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
        reg = 0
        for i in range(self.k):
            ot_plan = self.calc_pairwise_plan(exp_terms, i, (i + 1) % self.k, verbose=False)
            reg += (torch.sum(ot_plan)-1.0).abs()
        return reg


class MGW_NE_alg(MOT_NE_alg):
    def __init__(self, params, device, X):
        """
        EMGW agent
        built upon the MOT_NE_alg.

        """
        super().__init__(params, device)
        self.initialize_alg(X)
        # same optimizer for all matrices:

    def initialize_alg(self,X):
        """
        Initialize the A matrices of the MGW problem
        """
        if self.params.A_mgw_opt == 'autograd':
            # INIT As at models
            if self.cost_graph == 'full':
                # all (i,j) options
                pass
            elif self.cost_graph == 'circle':
                # k  matrices
                As = []
                self.opt_A = []
                for i in range(self.k):
                    A = A_model(self.params.dims[i],self.params.dims[(i+1)%self.k])
                    self.opt_A.append(torch.optim.Adam(A.parameters(), lr=1e-4))
                    As.append(A.cuda())
            elif self.cost_graph == 'tree':
                # k-1 matrices
                pass
            self.A_matrices = As
        else:
            # Doing first order optimization:
            self.tolerance = 1e-4
            self.max_iter = 100
            self.K_const = 32
            self.M = 1
            self.L = max(64,32*32*(1/9+4/(45*self.params.dims[0]))/self.eps-64)
            norms = [torch.norm(x, dim=1) for x in X]
            self.C_1 = [-4*norms[i][:,None]*norms[(i+1)%self.k][None,:] for i in range(self.k)]
            self.S1 = 0 # TD
            self.A_matrices = [torch.ones(self.params.dims[i],self.params.dims[(i+1)%self.k]) * 1e-5 for i in range(self.k)]
            self.C_matrices = [torch.ones(self.params.dims[i], self.params.dims[(i + 1) % self.k]) * 1e-5 for i in range(self.k)]

    def train_with_oracle(self, X):
        nemot_n_epochs = 5
        x_b = MultiTensorDataset(X)
        x_b = DataLoader(x_b, batch_size=self.batch_size, shuffle=True)

        self.tot_loss = []
        self.times = []

        for iter in range(self.max_iter):

            for epoch in range(nemot_n_epochs):
                l = []
                # perform an epoch
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        # k-margin loss
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss = -(sum(phi).mean() - self.eps * e_term)
                        loss.backward()

                        if self.params.clip_grads:
                            torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(),
                                                           self.params.max_grad_norm)
                        self.opt[k_ind].step()
                        l.append(loss.item())
                l = np.mean(l)
                self.tot_loss.append(-l + self.eps)
                print(f'iter: {iter}, finished NEMOT epoch {epoch}, loss: {-l + self.eps:.5f}')

            # PERFORM A SINGLE A UPDATE:
            # THIS IS LIMITED TO SIMPLIFIED PLANS!
            gamma = iter / (4 * self.L)
            tau = 2 / (iter + 2)
            P = self.calc_plan(X)
            print(f'iteration {iter}')
            for i in range(len(self.A_matrices)):
                A = self.A_matrices[i]
                C = self.C_matrices[i]
                grad = 64 * A - 32 * X[i].T @ P[i] @ X[i + 1]
                print(f'grad_i norm is {torch.linalg.norm(grad)}')
                B = torch.where(torch.abs(A - grad / (2 * self.L)) <= self.M / 2, A - grad / (2 * self.L),
                                self.M / 2)
                C = torch.where(torch.abs(C - grad * gamma) <= self.M / 2, A - grad * gamma, self.M / 2)
                A = tau * C + (1 - tau) * B
                self.A_matrices[i] = A
                self.C_matrices[i] = C

                    # if self.params.schedule:
                    #     for sched in self.scheduler:
                    #         sched.step()

    def calc_exp_term_mgw(self,phi,x):
        norms = [torch.norm(x_, dim=1) ** 2 for x_ in x]

        if self.params.A_mgw_opt == 'autograd':
            cost_list = [torch.diagonal(-4* norms[i][:,None]* norms[(i+1)%self.k][None,:] -32 * self.A_matrices[i]( (x[i], x[(i+1)%self.k])) ) for i in range(self.k)]
        else:
            cost_list = [torch.diagonal(
                -4 * norms[i][:, None] * norms[(i + 1) % self.k][None, :] - 32 * x[i] @ self.A_matrices[i].cuda() @ x[
                    (i + 1) % self.k].T) for i in range(self.k)]

        reduced_phi = torch.sum(torch.concatenate(phi, axis=1), axis=1)
        cost = sum(cost_list)
        e_term = torch.exp((reduced_phi - cost) / self.eps)
        return torch.mean(e_term)

    def calc_plan_mgw(self, X):
        # TD!!! plan calculation!!!
        if self.cost_graph == 'circle':
            phi = [self.models[i](X[i]) for i in range(self.k)]
            c = self.calc_cost_mgw(X)
            exp_terms = [torch.exp((0.5*(phi[i] + phi[ (i+1)%self.k ].T) - c[i])/self.eps) for i in range(self.k)]
            ot_plan = []
            for i in range(self.k):
                P = self.calc_pairwise_plan(exp_terms,i, (i+1)%self.k )
                if self.params.normalize_plan:
                    P = P/torch.sum(P)
                ot_plan.append(P)
                if self.params.check_P_sum and self.params.using_wandb:
                    print('h')
                    wandb.log({f'ot_plan_{i}_sum': torch.sum(ot_plan[i]).item()})
        else:
            phi = [self.models[i](X[:, :, i]) for i in range(self.k)]
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]
            c = self.calc_exp_term(phi, X)
            ot_plan = torch.exp((reshaped_term - c) / self.eps)
        return ot_plan


    def train_mgw_combined(self, X):
        """
        Training both A matrices and MGW via automatic differentiation
        :param X:
        :return:
        """
        x_b = MultiTensorDataset(X)
        x_b = DataLoader(x_b, batch_size=self.batch_size, shuffle=True)
        self.tot_loss = []
        self.times = []
        nemot_epochs = 3
        for epoch in range(self.num_epochs):
            A_opt_flag = epoch > 9 and epoch%nemot_epochs==0
            t0 = timer()
            l = []
            if A_opt_flag:
                # train A
                l = []
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss_ = sum(phi).mean() - self.eps * e_term
                        frob_norms = 32*sum([model.A.norm(p='fro')**2 for model in self.A_matrices])
                        loss = loss_ + frob_norms
                        loss.backward()

                        # if self.params.clip_grads:
                        #     torch.nn.utils.clip_grad_norm_(self.A_matrices[k_ind].parameters(),
                        #                                    self.params.max_grad_norm)
                        self.opt_A[k_ind].step()
                    l.append(loss.item())
                self.tot_loss.append(np.mean(l) + self.eps)
            else:
                # train NEMOT
                for i, data in enumerate(x_b):
                    self.zero_grad_models()
                    for k_ind in range(self.k):
                        # k-margin loss
                        phi = [self.models[i](data[i]) for i in
                               range(self.k)]  # each phi[i] should have shape (b,1)
                        e_term = self.calc_exp_term_mgw(phi, data)
                        loss = -(sum(phi).mean() - self.eps * e_term)
                        loss.backward()

                        if self.params.clip_grads:
                            torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(),
                                                           self.params.max_grad_norm)
                        self.opt[k_ind].step()
                        l.append(loss.item())
            epoch_time = timer() - t0
            l = np.mean(l)
            self.times.append(epoch_time)
            if A_opt_flag:
                print(f'finished epoch {epoch}, A opt loss: {l + self.eps:.5f}')
            else:
                print(f'finished epoch {epoch}, loss: {-l + self.eps:.5f}')

        #




        #
        # X_b = [DataLoader(x, batch_size=self.batch_size, shuffle=True) for x in X]
        # self.tot_loss = []
        # self.times = []
        # self.nemot_n_epochs = 5
        # for epoch in range(self.num_epochs):
        #     l = []
        #     t0 = timer()
        #     if epoch%self.nemot_n_epochs == 0 and epoch>0:  # M MATRICES EPOCH
        #         # train A matrices this epoch:
        #         # for i, data in enumerate(zip(X_b)):
        #         for data in zip(X_b):
        #             self.opt_A.zero_grad()
        #             # k-margin loss
        #             phi = [self.models[i](data[i]) for i in
        #                    range(self.k)]  # each phi[i] should have shape (b,1)
        #             e_term = self.calc_exp_term(phi, data)
        #             loss = -(sum(phi).mean() - self.eps * e_term)+32*sum([torch.norm(A, p='fro') for A in self.A_matrices])
        #             loss.backward()
        #
        #             if self.params.clip_grads:
        #                 torch.nn.utils.clip_grad_norm_(self.A_matrices, self.params.max_grad_norm)
        #
        #             self.opt_A.step()
        #             l.append(loss.item())
        #
        #         print_plan_overall = False
        #         if print_plan_overall:
        #             phi_all = [self.models[i](X[:, :, i]) for i in range(self.k)]
        #             print(self.calc_exp_term(phi_all, X))
        #
        #         l = np.mean(l)
        #         self.tot_loss.append(-l + self.eps)
        #         epoch_time = timer() - t0
        #         self.times.append(epoch_time)
        #         print(f'finished A_matrices epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')
        #
        #     else:
        #         # train NEMOT this epoch:
        #         for data in zip(*X_b):
        #             self.zero_grad_models()
        #             # data.to(self.device)
        #             for k_ind in range(self.k):
        #                 # k-margin loss
        #                 phi = [self.models[i](data[:, :, i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
        #                 e_term = self.calc_exp_term(phi, data)
        #                 loss = -(sum(phi).mean() - self.eps * e_term)
        #                 if self.params.regularize_pariwise_coupling and self.params.cost_graph != 'full':
        #                     # print('regularize_pariwise_coupling')
        #                     pairwise_reg = self.calc_pairwise_coupling_regularizer(phi, data)
        #                     reg_loss = loss + self.params.regularize_pariwise_coupling_reg * pairwise_reg
        #                     reg_loss.backward()
        #                 else:
        #                     loss.backward()
        #
        #                 if self.params.clip_grads:
        #                     torch.nn.utils.clip_grad_norm_(self.models[k_ind].parameters(), self.params.max_grad_norm)
        #
        #                 self.opt[k_ind].step()
        #                 l.append(loss.item())
        #
        #         print_plan_overall = False
        #         if print_plan_overall:
        #             phi_all = [self.models[i](X[:, :, i]) for i in range(self.k)]
        #             print(self.calc_exp_term(phi_all, X))
        #
        #         l = np.mean(l)
        #         self.tot_loss.append(-l + self.eps)
        #         epoch_time = timer() - t0
        #         self.times.append(epoch_time)
        #         print(f'finished NEMOT epoch {epoch}, loss={-l + self.eps:.5f}, took {epoch_time:.2f} seconds')
        #
        #     print_debug = True
        #     if epoch % 10 == 0 and print_debug and self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
        #         P = self.calc_plan(X)
        #         ot_cost = self.calc_ot_cost(P, X)
        #         print(f'ot_cost {ot_cost}')
        #
        #     if self.params.schedule:
        #         for sched in self.scheduler:
        #             sched.step()
        #             lr = sched.get_last_lr()[0]
        #             # print(f'updated learning rate {lr}')
        #
        #     # print(f'finished epoch {epoch}, loss={l / i}')
        # self.models_to_eval

    def save_results(self,X=None):
        S1 = self.calc_S1(X)
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params,
            # 'plan': plan,
        }
        # if self.params.cost_graph != 'full' and self.params.calc_ot_cost and self.params.cost_graph != 'tree':
        #     plan = self.calc_plan(X)
        #     ###
        #     ot_cost = self.calc_ot_cost(plan,X)
        #     ###
        #     data_to_save['ot_cost'] = ot_cost
        # else:
        #     ot_cost = 0
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss+self.eps,
                       'avg_time': avg_time
                       })
            # if self.params.cost_graph != 'full':
            #     wandb.log(({
            #            'ot_cost': ot_cost
            #            }))

        # if self.params.cost_graph != 'full':
        #     print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds, ot_cost {ot_cost:.5f}')
        # else:
        #     print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')

    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            cost = QuadCostGW(data, self.A_matrices)


        # NOW - BROADCAST!!
        return cost

    def calc_S1(self,X):
        """
        Generalized S1 calculation for a list of tensors [x0, x1, ..., x_{k-1}],
        each of shape (n, d).

        Returns a 1D torch tensor of length (k-1), where result[i] is the
        S1 value computed for the pair (X[i], X[i+1]).
        """
        k = self.k
        if k < 2:
            raise ValueError("Need at least 2 variables to form a pair.")

        results = []

        for i in range(k):
            x = X[i]
            y = X[(i + 1)%k]

            # Ensure x, y are 2D: (n, d)
            if x.dim() != 2 or y.dim() != 2:
                raise ValueError("Each tensor x[i] must be of shape (n, d).")

            n = x.shape[0]

            # 1) Compute norms (squared) along the rows (dim=1)
            #    x_norm_2 and y_norm_2 each of shape (n,)
            x_norm_2 = torch.norm(x, dim=1, p=2) ** 2
            y_norm_2 = torch.norm(y, dim=1, p=2) ** 2

            # Square them again to get x_norm_4, y_norm_4
            x_norm_4 = x_norm_2 ** 2
            y_norm_4 = y_norm_2 ** 2

            # 2) Means along the n dimension
            M2_x = x_norm_2.mean()
            M2_y = y_norm_2.mean()
            M4_x = x_norm_4.mean()
            M4_y = y_norm_4.mean()

            # 3) Average outer products => (d, d) shapes
            sig_x = x.t().matmul(x) / n  # shape (d, d)
            sig_y = y.t().matmul(y) / n  # shape (d, d)

            # 4) Frobenius norms squared
            #    (||sig_x||_F^2, ||sig_y||_F^2)
            #    Using p='fro' in torch.norm, then square it:
            F_x = torch.norm(sig_x, p='fro') ** 2
            F_y = torch.norm(sig_y, p='fro') ** 2

            # 5) S1 formula
            S1_val = (
                    2 * (M4_x + M4_y)
                    + 2 * (M2_x ** 2 + M2_y ** 2)
                    + 4 * (F_x + F_y)
                    - 4 * (M2_x * M2_y)
            )

            # Keep it as a scalar tensor (not converting to Python float)
            results.append(S1_val.unsqueeze(0))

        # Stack into a 1D tensor of length k-1
        return torch.cat(results, dim=0)

class Net_EOT(nn.Module):
    def __init__(self,dim,K,deeper=False):
        super(Net_EOT, self).__init__()
        self.deeper=deeper
        if deeper:
            self.fc1 = nn.Linear(dim, 10 * K)
            self.fc2 = nn.Linear(10 * K, 10 * K)
            self.fc3 = nn.Linear(10 * K, 1)
        else:
            self.fc1 = nn.Linear(dim, K)
            self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        if self.deeper:
            x1 = F.relu(self.fc1(x))
            x11 = F.relu(self.fc2(x1))
            x2 = self.fc3(x11)
        else:
            x1 = F.relu(self.fc1(x))
            x2 = self.fc2(x1)
        return x2


class NE_mot_model(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(NE_mot_model, self).__init__()
        self.dim = dim

        # Taos's N-GW model
        self.fc1 = nn.Linear(dim, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class A_model(nn.Module):
    def __init__(self, d_in, d_out):
        super(A_model, self).__init__()
        self.A = nn.Parameter(torch.full((d_in, d_out), 1e-3))

    def forward(self, x):
        x1, x2 = x
        return x1 @ self.A @ x2.T