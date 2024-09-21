import numpy as np
import torch.nn as nn
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
from utils.data_fns import QuadCost
import pickle
import os


class MOT_NE_alg():
    def __init__(self, params, device):
        self.models = []
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
            if epoch%10==0 and print_debug and self.params.cost_graph != 'full':
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
        if self.params.cost_graph != 'full':
            plan = self.calc_plan(X)
            ###
            ot_cost = self.calc_ot_cost(plan,X)
            ###
            data_to_save['ot_cost'] = ot_cost
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
                cost = QuadCost(data, mod=self.cost_graph)
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
