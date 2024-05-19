import numpy as np
from utils.data_fns import QuadCost, calc_ent
import copy
import random
from timeit import default_timer as timer
import pickle
import os
import wandb


class MOT_Sinkhorn():
    def __init__(self, params, X, MU):
        self.eps = params.eps
        self.eta = 1/self.eps
        self.k = params.k
        self.mus = MU
        self.n = params.n
        self.tol = 1e-10
        params['cost_graph'] = 'full'
        self.calc_kernel(params['cost_graph'], X)
        self.using_wandb = params['using_wandb']
        self.params=params



    def calc_kernel(self, cost_graph, X):
        """
        calculate the kernel.
        X has shape (n,d,k)
        Kernel shape depends on cost graphical structure
        """
        if cost_graph == 'circle':
            # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
            # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
            K = []
            # for i in range(self.k):
            C = QuadCost(data=X, mod='circle')
            K = [np.exp(-c/self.eps) for c in C]

        elif cost_graph == 'tree':
            # TD
            K=0

        elif cost_graph == 'full':
            # TD - figure out how to generate cost tensor in QuadCost
            C = QuadCost(data=X, mod='full')
            self.cost = C
            K = np.exp(-C/self.eps)

        self.kernel = K

    def solve_sinkhorn(self, method='cyclic'):
        """
        eta is regularization parameter
        tol is the error tolerance for stopping
        method is "greedy", "random", or "cyclic"
        """
        eta = self.eta
        tol = self.tol
        naive = True
        self.tot_loss = []
        self.times = []

        self.phi = [np.zeros(self.n)]*self.k

        assert(method in ['greedy','random','cyclic'])
        t00 = timer()
        curriter = -1
        while True:
            t0 = timer()
            curriter += 1
            if curriter % self.k == 0:
                """ Every k iterations check whether we should exit """
                (worsti,worsterr) = self._sinkhorn_worst_error(naive)

                if worsterr < tol:
                    break
            # if curriter % 1000 == 0:
            if curriter % 10 == 0:
                print('Iteration ',curriter)
                print('worsterr',worsterr)
                loss = self.calc_ot_cost(training=True)
                print('OT cost', loss)
                self.tot_loss = [loss]



            """ Pick scaling index """
            scalei = -1

            if method == 'greedy':
                """ Greedy selection rule """
                (worsti, worsterr) = self._sinkhorn_worst_error(eta, p, naive)
                print('worsterr',worsterr)
                if worsterr < tol:
                    continue
                scalei = worsti

            elif method == 'random':
                """ Random selection rule """
                scalei = random.randint(0,self.k-1)

            elif method == 'cyclic':
                """ Cyclic selection rule """
                scalei = curriter % self.k

            else:
                assert(False)

            """ Rescale weights on scalei """

            if naive:
                mi = self.marginalize_naive(scalei)
            else:
                mi = self.marginalize(scalei)
            ratio = self.mus[scalei] / mi
            self.phi[scalei] = self.phi[scalei] + np.log(ratio) / eta
            # print(f'iteration: {curriter}, took {timer()-t0:.2f} seconds')
            self.times.append(timer() - t0)

        t0 = timer()
        self.phi, self.rankone = self._round_sinkhorn(naive)
        print(f'Rounding step took, took {timer() - t0:.2f} seconds')

        print(f'Finished, n={self.n}, k={self.k}, took {timer() - t00:.2f} seconds')
        return


    def marginalize(self, i):
        """
        Implementation of marginalization method for circle cost.
        """
        # update alpha[i] (:= alpha(i+1,i))
        alpha = self.calc_alpha(i)

        # calculate marginal
        scaled_K = self.phi[i] * self.kernel[i+1]
        scaled_alpha = self.phi[i+1] * alpha
        result = scaled_K * scaled_alpha.T
        mi = result @ np.ones((result.shape[1],))
        return mi

    def calc_alpha(self,i):
        """
        alg calcs alpha resursively
        calculating alpha(i+1,i) starts from alpha(j,i) from j=i-1 going back to j=i+1 via modolu
        :param i:
        :return:
        """
        for j in range(1, self.k):
            ind = (i-j)%self.k
            if j == 1:
                alpha = self.kernel[i]
            else:
                alpha = self.kernel[ind] @ (self.phi((ind+1)@self.k) * alpha)
        return alpha


    def marginalize_naive(self, i, outplan=False):
        # assert(False)
        """
        NAIVE, BRUTE FORCE IMPLEMENTATION OF THE FOLLOWING:
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-\eta C].
        Let d_i = \exp[\eta p_i].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_i(P).
        """
        eta = self.eta
        scaled_cost_tensor = copy.deepcopy(self.kernel)

        for scale_axis in range(self.k):
            dim_array = np.ones((1,scaled_cost_tensor.ndim),int).ravel()
            dim_array[scale_axis] = -1
            p_scale_reshaped = self.phi[scale_axis].reshape(dim_array)
            p_scaling = np.exp(eta * p_scale_reshaped)
            # print(p_scaling.shape)
            scaled_cost_tensor = scaled_cost_tensor * p_scaling
            # print(scaled_cost_tensor)
        if outplan:
            return scaled_cost_tensor
        mi = np.apply_over_axes(np.sum, scaled_cost_tensor, [j for j in range(self.k) if j != i])
        mi = mi.flatten()
        return mi



    def _round_sinkhorn(self, naive=False):
        """ Round sinkhorn solution with a rank-one perturbation """
        p = copy.deepcopy(self.phi)
        eta = self.eta
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                mi = self.marginalize(i)
            badratio = self.mus[i] / mi
            minbadratio = np.minimum(1, badratio)
            p[i] = p[i] + np.log(minbadratio) / eta

        rankone = []
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                mi = self.marginalize(i)
            erri = self.mus[i] - mi
            assert(np.all(erri >= -1e-10))
            erri = np.maximum(0,erri)
            if i > 0 and np.sum(np.abs(erri)) > 1e-8:
                rankone.append(erri /  np.sum(np.abs(erri)))
            else:
                rankone.append(erri)

        return p, rankone


    def _sinkhorn_worst_error(self,naive=False):
        """
        Compute the worst error of any marginal. Used for the termination
        condition of solve_sinkhorn.
        """
        worsti = -1
        worsterr = -1
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                mi = self.marginalize(i)
            erri = np.sum(np.abs(mi - self.mus[i]))
            if erri > worsterr:
                worsti = i
                worsterr = erri
        return (worsti, worsterr)

    def calc_ot_cost(self, training=False):
        P = self.calc_plan(training)
        KL = sum([calc_ent(mu) for mu in self.mus]) - calc_ent(P)
        return np.sum(P*self.cost) + self.eps*KL


    def calc_plan(self, training):
        P = self.marginalize_naive(0, outplan=True)
        if training:
            return P
        for index in  range(1,len(self.rankone)):
            if index == 1:
                rankone = np.tensordot(self.rankone[index-1], self.rankone[index], axes=0)
            else:
                rankone = np.tensordot(rankone, self.rankone[index], axes=0)
        return P + rankone
        #     #### OLD:
        #     # Create a shape of length k with 1s except at the index position
        #     shape = [1] * self.k
        #     shape[index] = -1
        #     reshapred_rankone.append(self.rankone[index].reshape(shape))
        #     P = P * (vec.reshape(shape))
        # R = sum(reshapred_rankone)
        # return P*np.exp(-self.kernel/self.eps)+R

    def save_results(self):
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params
            }
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss, 'avg_time': avg_time})


        print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')
