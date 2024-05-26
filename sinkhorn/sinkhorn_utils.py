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
        self.params = params
        # params['cost_graph'] = 'full'
        self.calc_kernel(params['cost_graph'], X)
        self.using_wandb = params['using_wandb']




    def calc_kernel(self, cost_graph, X):
        """
        calculate the kernel.
        X has shape (n,d,k)
        Kernel shape depends on cost graphical structure
        """
        if self.params['alg'] == 'sinkhorn_mot':
            if cost_graph == 'circle':
                # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
                # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
                # for i in range(self.k):
                C = QuadCost(data=X, mod='circle')
                K = [np.exp(-c/self.eps) for c in C]


            elif cost_graph == 'full':
                C = QuadCost(data=X, mod='full')
                self.cost = C
                K = np.exp(-C/self.eps)

            elif cost_graph == 'tree':
                # TD
                K=0

        elif self.params['alg'] == 'sinkhorn_gw':
            if cost_graph == 'circle':
                #TD!!!
                # calculate consecutive couples, Each Ki is a matrix Ki = exp(-Ci/eps)
                # Ci = cost(x_i,x_{i+1}), therefore has a shape (n x n)
                C = QuadCost(data=X, mod='circle')
                K = [np.exp(-c / self.eps) for c in C]

            elif cost_graph == 'tree':
                # TD
                K = 0

            elif cost_graph == 'full':
                C = QuadCost(data=X, mod='full')
                self.cost = C
                K = np.exp(-C / self.eps)


        self.kernel = K

    def solve_sinkhorn(self, method='cyclic'):
        """
        eta is regularization parameter
        tol is the error tolerance for stopping
        method is "greedy", "random", or "cyclic"
        """
        eta = self.eta
        tol = self.tol
        naive = self.params.cost_graph == 'full'
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
                mi = self.marginalize_circle(scalei)

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

    def marginalize_circle_d(self, margidx):
        """
        Adaptation of the Euler flow implementation of marginalization under a circle cost
        """

        n = self.n
        eta = self.eta
        """ Compute transition matrices  margidx --> k ---> margidx"""
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        # trying cyclic index implementation

        trans_m = np.eye(n)
        for i in range(self.k-1):
            """
            starting from margidx+1, we calculate the transition
            mdx+1->mdx+2->...k-1->k->0->mdx-1
            """
            # idx = (margidx+i+1)%self.k
            idx = (margidx+i+1)%self.k
            currcolscaling = np.diag(np.exp(eta * self.phi[idx]))
            trans_m = trans_m @ currcolscaling
            trans_m = trans_m @ self.kernel[idx]
        # scaled = np.diag(trans_m)
        scaled = np.diag(trans_m) * np.exp(eta * self.phi[margidx])

        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        # ### adapt Altschuler's:
        # trans1 = np.eye(n)
        # transk = np.eye(n)
        #
        # for i in range(margidx):
        #     currcolscaling = np.diag(np.exp(eta * self.phi[i]))
        #     trans1 = trans1 @ currcolscaling
        #     trans1 = trans1 @ self.kernel[i]
        #
        # for i in range(margidx+1, self.k):
        #     transk = transk @ self.kernel[i]
        #     currcolscaling = np.diag(np.exp(eta * self.phi[i]))
        #     transk = transk @ currcolscaling
        #
        # # NOT SURE WE NEED THIS:
        # # transk1 = transk @ self.kernel[self.k-1]
        # transk1 = transk
        # # MAYBE ADD MULT BY MARGIDX KERNEL?
        # notscaled = np.diag(transk1 @ trans1)
        #
        # scaled = notscaled
        # # scaled = notscaled * np.exp(eta * self.phi[margidx])
        return scaled

    def marginalize_circle(self, margidx):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i] for all i \in [k].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_{margidx}(P).
        """

        n = self.n
        reg_cost = self.kernel
        # reg_cost = [self.kernel[0]]*self.k
        reg_loop_cost = self.kernel[self.k-1]
        eta = self.eta
        p = self.phi

        """ Compute transition matrices 1 --> margidx, margidx --> k """
        trans1 = np.eye(n)
        transk = np.eye(n)

        # trans1[i,j] is the mass of cost of transitioning from i at time 1 to j at time margidx
        # include the potentials in [1,margidx)
        for i in range(margidx):
            currcolscaling = np.diag(np.exp(eta * p[i]))
            trans1 = trans1 @ currcolscaling
            trans1 = trans1 @ reg_cost[i]

        # transk[i,j] is the mass of the cost of transitioning from i at time margidx to j at time k
        # include the potentials in (margidx,k]
        for i in range(margidx + 1, self.k):
            transk = transk @ reg_cost[i]
            currcolscaling = np.diag(np.exp(eta * p[i]))
            transk = transk @ currcolscaling

        # print('trans1',trans1)
        # print('transk',transk)

        # transk1[i,j] is the mass of the cost of transitioning from i at time margidx to j at time 1 (going through time k and looping back to time 1)
        # includes the potentials in (margidx,k]

        # transk1 = transk @ reg_loop_cost
        transk1 = self.kernel[margidx]@transk

        # For each fixed l, compute trans1[:,l] \dot transk1[l,:].
        # This marginalizes over time 1, but still does not compute the potentials at margidx.
        notscaled = np.diag(transk1 @ trans1)

        scaled = notscaled * np.exp(eta * p[margidx])
        # comparison = self.marginalize_naive(eta, p, margidx)
        # assert(np.all(np.isclose(scaled, comparison)))

        return scaled


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
                mi = self.marginalize_circle(i)
            badratio = self.mus[i] / mi
            minbadratio = np.minimum(1, badratio)
            p[i] = p[i] + np.log(minbadratio) / eta

        rankone = []
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(i)
            else:
                mi = self.marginalize_circle(i)
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
                mi = self.marginalize_circle(i)
            erri = np.sum(np.abs(mi - self.mus[i]))
            if erri > worsterr:
                worsti = i
                worsterr = erri
        return (worsti, worsterr)

    def calc_ot_cost(self, training=False):
        """
        currently implemeneted for full cost structures
        :param training:
        :return:
        """
        if self.params.cost_graph == 'circle':
            return 0
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
