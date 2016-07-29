""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo, gauss_fit_joint_prior
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList


LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            policy_prior = self._hyperparams['policy_prior']
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy()

        # Step adjustment (also updates policy fit)
        self._update_step_size()

        # C-step
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._update_policy_fit(m, init=True)
            # TODO: this doesn't make sense with clusters
            self._eval_cost(m)
            # Adjust step size relative to the previous iteration.
            if self.iteration_count > 0:
                self._stepadjust(m)

    def _update_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def _update_policy_fit(self, m, init=False):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
            init: Whether this is the initial fitting of the policy.
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        pol_mu, pol_sig = self.policy_opt.prob(samples.get_obs().copy())[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        if init:
            samples = SampleList(self.cur[m].sample_list)
            mode = self._hyperparams['policy_sample_mode']
        else:
            samples = SampleList([])
            mode = 'add' # Don't replace with empty samples
        policy_prior.update(samples, self.policy_opt, mode)

        # Collapse policy covariances. This is not really correct, but
        # it works fine so long as the policy covariance doesn't depend
        # on state.
        pol_sig = np.mean(pol_sig, axis=0)
        # Estimate the policy linearization at each time step.
        for t in range(T):
            # Assemble diagonal weights matrix and data.
            dwts = (1.0 / N) * np.ones(N)
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate((Ts, Ps), axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = policy_prior.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # On the first time step, always slightly regularize covariance.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8 * np.eye(dX)
            # Perform computation.
            pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0,
                                                        dwts, dX, dU, sig_reg)
            pol_S += pol_sig[t, :, :]
            pol_info.pol_K[t, :, :], pol_info.pol_k[t, :] = pol_K, pol_k
            pol_info.pol_S[t, :, :], pol_info.chol_pol_S[t, :, :] = \
                    pol_S, sp.linalg.cholesky(pol_S)

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        Algorithm._advance_iteration_variables(self)
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                    self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Get the necessary linearizations
        prev_nn = self.prev[m].pol_info.traj_distr()
        cur_nn = self.cur[m].pol_info.traj_distr()
        cur_traj_distr = self.cur[m].traj_distr

        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        prev_laplace_cost = self.traj_opt.estimate_cost(
            prev_nn, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the prev samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_cost = self.traj_opt.estimate_cost(
            cur_traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        if (self._hyperparams['step_rule'] == 'classic'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_traj_distr, self.cur[m].traj_info
            )
        elif (self._hyperparams['step_rule'] == 'global'):
            new_actual_laplace_cost = self.traj_opt.estimate_cost(
                cur_nn, self.cur[m].traj_info
            )
        else:
            raise NotImplementedError

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual costective values based on the samples.
        prev_mc_cost = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_cost = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, prev_mc_cost, new_mc_cost)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(prev_laplace_cost) - \
                np.sum(new_predicted_laplace_cost)
        actual_impr = np.sum(prev_laplace_cost) - \
                np.sum(new_actual_laplace_cost)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(prev_laplace_cost), prev_mc_cost)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_cost), new_mc_cost)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_cost), new_mc_cost)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        pol_info = self.cur[m].pol_info
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta)

        return fCm, fcv
