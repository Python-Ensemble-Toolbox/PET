"""Stochastic Monte-Carlo optimization compatible with OptimizerBase."""

import numpy as np

from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport
import popt.optimization_methods.subroutines.optimizers as opt

__author__ = ""
__all__ = ["SmcOpt"]


class SmcOpt(OptimizerBase):
    """Sequential Monte-Carlo optimizer with resampling and backtracking."""

    NAME = "SmcOpt"

    def __init__(self, x0, fun, sens=None, args=(), bounds=None, callback=None, **options):
        """
        Parameters
        ----------
        x0 : ndarray
            Initial state

        fun : callable
            objective function

        sens : callable
            Ensemble sensitivity function

        args : tuple
            Initial covariance tuple where ``args[0]`` is the covariance matrix used for sampling.

        bounds : list, optional
            (min, max) pairs for each element in x. None is used to specify no bound.

        callback : callable, optional
            Callback invoked after successful updates.

        options : dict
            SmcOpt configuration, plus everything :class:`OptimizerBase` takes
            (``transform`` is forced off: SmcOpt works in physical coordinates).

            - tol: convergence tolerance for the objective function (default 1e-6). Also used as ``ftol`` when given.
            - alpha: weight between previous and new step (default 0.1)
            - alpha_maxiter: maximum number of backtracking trials (default 5)
            - resample: number indicating how many times resampling is tried if no improvement is found
            - cov_factor: factor used to shrink the covariance for each resampling trial (default 0.5)
            - inflation_factor: term used to weight down prior influence (default 1.0)
            - survival_factor: fraction of surviving samples (clipped to [0.1, 1.0])
            - best_func: best objective value seen before this run (default: the initial objective)
            - savefolder/save_folder: folder used when saveit is true (default './')
        """
        if sens is None or not callable(sens):
            raise ValueError("SmcOpt requires a callable sensitivity function 'sens'.")
        if len(args) < 1:
            raise ValueError("SmcOpt requires initial covariance as args[0].")

        # SmcOpt historically operates in physical coordinates.
        options = {**options, "transform": False}
        super().__init__(x0, fun, jac=None, hess=None, args=(), bounds=bounds, callback=callback, **options)

        self.sens = sens

        # SmcOpt controls
        self.obj_func_tol = options.get("tol", 1e-6)
        self.ftol = options.get("tol", options.get("ftol", self.ftol))
        self.alpha = options.get("alpha", 0.1)
        self.alpha_iter_max = options.get("alpha_maxiter", 5)
        self.max_resample = options.get("resample", 0)
        self.cov_factor = options.get("cov_factor", 0.5)
        self.inflation_factor = options.get("inflation_factor", 1.0)
        self.survival_factor = float(np.clip(options.get("survival_factor", 1.0), 0.1, 1.0))
        self.savefolder = options.get("savefolder", options.get("save_folder", "./"))
        self.alpha_iter = 0

        # Dynamic SMC state
        self.cov = np.asarray(args[0], dtype=float)
        self.best_state = None
        self.best_func = None  # set when the run starts, from `best_func` or the initial objective
        self.sens_njev = 0

        self.optimizer = opt.GradientDescent(self.alpha, 0.0)

    @property
    def obj_func_values(self):
        """Legacy alias for ``fk``."""
        return self.fk

    def _start(self):
        # The best value seen so far starts at the initial objective, which
        # the first log row and result already show.
        if self.fk is None:
            self.fk = self.fun(self.xk)
        self.best_func = float(np.mean(self.options.get("best_func", self.fk)))
        super()._start()

    def update_step(self) -> StepReport:
        """Perform one SMC update step with backtracking and optional resampling."""
        self.optimizer.restore_parameters()
        resampling_iter = 0
        inflate = 2.0 * (self.inflation_factor + self.iteration)

        while resampling_iter <= self.max_resample:
            shrink = self.cov_factor ** resampling_iter
            self.optimizer.apply_backtracking(np.sqrt(self.cov_factor) ** resampling_iter)

            sens_matrix, self.best_state, best_func_tmp = self.sens(
                self.xk,
                inflate,
                shrink * self.cov,
                self.survival_factor,
                epf=self.epf,
            )
            self.sens_njev += 1

            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state = self.optimizer.apply_smc_update(self.xk, sens_matrix, iter=self.iteration)
                new_state = self.bound_handler.project_to_bounds(new_state)

                new_func_values = self.fun(new_state)

                improved_objective = np.mean(self.fk) - np.mean(new_func_values) > self.obj_func_tol
                improved_best = (self.best_func - best_func_tmp) > self.obj_func_tol
                if improved_objective or improved_best:
                    self._accept_step(new_state, new_func_values, best_func_tmp, improved_best)
                    return StepReport(True)

                if self.alpha_iter < self.alpha_iter_max:
                    self.optimizer.apply_backtracking()
                    self.alpha_iter += 1
                else:
                    break

            if (resampling_iter < self.max_resample) and (np.mean(new_func_values) > np.mean(self.fk)):
                resampling_iter += 1
                self.optimizer.restore_parameters()
                continue

            return StepReport(False, "SmcOpt failed to find an improving step.")

        return StepReport(False, "SmcOpt exhausted all resampling attempts.")

    def _accept_step(self, new_state, new_func_values, best_func_tmp, improved_best):
        self._commit_step(new_state, new_func_values)
        if improved_best:
            self.best_func = float(best_func_tmp)
        self.optimizer.restore_parameters()

    def _update_optimize_result(self):
        result = super()._update_optimize_result()
        result["fun"] = float(np.mean(self.fk))
        result["njev"] = self.sens_njev
        result["best_func"] = self.best_func
        return result

    def _get_restart_state(self) -> dict:
        return {
            "cov": self.cov,
            "best_state": self.best_state,
            "best_func": self.best_func,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "obj_func_tol": self.obj_func_tol,
            "sens_njev": self.sens_njev,
            "optimizer_state": dict(self.optimizer.__dict__),
        }

    def _set_restart_state(self, state: dict) -> None:
        self.cov = state.get("cov", self.cov)
        self.best_state = state.get("best_state", self.best_state)
        self.best_func = state.get("best_func", self.best_func)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        self.obj_func_tol = state.get("obj_func_tol", self.obj_func_tol)
        self.sens_njev = state.get("sens_njev", self.sens_njev)
        self.optimizer.__dict__.update(state.get("optimizer_state", {}))

    def log_columns(self) -> dict:
        """The row of the iteration log: iteration, backtracking attempts, objective, best objective seen, step size."""
        return {
            "iter.": self.iteration,
            "alpha_iter": self.alpha_iter,
            "obj_func": float(np.mean(self.fk)),
            "best_func": float(self.best_func),
            "step-size": self.alpha,
        }
