"""The contract a forward simulator must satisfy to be driven by the base ensemble."""

from typing import Protocol, runtime_checkable

__all__ = ["ForwardSimulator"]


@runtime_checkable
class ForwardSimulator(Protocol):
    """What :meth:`ensemble.ensemble.BaseEnsemble.calc_prediction` requires of a simulator.

    Two members are required, and they are all that ``isinstance(sim,
    ForwardSimulator)`` checks:

    ``input_dict``
        The parsed simulator section of the config. The ensemble reads
        ``parallel`` (local workers, default 1) and ``hpc`` from it.
    ``run_fwd_sim(state, member_index)``
        Run one realisation. ``state`` maps each state variable to that
        member's values; ``member_index`` is the member's position in the
        ensemble. Return one of

        - a list with one dict per report point, keyed by data type,
        - a ``pandas.DataFrame`` with report points as index and data types
          as columns,
        - ``False`` when the run failed, so the member can be replaced, or
        - ``(output, adjoint)`` when ``compute_adjoints`` is true. The adjoint
          is a frame with the sensitivity of each observed value to the
          state, or, from a simulator in weighted-adjoint mode, a
          ``pandas.Series`` with one gradient per parameter name.

    Members the ensemble looks for with ``hasattr``/``getattr`` and uses only
    when present:

    ``setup_fwd_run(level=...)``
        Called once before each prediction, with the fidelity level.
    ``true_order``
        ``[index_name, index_values]`` used to index the returned records.
    ``datatype``
        Fallback column filter when the observed data has no columns yet.
    ``compute_adjoints``
        Whether ``run_fwd_sim`` returns ``(output, adjoint)``. Default False.
    ``adjoint_weights``
        Weighted-adjoint mode only: a callable ``(output, member_index)``
        that the simulator calls with each member's output to get the
        weights :math:`w`, as a frame laid out like the output. It then
        returns the gradient of :math:`\\sum_i w_i d_i` per parameter. The
        caller sets it; :class:`pipt.ensembles.DataMisfitWeights` gives the
        weights of the data misfit.

    The ensemble also *assigns* ``redund_sim`` (a backup simulator, or
    ``None``) onto the simulator when it is constructed. The analytical models
    in :mod:`simulator` are the smallest complete examples.
    """

    input_dict: dict

    def run_fwd_sim(self, state, member_index, *args, **kwargs):
        """Run one member; the class docstring lists the accepted return values."""
