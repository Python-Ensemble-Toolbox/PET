"""Checkpoint/restart machinery shared by PIPT and POPT.

Both the optimization schemes in :mod:`popt.optimization_methods` and the
assimilation schemes in :mod:`pipt.update_schemes` are long-running iterative
algorithms that need to survive interruption. The persistence logic is
identical for both, so it lives here rather than being duplicated per package.

A host class must provide:

- ``restart`` (bool): whether a checkpoint should be restored on startup.
- ``restart_file`` (str): path to the checkpoint file.
- ``logger``: a :class:`ensemble.logger.PetLogger` or ``None``.
- ``_get_base_restart_state()`` / ``_set_base_restart_state(state)``: serialize
  and restore the state owned by the algorithm base class.
- ``_get_restart_state()`` / ``_set_restart_state(state)``: the same, for state
  owned by the concrete subclass. Both default to storing nothing, so only a
  host that carries its own iteration state needs to implement them.

Checkpoints record the writing class, so a file written by one algorithm cannot
silently be loaded into another.
"""

import os
import pickle

import numpy as np

__all__ = ["RestartMixin"]


class RestartMixin:
    """Reusable checkpoint and restart functionality for iterative algorithms."""

    RESTART_VERSION = 1

    def _get_restart_state(self) -> dict:
        """Serialize state owned by the concrete algorithm. Override as needed.

        Defaulted here so a host with nothing of its own to checkpoint -- every
        PIPT scheme, as it happens -- inherits the pair rather than declaring
        two empty methods to satisfy the protocol.
        """
        return {}

    def _set_restart_state(self, state: dict) -> None:
        """Restore state owned by the concrete algorithm. Override as needed."""

    def save_restart(self):
        """Save the current optimizer state to a restart file."""
        payload = self._build_restart_payload()
        self._write_restart_payload(payload)

    def load_restart(self):
        """Restore optimizer state from a restart file."""
        payload = self._read_restart_payload()
        self._restore_from_restart_payload(payload)
        self._restart_loaded = True
        if self.logger:
            self.logger(
                f"Loaded restart checkpoint from "
                f"'{self.restart_file}'"
            )

    def clear_restart(self):
        """Delete the restart file if it exists."""
        if self._restart_exists():
            os.remove(self.restart_file)

    # ------------------------------------------------------------------
    # Restart lifecycle
    # ------------------------------------------------------------------
    def _maybe_restore_restart(self) -> bool:
        """Restore a checkpoint if restart is enabled."""
        if not self.restart or not self._restart_exists():
            return False
        self.load_restart()
        return True

    def _restart_exists(self) -> bool:
        """Return True if a restart file exists."""
        return os.path.exists(self.restart_file)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _write_restart_payload(self, payload) -> None:
        """Atomically write a restart payload to disk."""
        restart_dir = os.path.dirname(self.restart_file)
        if restart_dir:
            os.makedirs(restart_dir, exist_ok=True)

        tmp_path = f"{self.restart_file}.tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp_path, self.restart_file)

    def _read_restart_payload(self) -> dict:
        """Read a restart payload from disk."""
        with open(self.restart_file, "rb") as handle:
            return pickle.load(handle)

    # ------------------------------------------------------------------
    # Payload construction and restoration
    # ------------------------------------------------------------------
    def _build_restart_payload(self) -> dict:
        """Create a serializable restart payload."""
        return {
            "version": self.RESTART_VERSION,
            "module": type(self).__module__,
            "class_name": type(self).__name__,
            "random_state": np.random.get_state(),
            "base_state": self._get_base_restart_state(),
            "subclass_state": self._get_restart_state(),
        }

    def _restore_from_restart_payload(self, payload) -> None:
        """Restore optimizer state from a payload."""
        self._validate_restart_payload(payload)
        self._set_base_restart_state(
            payload["base_state"]
        )
        self._set_restart_state(
            payload.get("subclass_state", {})
        )
        rng_state = payload.get("random_state")
        if rng_state is not None:
            np.random.set_state(rng_state)

    def _validate_restart_payload(self, payload) -> None:
        """Validate restart payload compatibility."""

        version = payload.get("version")
        module = payload.get("module")
        class_name = payload.get("class_name")

        if version != self.RESTART_VERSION:
            raise RuntimeError(
                f"Restart file '{self.restart_file}' "
                f"has unsupported version {version}."
            )

        if (
            module != type(self).__module__
            or class_name != type(self).__name__
        ):
            raise RuntimeError(
                f"Restart file '{self.restart_file}' "
                f"does not match {type(self).__name__}."
            )
