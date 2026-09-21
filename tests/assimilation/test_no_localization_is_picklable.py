"""The stand-in for 'no localization' must survive pickling, because the
ensemble that holds it is pickled by ``emergency_dump`` and by the restart
file -- exactly when a run has crashed."""

import pickle

from pipt.ensembles.ensemble_base import NoLocalization


def test_no_localization_round_trips_through_pickle():
    restored = pickle.loads(pickle.dumps(NoLocalization()))
    assert restored.name is None
