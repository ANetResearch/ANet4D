import numpy as np
import pytest

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.traffic_admission import DepartureReservations


def gate():
    size = np.array([4.0, 2.0, 1.5])
    return DepartureReservations(
        {
            "through": (route_arrays([[0, 0], [100, 0]], [0, 0]), size),
            "merge": (route_arrays([[20, 3], [30, 0], [100, 0]], [0, 0, 0]), size),
            "cross": (route_arrays([[50, -20], [50, 20]], [np.pi / 2, np.pi / 2]), size),
            "other_street": (route_arrays([[0, 100], [100, 100]], [0, 0]), size),
        }
    )


def test_merge_waits_in_berth_until_actual_release_independent_street_runs():
    admission = gate()
    assert admission.request("through", 0.0)
    assert not admission.request("merge", 2.0)
    assert admission.request("other_street", 2.0)
    # A clock advance cannot release the earlier car or permit a merge.
    assert not admission.request("merge", 10000.0)
    admission.release("through")
    assert admission.request("merge", 10000.0)
    assert admission.active == {"merge", "other_street"}


def test_conflicting_waiters_keep_fifo_even_when_backend_order_changes():
    admission = gate()
    assert admission.request("through", 0.0)
    assert not admission.request("merge", 1.0)
    assert not admission.request("cross", 2.0)
    admission.release("through")
    assert not admission.request("cross", 3.0)
    assert admission.request("merge", 3.0)
    admission.release("merge")
    assert admission.request("cross", 4.0)
    with pytest.raises(ValueError):
        admission.request("cross", 3.0)
    with pytest.raises(ValueError):
        admission.request("unknown", 5.0)
    with pytest.raises(ValueError):
        admission.release("through")
