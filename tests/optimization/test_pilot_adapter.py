import pytest
from lawless.optimization.pilot_adapter import PilotAdapter, PilotTimeout
def test_expired_deadline_raises():
    adapter = PilotAdapter(deadline=0.0)
    with pytest.raises(PilotTimeout):
        adapter.check()

def test_max_steps_fixed():
    assert PilotAdapter(deadline=float("inf")).max_steps == 2400

def test_timeout_type():
    assert issubclass(PilotTimeout, TimeoutError)
