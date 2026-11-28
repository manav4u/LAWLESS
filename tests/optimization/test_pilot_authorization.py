from lawless.optimization.pilot_authorization import (
    planned_manifest, planned_identity, inspect_authorization)
def test_planned_manifest_budget():
    m = planned_manifest()
    assert m["methods"] == ["GA", "OpenAI-ES", "CMA-ES"]
    assert m["seeds"] == [11, 23, 37]
    assert m["candidate_budget"] == 60
    assert m["episode_budget"] == 180

def test_identity_deterministic():
    assert planned_identity() == planned_identity()
    assert len(planned_identity()) == 64


def test_planned_manifest_exact():
    m = planned_manifest()
    assert m == {"methods": ["GA", "OpenAI-ES", "CMA-ES"],
        "seeds": [11, 23, 37], "generations": 10,
        "population": 6, "sigma": 0.1, "seed": 7,
        "candidate_budget": 60, "episode_budget": 180}

def test_authorization_denied_without_file(tmp_path):
    result = inspect_authorization(tmp_path, tmp_path / "missing.json")
    assert result["authorized"] is False
    assert result["experiment_launched"] is False

def test_authorization_module_is_inspection_only():
    import lawless.optimization.pilot_authorization as module
    assert not hasattr(module, "run_pilot")
    assert not hasattr(module, "launch_experiment")
