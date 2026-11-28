import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path


METHODS = ["GA", "OpenAI-ES", "CMA-ES"]
SEARCH_SEEDS, HELDOUT_SEEDS = [11, 23, 37], [101, 211, 307]
GENERATIONS, POPULATION, SIGMA, OPT_SEED = 10, 6, 0.1, 7
DELEGATION = {"authorization_basis": "delegated_project_execution_decision",
 "authorizing_capacity": "delegated_project_authority",
 "personal_human_attestation": False}


def planned_manifest():
    return {"methods": METHODS, "seeds": SEARCH_SEEDS,
      "generations": GENERATIONS, "population": POPULATION,
      "sigma": SIGMA, "seed": OPT_SEED,
      "candidate_budget": 60, "episode_budget": 180}

def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def planned_identity():
    return canonical_hash(planned_manifest())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")

def current_commit(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"],
        cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def deny(code, detail):
    return {"decision": "denied", "authorized": False,
        "dry_run": True, "experiment_launched": False,
        "failures": [{"code": code, "detail": detail}]}

def inspect_authorization(repo_root, authorization_path, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return deny("clock", "Timezone-aware clock required")
    try:
        commit = current_commit(repo_root)
        approval = read_json(authorization_path)
    except Exception as exc:
        return deny("read_error", str(exc))
    required = {"schema_version", "authorized", "issued_utc",
      "expires_utc", "checkout_commit", "experiment_manifest_sha256",
      "max_candidate_evaluations", *DELEGATION}
    if not isinstance(approval, dict) or required - approval.keys():
        return deny("fields", "Authorization fields missing")

    if any(approval[k] != v for k, v in DELEGATION.items()):
        return deny("delegation", "Delegated metadata mismatch")
    if approval["authorized"] is not True or approval["schema_version"] != 1:
        return deny("approval", "Authorization not approved")
    if approval["checkout_commit"] != commit:
        return deny("commit", "Checkout commit mismatch")
    if approval["experiment_manifest_sha256"] != planned_identity():
        return deny("identity", "Planned manifest identity mismatch")
    limit = approval["max_candidate_evaluations"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 183:
        return deny("budget", "Aggregate budget must be at least 183")

    try:
        issued = datetime.fromisoformat(approval["issued_utc"].replace("Z","+00:00"))
        expires = datetime.fromisoformat(approval["expires_utc"].replace("Z","+00:00"))
        if issued.tzinfo is None or expires.tzinfo is None:
            raise ValueError("Timezone required")
    except Exception as exc:
        return deny("timestamps", str(exc))
    now = now.astimezone(timezone.utc)
    if not issued <= now < expires or expires <= issued:
        return deny("window", "Outside authorization window")
    return {"decision": "authorized", "authorized": True,
      "checkout_commit": commit, "experiment_manifest_sha256": planned_identity(),
      "requested_candidate_evaluations": 183,
      "authorized_candidate_evaluations": limit,
      "checked_utc": now.isoformat(), "dry_run": True,
      "experiment_launched": False, "failures": []}
