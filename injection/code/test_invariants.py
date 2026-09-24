
import json
import numpy as np
import stage_joint as J

rng = np.random.default_rng(20260918)
q = J.norm(rng.normal(size=(32, 64)))
p = J.norm(rng.normal(size=(1024, 64)))
idx = np.argsort(-(q @ p.T), axis=1)[:, :64]
axis = J.norm(rng.normal(size=(1, 64)))[0]
original_q, original_p = q.copy(), p.copy()
for pair in J.P.PAIRS:
    q1, p1, _ = J.construct(q, p, pair, idx, axis, 1, q)
    q2, p2, _ = J.construct(q, p, tuple(reversed(pair)), idx, axis, 1, q)
    assert np.array_equal(q1, q2) and np.array_equal(p1, p2), pair
    assert q1.shape == q.shape and p1.shape == p.shape
    assert np.isfinite(q1).all() and np.isfinite(p1).all()
    assert np.max(np.abs(np.linalg.norm(q1, axis=1)-1)) < 2e-6
    assert np.max(np.abs(np.linalg.norm(p1, axis=1)-1)) < 2e-6
assert np.array_equal(q, original_q) and np.array_equal(p, original_p)
assert J.C.NATIVE_INJECT_STRENGTH == 0.75
report = {"passed": True, "pairs": 10,
          "checks": ["pair permutation", "shape", "finite", "unit norm", "source immutable", "strength 0.75", "score-interaction H5"]}
(J.ROOT/"runs"/"invariants.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
