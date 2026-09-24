import numpy as np


def decompose(scores):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or min(s.shape) < 2 or not np.isfinite(s).all():
        raise ValueError('need a finite query-by-candidate matrix with >=2 rows/columns')
    row_centered = s-s.mean(axis=1, keepdims=True)
    candidate_effect = row_centered.mean(axis=0, keepdims=True)
    interaction = row_centered-candidate_effect
    interaction_energy = float(np.mean(interaction**2))
    candidate_energy = float(np.mean(candidate_effect**2))
    total = interaction_energy+candidate_energy
    
    
    degenerate = total <= 1e-24
    share = 0. if degenerate else interaction_energy/total
    return {'interaction_share': float(share),
            'interaction_energy': interaction_energy,
            'candidate_main_effect_energy': candidate_energy,
            'row_centered_total_energy': total, 'degenerate': bool(degenerate)}


def anchor_ids(n_pool, n_anchors=1024, seed=42):
    return np.random.default_rng(seed).choice(n_pool, min(n_anchors, n_pool), replace=False)


def from_embeddings(query, pool, ids):
    return decompose(np.asarray(query, dtype=np.float64) @
                     np.asarray(pool[ids], dtype=np.float64).T)


def contract_queries(query, clean_train, strength=.75):
    q = np.asarray(query, dtype=np.float64)
    axis = np.asarray(clean_train, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-10:
        raise ValueError('clean train mean has no reliable common direction')
    axis /= norm
    t = float(strength)
    if not 0 <= t <= 1:
        raise ValueError('strength must lie in [0,1]')
    modified = (1-t)*q+t*axis
    modified /= np.maximum(np.linalg.norm(modified, axis=1, keepdims=True), 1e-12)
    return modified.astype(np.float32)


def four_point_rms(scores):
    s = np.asarray(scores, dtype=np.float64)
    nr, nc = len(s)//2, s.shape[1]//2
    delta = (s[0:2*nr:2, 0:2*nc:2]-s[0:2*nr:2, 1:2*nc:2]
             -s[1:2*nr:2, 0:2*nc:2]+s[1:2*nr:2, 1:2*nc:2])
    return float(np.sqrt(np.mean(delta**2)))
