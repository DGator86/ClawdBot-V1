# Re-export common utilities for backward compatibility
# gnosis.utils was originally a single file; now it's a package.
# Import the shared helpers so `from gnosis.utils import ...` still works.
import os as _os
import sys as _sys

# Make the _utils_compat module's names available here
_compat_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "_utils_compat.py")
if _os.path.exists(_compat_path):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("gnosis._utils_compat", _compat_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    drop_future_return_cols = _mod.drop_future_return_cols
    safe_merge_no_truth = _mod.safe_merge_no_truth
    vectorized_abstain_mask = _mod.vectorized_abstain_mask
else:
    # Fallback stubs
    def drop_future_return_cols(df):
        cols = [c for c in df.columns if c.startswith("future_return")]
        return df.drop(columns=cols, errors="ignore") if cols else df

    def safe_merge_no_truth(left, right, on, how="left"):
        right_clean = right[[c for c in right.columns if not c.startswith("future_return")]]
        return left.merge(right_clean, on=on, how=how)

    def vectorized_abstain_mask(s_labels, s_pmax, confidence_floor=0.65):
        import numpy as np
        return np.ones(len(s_labels), dtype=bool)

# Try to import DigitalOceanClient if available
try:
    from .digitalocean_client import DigitalOceanClient
except Exception:
    DigitalOceanClient = None
