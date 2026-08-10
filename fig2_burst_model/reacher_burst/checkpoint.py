"""
Checkpoint I/O, backward-compatible with the original ``burst_vprob_reacher.py``.

Pickles written by that script reference the top-level modules ``goal`` and
``optimizer``.  :class:`_LegacyUnpickler` remaps those names to the packaged
modules at load time — without touching ``sys.modules``, so importing this
package never shadows the repository's own ``goal.py`` / ``optimizer.py``.

Checkpoints written *by this package* reference ``reacher_burst.network`` /
``reacher_burst.optimizer``; the payload layout (``network``, ``Jout_sub``,
``adam_sub``) is unchanged.
"""
import pickle
from pathlib import Path

import numpy as np

from . import network as _network
from . import optimizer as _optimizer
from .agent import BurstAgent

_LEGACY_MODULES = {'goal': _network, 'optimizer': _optimizer}


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        mod = _LEGACY_MODULES.get(module)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
        return super().find_class(module, name)


def save(path, agent):
    """Write ``{'network', 'Jout_sub', 'adam_sub'}`` (original payload layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump({'network': agent.network,
                     'Jout_sub': agent.Jout_sub,
                     'adam_sub': agent.adam_sub}, fh)
    return path


def load(path):
    """Raw checkpoint dict, legacy module names remapped."""
    with open(path, 'rb') as fh:
        return _LegacyUnpickler(fh).load()


def load_agent(path, mu_offset, out_dim=None, num_g=None, n_b=None):
    """Rebuild a :class:`~reacher_burst.agent.BurstAgent` from a checkpoint."""
    from . import config as C

    st = load(path)
    network, Jout_sub, adam_sub = st['network'], st['Jout_sub'], st['adam_sub']

    out_dim = out_dim if out_dim is not None else Jout_sub[0].shape[0]
    num_g = num_g if num_g is not None else len(Jout_sub)
    n_b = n_b if n_b is not None else C.N_B

    network.gamma = getattr(network, 'gamma', C.GAMMA)
    network.out_dim = out_dim
    network.num_g = num_g
    return BurstAgent(network, Jout_sub, adam_sub, mu_offset, out_dim, num_g, n_b)


def load_learning_curve(path):
    """Load the saved ``R_AVG`` trajectory (avg reward every 100 episodes)."""
    path = Path(path)
    return np.load(path) if path.exists() else None
