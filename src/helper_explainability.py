import torch
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable


def getSimpleInput(val: float, criteria_nr: int) -> torch.FloatTensor:
    return torch.FloatTensor([[val] * criteria_nr]).view(1, 1, criteria_nr).cpu()


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


# For some reason which is completely beyond me, inp is necessary
# - seemingly to facilitate some hook interactions
def append_output(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = []
    if not hasattr(hook, "name"):
        hook.name = mod.__class__.__name__
    data = hook.stats
    data.append(outp.data)


def plot_marginal_functions(model, criteria_names: Iterable[str], no_criteria: int) -> None:
    hook = Hook(model.method.criterionLayerCombine, append_output)
    xs = []
    with torch.no_grad():
        for i in range(201):
            val = i / 200.0
            x = getSimpleInput(val, no_criteria)
            xs.append(val)
            model(x)

    outs = np.array(torch.stack(hook.stats)[:, 0].detach().cpu())
    outs = outs * model.method.sum_layer.weight.detach().numpy()[0]
    outs = outs[::3] - outs[::3][0]
    outs = outs / outs[-1].sum()
    for i in range(no_criteria):
        plt.plot(xs, outs[:, i], color="black")
        plt.ylabel(f"marginal value $u_{i+1}$: {criteria_names[i]}", fontsize=14)
        plt.xlabel(f"performance $g_{i+1}(a_i)$", fontsize=14)
        plt.show()


if __name__ == "__main__":
    pass
