"""Run a script whose `import torch` resolves to densecore (prototype of `densecore run`)."""
import runpy, sys, types
import densecore

parallel = types.ModuleType("torch.nn.parallel")
class DistributedDataParallel:  # imported by nanoGPT at top level; only used with DDP
    def __init__(self, *a, **k):
        raise NotImplementedError("distributed training is unsupported")
parallel.DistributedDataParallel = DistributedDataParallel
distributed = types.ModuleType("torch.distributed")
def _no_distributed(*a, **k):
    raise NotImplementedError("distributed training is unsupported")
distributed.init_process_group = distributed.destroy_process_group = _no_distributed

sys.modules.update({"torch": densecore, "torch.nn": densecore.nn, "torch.nn.functional": densecore.functional,
                    "torch.optim": densecore.optim, "torch.optim.lr_scheduler": densecore.optim.lr_scheduler,
                    "torch.utils": densecore.utils, "torch.utils.data": densecore.utils.data,
                    "torch.nn.parallel": parallel, "torch.distributed": distributed})
densecore.nn.parallel, densecore.distributed = parallel, distributed
script, sys.argv = sys.argv[1], sys.argv[1:]
sys.path.insert(0, ".")
runpy.run_path(script, run_name="__main__")
