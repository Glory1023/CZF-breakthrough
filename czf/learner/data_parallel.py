import operator
import torch
import warnings
from itertools import chain
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch._utils import (_get_all_device_indices, _get_available_device_type,
                          _get_device_index, _get_devices_properties)
# parallel_apply
import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper


def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(
                imbalance_warn.format(device_ids[min_pos],
                                      device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallelWrapper(torch.nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(
            map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)
        self.v_supp = self.module.v_supp
        self.r_supp = self.module.r_supp

    def parallel_forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_forward_representation(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.forward_representation(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.forward_representation(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_representation(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_forward_dynamics(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.forward_dynamics(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.forward_dynamics(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_dynamics(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def parallel_apply_representation(self,
                                      modules,
                                      inputs,
                                      kwargs_tup=None,
                                      devices=None):
        assert len(modules) == len(inputs)
        if kwargs_tup is not None:
            assert len(modules) == len(kwargs_tup)
        else:
            kwargs_tup = ({}, ) * len(modules)
        if devices is not None:
            assert len(modules) == len(devices)
        else:
            devices = [None] * len(modules)
        devices = list(map(lambda x: _get_device_index(x, True), devices))
        lock = threading.Lock()
        results = {}
        grad_enabled, autocast_enabled = torch.is_grad_enabled(
        ), torch.is_autocast_enabled()

        def _worker(i, module, input, kwargs, device=None):
            torch.set_grad_enabled(grad_enabled)
            if device is None:
                device = get_a_var(input).get_device()
            try:
                with torch.cuda.device(device), autocast(
                        enabled=autocast_enabled):
                    # this also avoids accidental slicing of `input` if it is a Tensor
                    if not isinstance(input, (list, tuple)):
                        input = (input, )
                    output = module.forward_representation(*input, **kwargs)
                with lock:
                    results[i] = output
            except Exception:
                with lock:
                    results[i] = ExceptionWrapper(
                        where="in replica {} on device {}".format(i, device))

        if len(modules) > 1:
            threads = [
                threading.Thread(target=_worker,
                                 args=(i, module, input, kwargs, device))
                for i, (module, input, kwargs, device) in enumerate(
                    zip(modules, inputs, kwargs_tup, devices))
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, ExceptionWrapper):
                output.reraise()
            outputs.append(output)
        return outputs

    def parallel_apply_dynamics(self,
                                modules,
                                inputs,
                                kwargs_tup=None,
                                devices=None):
        assert len(modules) == len(inputs)
        if kwargs_tup is not None:
            assert len(modules) == len(kwargs_tup)
        else:
            kwargs_tup = ({}, ) * len(modules)
        if devices is not None:
            assert len(modules) == len(devices)
        else:
            devices = [None] * len(modules)
        devices = list(map(lambda x: _get_device_index(x, True), devices))
        lock = threading.Lock()
        results = {}
        grad_enabled, autocast_enabled = torch.is_grad_enabled(
        ), torch.is_autocast_enabled()

        def _worker(i, module, input, kwargs, device=None):
            torch.set_grad_enabled(grad_enabled)
            if device is None:
                device = get_a_var(input).get_device()
            try:
                with torch.cuda.device(device), autocast(
                        enabled=autocast_enabled):
                    # this also avoids accidental slicing of `input` if it is a Tensor
                    if not isinstance(input, (list, tuple)):
                        input = (input, )
                    output = module.forward_dynamics(*input, **kwargs)
                with lock:
                    results[i] = output
            except Exception:
                with lock:
                    results[i] = ExceptionWrapper(
                        where="in replica {} on device {}".format(i, device))

        if len(modules) > 1:
            threads = [
                threading.Thread(target=_worker,
                                 args=(i, module, input, kwargs, device))
                for i, (module, input, kwargs, device) in enumerate(
                    zip(modules, inputs, kwargs_tup, devices))
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, ExceptionWrapper):
                output.reraise()
            outputs.append(output)
        return outputs

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
