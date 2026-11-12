"""Additional RPC methods injected into the native vLLM V1 worker."""
from __future__ import annotations

from .state import CacheTuneState
import torch


class CacheTuneWorkerExtension:
    # Extension methods intentionally do not override any native Worker method.
    def _cachetune_get_model(self):
        """Unwrap native causal and conditional-generation model containers."""
        outer = self.model_runner.model
        target = getattr(outer, "language_model", outer)
        model = getattr(target, "model", None)
        if model is None:
            raise RuntimeError("The native model has no CacheTune text-model container")
        return model

    def _cachetune_get_state(self) -> CacheTuneState:
        model = self._cachetune_get_model()
        state = getattr(model, "_cachetune_state", None)
        if not isinstance(state, CacheTuneState):
            raise RuntimeError("The native model has no CacheTune adapter")
        return state

    def cachetune_reset(self, clear_cache: bool = True):
        result = self._cachetune_get_state().reset(clear_cache=clear_cache)
        if clear_cache:
            model = self._cachetune_get_model()
            model._cachetune_debug = None
            model._cachetune_debug_armed = False
        return result

    def cachetune_start_collect(self):
        return self._cachetune_get_state().start_collect()

    def cachetune_append_chunk(self, expected_ids, drop_bos: bool = False):
        return self._cachetune_get_state().append_chunk(expected_ids, drop_bos=drop_bos)

    def cachetune_prepare(self, prompt_ids, suffix_len: int, ratio: float, **options):
        return self._cachetune_get_state().prepare(prompt_ids, suffix_len, ratio, **options)

    def cachetune_enable(self, prompt_ids):
        return self._cachetune_get_state().enable(prompt_ids)

    def cachetune_disable(self):
        return self._cachetune_get_state().disable()

    def cachetune_diagnostics(self):
        result = self._cachetune_get_state().diagnostics()
        from . import __version__
        result["adapter_version"] = __version__
        result["adapter_file"] = __file__
        outer = self.model_runner.model
        result["model_class"] = f"{type(outer).__module__}.{type(outer).__qualname__}"
        model = self._cachetune_get_model()
        result["text_model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        result["worker_class"] = f"{type(self).__module__}.{type(self).__qualname__}"
        result["runner_class"] = f"{type(self.model_runner).__module__}.{type(self.model_runner).__qualname__}"
        return result

    def cachetune_profile_transfer(self, repeats=3):
        """Measure serial compact-source H2D cost for the GSS warm start.

        The profiling RPC is outside timed generation. It never changes cache
        tensors or prepares a request; full-request TTFT remains the GSS
        objective. Hybrid branch sources are counted together with KV sources.
        """
        import math
        import statistics

        state = self._cachetune_get_state()
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not state.prepared or state.cpu_kv_cache is None or state.active:
            raise ValueError("Profile a prepared, inactive CPU cache")
        count = int(state._reused_cpu.numel())
        if count == 0:
            raise ValueError("Transfer profiling needs at least one reused token")
        sources = [tensor for pair in state.cpu_kv_cache[state.first_fusion_layer:]
                   for tensor in pair]
        sources += [tensor for row in getattr(state, "_branch_sources", {}).values()
                    for tensor in row.values()]
        if not sources or any(x.device.type != "cpu" or not x.is_pinned() for x in sources):
            raise ValueError("Expected immutable pinned CPU sources")
        with torch.cuda.device(state.device):
            destinations = [torch.empty_like(x, device=state.device) for x in sources]
            durations = []
            for index in range(repeats + 1):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for dst, src in zip(destinations, sources):
                    dst.copy_(src, non_blocking=True)
                end.record()
                end.synchronize()
                if index:
                    durations.append(start.elapsed_time(end) / 1000.0)
        seconds = statistics.mean(durations)
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("Transfer profile returned an invalid duration")
        return {"seconds": seconds, "seconds_per_reused_token": seconds / count,
                "transferred_bytes": sum(x.numel() * x.element_size() for x in sources),
                "reused_tokens": count, "repeats": repeats,
                "measurements_seconds": durations,
                "scope": "Serial H2D only; warm-start profile outside generation"}

    def cachetune_debug_start(self, layers=None, raw_qkv=False,
                             attention_proxy=False, suffix_len=1):
        """Arm one prefill's legacy diagnostic exports, outside timed runs."""
        from .debug import CacheTuneDebugCapture
        self._cachetune_get_state()  # Reject architectures without this adapter.
        model = self._cachetune_get_model()
        if any(getattr(layer, "layer_type", None) == "linear_attention" for layer in model.layers):
            raise NotImplementedError("Hybrid raw-QKV/attention-proxy debug capture is not implemented")
        if layers is not None and any(
                isinstance(i, bool) or not isinstance(i, int)
                or not 0 <= i < len(model.layers) for i in layers):
            raise ValueError("Debug layer IDs must lie within this model")
        if not raw_qkv and not attention_proxy:
            raise ValueError("Enable raw_qkv and/or attention_proxy for debug capture")
        model._cachetune_debug = CacheTuneDebugCapture(
            layers=layers, raw_qkv=raw_qkv, attention_proxy=attention_proxy,
            suffix_len=suffix_len)
        model._cachetune_debug_armed = True
        return {"armed": True, "layers": layers, "suffix_len": suffix_len,
                "raw_qkv": raw_qkv, "attention_proxy": attention_proxy}

    def cachetune_debug_finish(self, output_path):
        """Write CPU tensors locally; V1 utility RPC coerces tensor payloads."""
        import os
        from pathlib import Path
        self._cachetune_get_state()
        model = self._cachetune_get_model()
        debugger = getattr(model, "_cachetune_debug", None)
        if debugger is None or getattr(model, "_cachetune_debug_armed", False):
            raise RuntimeError("No completed debug prefill is available")
        result = debugger.finish()
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        torch.save(result, temporary)
        os.replace(temporary, path)
        model._cachetune_debug = None
        model._cachetune_debug_armed = False
        return {"path": str(path), "format": "torch.save CPU tensors",
                "layers": sorted(result["layers"]), "bytes": path.stat().st_size}

    def cachetune_debug_reset(self):
        self._cachetune_get_state()
        model = self._cachetune_get_model()
        model._cachetune_debug = None
        model._cachetune_debug_armed = False
        return {"cleared": True}

    def cachetune_probe_start(self):
        state = self._cachetune_get_state()
        state._probe_next, state._probe_logits = True, None
        return {"armed": True}

    def cachetune_fingerprints(self):
        """Return diagnostic byte hashes through a named, serializable V1 RPC."""
        import hashlib

        state = self._cachetune_get_state()

        def fingerprint(tensor):
            value = tensor.detach().cpu().contiguous()
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}

        logits = getattr(state, "_probe_logits", None)
        if logits is None:
            raise RuntimeError("No captured first-token logits for fingerprinting")
        result = {"logits": fingerprint(logits)}
        recurrent = getattr(state, "_recurrent_snapshot", None)
        if recurrent:
            result["recurrent"] = {
                str(layer): {name: fingerprint(tensor) for name, tensor in sorted(row.items())}
                for layer, row in sorted(recurrent.items())}
        return result

    def cachetune_probe_finish(self, set_reference=False):
        state = self._cachetune_get_state()
        logits = getattr(state, "_probe_logits", None)
        if logits is None:
            raise RuntimeError("No first-token logits captured")
        result = {"finite":bool(torch.isfinite(logits).all()), "shape":list(logits.shape),
                  "argmax":logits.argmax(dim=-1).tolist()}
        if set_reference:
            state._probe_reference = logits.clone()
        else:
            reference = getattr(state, "_probe_reference", None)
            if reference is None or reference.shape != logits.shape:
                raise RuntimeError("A matching first-logit reference is required")
            difference = logits - reference
            result.update(max_absolute_error=float(difference.abs().max()),
                          relative_l2=float(difference.norm() / logits.norm().clamp_min(1e-12)),
                          reference_argmax=reference.argmax(dim=-1).tolist())
        return result

    def cachetune_recurrent_probe_start(self):
        """Arm a CPU snapshot of this request's native prefill-end GDN states."""
        state = self._cachetune_get_state()
        if not any(getattr(layer, "layer_type", None) == "linear_attention"
                   for layer in self._cachetune_get_model().layers):
            raise ValueError("Recurrent-state probes require a hybrid model")
        state._recurrent_probe_next, state._recurrent_snapshot = True, None
        return {"armed": True, "endpoint": "prefill_end"}

    def cachetune_recurrent_probe_finish(self, set_reference=False):
        """Return JSON metrics only; CPU tensors remain inside the worker."""
        state = self._cachetune_get_state()
        snapshot = getattr(state, "_recurrent_snapshot", None)
        if getattr(state, "_recurrent_probe_next", False) or not snapshot:
            raise RuntimeError("No completed recurrent-state prefill snapshot is available")
        for layer, buffers in snapshot.items():
            if set(buffers) != {"conv", "ssm"} or any(
                    not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    for tensor in buffers.values()):
                raise RuntimeError(f"Layer {layer}: recurrent snapshot must contain CPU conv/ssm tensors")
        reference = None
        if not set_reference:
            reference = getattr(state, "_recurrent_reference", None)
            if reference is None or set(reference) != set(snapshot):
                raise RuntimeError("A matching recurrent-state reference is required")
        result = {"endpoint": "prefill_end", "finite": True, "layers": {}}
        if reference is not None:
            result["all_equal"] = True
        for layer, buffers in sorted(snapshot.items()):
            layer_result = {}
            for name, tensor in buffers.items():
                finite = bool(torch.isfinite(tensor).all())
                metrics = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite}
                result["finite"] = result["finite"] and finite
                if reference is not None:
                    baseline = reference[layer].get(name)
                    if (baseline is None or baseline.shape != tensor.shape
                            or baseline.dtype != tensor.dtype):
                        raise RuntimeError(f"Layer {layer} {name}: recurrent reference shape/dtype mismatch")
                    difference = tensor.float() - baseline.float()
                    equal = bool(torch.equal(tensor, baseline))
                    baseline_finite = bool(torch.isfinite(baseline).all())
                    comparison_finite = finite and baseline_finite
                    metrics.update(equal=equal, reference_finite=baseline_finite,
                                   max_absolute_error=(float(difference.abs().max()) if difference.numel() else 0.0)
                                   if comparison_finite else None,
                                   relative_l2=float(difference.norm() / baseline.float().norm().clamp_min(1e-12))
                                   if comparison_finite else None)
                    result["finite"] = result["finite"] and baseline_finite
                    result["all_equal"] = result["all_equal"] and equal
                layer_result[name] = metrics
            result["layers"][str(layer)] = layer_result
        if set_reference:
            state._recurrent_reference = {
                layer: {name: tensor.clone() for name, tensor in buffers.items()}
                for layer, buffers in snapshot.items()}
            result["reference_saved"] = True
        return result
