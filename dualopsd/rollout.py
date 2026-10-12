"""Rollout engine: in-process vLLM with refreshable LoRA adapters."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .types import RolloutBatch


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _right_pad(rows: list[list[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows))
    values = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(values)
    for index, row in enumerate(rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, : len(row)] = 1
    return values, mask


def _right_pad_floats(rows: list[list[float]], width: int, device: torch.device) -> torch.Tensor:
    values = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.float32, device=device)
    return values


def _batch(tokenizer, prompts, completion_rows, logprob_rows, example_indices, texts, device) -> RolloutBatch:
    completion_ids, completion_mask = _right_pad(completion_rows, tokenizer.pad_token_id, device)
    prompt_rows = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    prompt_ids, prompt_mask = _right_pad(prompt_rows, tokenizer.pad_token_id, device)
    return RolloutBatch(
        prompt_texts=list(prompts),
        completion_texts=texts,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        prompt_lengths=torch.tensor([len(row) for row in prompt_rows], dtype=torch.long, device=device),
        old_logprobs=_right_pad_floats(logprob_rows, completion_ids.shape[1], device),
        example_indices=torch.as_tensor(example_indices, dtype=torch.long, device=device),
    )


class VLLMRolloutEngine:
    """In-process vLLM generation with named LoRA adapters.

    Adapters are exported to ``adapter_root`` and marked dirty after every
    optimizer step, so a rollout never uses stale weights. vLLM returns the
    sampled-token log-probabilities at generation time.
    """

    def __init__(
        self,
        model,
        tokenizer,
        adapters,
        *,
        model_name: str,
        dtype: str,
        lora_rank: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        seed: int = 42,
        sleep_level: int = 0,
        adapter_root: str | Path | None = None,
        llm=None,
        sampling_params_class=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters = adapters
        self.sleep_level = sleep_level
        self._sleeping = False
        self._sleep_calls = 0
        self._wake_calls = 0
        self._sleep_seconds = 0.0
        self._wake_seconds = 0.0
        self._post_sleep_free_mib = 0.0
        if llm is None:
            from vllm import LLM

            llm = LLM(
                model=model_name,
                dtype=dtype,
                trust_remote_code=False,
                enable_lora=True,
                max_lora_rank=lora_rank,
                max_loras=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                seed=seed,
                disable_log_stats=True,
                enable_sleep_mode=sleep_level > 0,
            )
        self.llm = llm
        if sampling_params_class is None:
            from vllm import SamplingParams

            sampling_params_class = SamplingParams
        self.sampling_params_class = sampling_params_class
        if adapter_root is None:
            root = Path(tempfile.mkdtemp(prefix="dualopsd_lora_", dir=os.environ.get("TMPDIR", "/tmp")))
            atexit.register(shutil.rmtree, root, ignore_errors=True)
        else:
            root = Path(adapter_root)
            root.mkdir(parents=True, exist_ok=True)
        self.adapter_root = root
        self._adapter_ids = {self.adapters.student: 1, self.adapters.teacher: 2}
        self._requests: dict[str, Any] = {}
        self._dirty = {self.adapters.student, self.adapters.teacher}

    def _engine(self):
        engine = getattr(self.llm, "llm_engine", None)
        return engine if engine is not None and hasattr(engine, "list_loras") else None

    def _remove_loaded(self, adapter_name: str) -> None:
        engine = self._engine()
        adapter_id = self._adapter_ids[adapter_name]
        if engine is not None and adapter_id in engine.list_loras():
            engine.remove_lora(adapter_id)

    def _prepare_request(self, adapter_name: str):
        request = self._materialize(adapter_name)
        engine = self._engine()
        if engine is not None:
            for loaded_id in list(engine.list_loras()):
                if loaded_id != self._adapter_ids[adapter_name]:
                    engine.remove_lora(loaded_id)
        return request

    def _materialize(self, adapter_name: str):
        if adapter_name not in self._dirty and adapter_name in self._requests:
            return self._requests[adapter_name]
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        from vllm.lora.request import LoRARequest

        self._remove_loaded(adapter_name)
        destination = self.adapter_root / adapter_name
        destination.mkdir(parents=True, exist_ok=True)
        self.model.peft_config[adapter_name].save_pretrained(destination)
        state = get_peft_model_state_dict(self.model, adapter_name=adapter_name)
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
            destination / "adapter_model.safetensors",
        )
        request = LoRARequest(adapter_name, self._adapter_ids[adapter_name], str(destination))
        self._requests[adapter_name] = request
        self._dirty.discard(adapter_name)
        return request

    def sync_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self._adapter_ids:
            raise ValueError(f"unknown adapter: {adapter_name}")
        # Engine RPCs are invalid while vLLM sleeps; stale adapters are removed
        # lazily after wake-up, right before the next generate call.
        if not self._sleeping:
            self._remove_loaded(adapter_name)
        self._dirty.add(adapter_name)

    def sleep(self) -> None:
        if self.sleep_level == 0 or self._sleeping:
            return
        started = time.monotonic()
        self.llm.sleep(level=self.sleep_level)
        self._sleep_seconds += time.monotonic() - started
        self._sleep_calls += 1
        self._sleeping = True
        if torch.cuda.is_available():
            self._post_sleep_free_mib = torch.cuda.mem_get_info()[0] / (1024**2)

    def wake_up(self) -> None:
        if not self._sleeping:
            return
        started = time.monotonic()
        self.llm.wake_up()
        self._wake_seconds += time.monotonic() - started
        self._wake_calls += 1
        self._sleeping = False

    def runtime_metrics(self) -> dict[str, float]:
        return {
            "system/vllm_sleep_calls": float(self._sleep_calls),
            "system/vllm_wake_calls": float(self._wake_calls),
            "system/vllm_sleep_seconds": self._sleep_seconds,
            "system/vllm_wake_seconds": self._wake_seconds,
            "system/vllm_post_sleep_free_mib": self._post_sleep_free_mib,
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        adapter_name: str,
        prompts: Sequence[str],
        num_generations: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> RolloutBatch:
        if not prompts or num_generations < 1:
            raise ValueError("generate requires prompts and at least one generation")
        if adapter_name not in self._adapter_ids:
            raise ValueError(f"unknown adapter: {adapter_name}")
        self.wake_up()
        sampling_kwargs: dict[str, Any] = dict(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else 0,
            max_tokens=max_new_tokens,
            logprobs=0,
            detokenize=False,
            ignore_eos=False,
        )
        if seed is not None:
            sampling_kwargs["seed"] = seed
        outputs = self.llm.generate(
            list(prompts),
            sampling_params=self.sampling_params_class(**sampling_kwargs),
            lora_request=self._prepare_request(adapter_name),
            use_tqdm=False,
        )
        completion_rows, logprob_rows, expanded, indices = [], [], [], []
        for problem_index, request_output in enumerate(outputs):
            for completion in sorted(request_output.outputs, key=lambda item: item.index):
                token_ids = list(completion.token_ids)
                if not token_ids or completion.logprobs is None:
                    raise RuntimeError("vLLM must return completion tokens and sampled log-probabilities")
                logprobs = []
                for token_id, step_values in zip(token_ids, completion.logprobs):
                    value = step_values.get(token_id)
                    if value is None:
                        raise RuntimeError("vLLM omitted the sampled token from requested log-probabilities")
                    logprobs.append(float(value.logprob))
                if len(logprobs) != len(token_ids):
                    raise RuntimeError("vLLM token/log-probability lengths are not aligned")
                completion_rows.append(token_ids)
                logprob_rows.append(logprobs)
                expanded.append(prompts[problem_index])
                indices.append(problem_index)
        texts = [self.tokenizer.decode(row, skip_special_tokens=True) for row in completion_rows]
        return _batch(self.tokenizer, expanded, completion_rows, logprob_rows, indices, texts, model_device(self.model))
