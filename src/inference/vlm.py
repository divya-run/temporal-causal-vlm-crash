"""
Thin wrapper around Qwen2.5-VL for both:
  - cloud-GPU reference runs (full precision, configs/base.yaml default), and
  - on-device Jetson runs (quantized — see src/edge/jetson_bench.py),
so the eval loop (src/eval/*) doesn't need to know which environment it's in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VLMResponse:
    raw_text: str
    predicted_letter: str | None
    latency_ms: float
    peak_memory_mb: float | None = None


class QwenVLWrapper:
    def __init__(self, model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype)
        self._model = None
        self._processor = None

    def load(self) -> None:
        # Deferred import so this module can be imported (e.g. for unit tests
        # of prompt/sampling logic) without requiring transformers/torch+cuda
        # to be installed in every environment.
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map=self.device
        )
        self._model.eval()

    def answer(
        self,
        frames: list,  # list of PIL.Image, already sampled per src/sampling
        system_prompt: str,
        user_prompt: str,
        frame_labels: list[str] | None = None,
        max_new_tokens: int = 16,
    ) -> VLMResponse:
        if self._model is None:
            self.load()

        content = []
        for i, frame in enumerate(frames):
            if frame_labels:
                content.append({"type": "text", "text": f"[{frame_labels[i]}]"})
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=frames, return_tensors="pt").to(
            self._model.device
        )

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

        starter.record()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        ender.record()
        torch.cuda.synchronize()

        latency_ms = starter.elapsed_time(ender)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        gen_text = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        predicted_letter = _extract_letter(gen_text)
        return VLMResponse(
            raw_text=gen_text,
            predicted_letter=predicted_letter,
            latency_ms=latency_ms,
            peak_memory_mb=peak_mem_mb,
        )


def _extract_letter(text: str) -> str | None:
    for ch in text.strip():
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return None
