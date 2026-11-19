"""Text generation for the harness: vLLM first, Hugging Face transformers as fallback.

Adapted from research-2 (docs/research-2/scripts/gen_backend.py). Never uses
trust_remote_code: downloaded model code would run on the host.
"""

import os
import time


class Gen:
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        max_len: int = 4096,
        backend: str = "auto",
        seed: int = 0,
    ):
        self.model_id, self.max_len, self.backend = model, max_len, None
        t0 = time.time()
        if backend in ("auto", "vllm"):
            try:
                from vllm import LLM

                self.llm = LLM(
                    model=model, revision=revision, dtype="bfloat16", max_model_len=max_len,
                    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.55")), enforce_eager=True,
                    enable_prefix_caching=True, trust_remote_code=False, seed=seed,
                )  # fmt: skip
                self.tok = self.llm.get_tokenizer()
                self.backend = "vllm"
            except Exception as e:
                if backend == "vllm":
                    raise
                print(f"[gen] vLLM failed for {model}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                self.llm = None
        if self.backend is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=torch.bfloat16, device_map="cuda"
            ).eval()
            self.backend = "hf"
        self.load_s = round(time.time() - t0, 1)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])

    def generate(self, prompts: list[str], stop: list[str], max_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> list[str]:  # fmt: skip
        if not prompts:
            return []
        if self.backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(
                temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed,
            )
            return [o.outputs[0].text for o in self.llm.generate(prompts, sp, use_tqdm=False)]
        return self._hf(prompts, stop, max_tokens, temperature, seed)

    def _hf(self, prompts, stop, max_tokens, temperature, seed, bs=8):
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        res = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            kw = dict(
                max_new_tokens=max_tokens,
                pad_token_id=self.tok.pad_token_id,
                stop_strings=stop,
                tokenizer=self.tok,
            )
            kw.update(
                dict(do_sample=True, temperature=temperature, top_p=0.95)
                if temperature > 0
                else dict(do_sample=False)
            )
            with torch.no_grad():
                out = self.model.generate(**enc, **kw)
            for t in self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True):
                cut = min([t.find(s) for s in stop if s in t] or [len(t)])
                res.append(t[:cut])
        return res
