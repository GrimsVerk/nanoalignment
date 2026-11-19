"""Text generation for base models: vLLM first, Hugging Face transformers as fallback."""
import os, time


class Gen:
    def __init__(self, model_id, revision=None, max_len=4096, backend="auto", gpu_util=None):
        self.model_id, self.max_len = model_id, max_len
        self.backend = None
        t0 = time.time()
        if backend in ("auto", "vllm"):
            try:
                from vllm import LLM
                util = gpu_util or float(os.environ.get("GPU_UTIL", "0.55"))
                self.llm = LLM(model=model_id, revision=revision, dtype="bfloat16",
                               max_model_len=max_len, gpu_memory_utilization=util,
                               enforce_eager=True, enable_prefix_caching=True,
                               max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "128")),
                               trust_remote_code=False, seed=0)
                self.tok = self.llm.get_tokenizer()
                self.backend = "vllm"
            except Exception as e:  # noqa
                if backend == "vllm":
                    raise
                print(f"[gen] vLLM failed for {model_id}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                self._free_vllm()
        if self.backend is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda").eval()
            self.backend = "hf"
        self.load_s = time.time() - t0

    def _free_vllm(self):
        try:
            import gc, torch
            self.llm = None
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    def n_tokens(self, text):
        return len(self.tok(text)["input_ids"])

    def generate(self, prompts, stop, max_tokens=256, temperature=0.0, n=1, seed=0, json_schema=None):
        """Return list (per prompt) of list (n samples) of completion strings."""
        if not prompts:
            return []
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                                max_tokens=max_tokens, stop=stop, seed=seed, **self._so(json_schema))
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
            return [[c.text for c in o.outputs] for o in outs]
        return self._hf_generate(prompts, stop, max_tokens, temperature, n, seed)

    @staticmethod
    def _so(schema):
        if schema is None:
            return {}
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=schema)}

    def _hf_generate(self, prompts, stop, max_tokens, temperature, n, seed, bs=8):
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        res = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            kw = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id,
                      stop_strings=stop, tokenizer=self.tok, num_return_sequences=n)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(**enc, **kw)
            L = enc["input_ids"].shape[1]
            texts = self.tok.batch_decode(out[:, L:], skip_special_tokens=True)
            for j in range(len(chunk)):
                group = []
                for t in texts[j * n:(j + 1) * n]:
                    cut = min([t.find(s) for s in stop if s in t] or [len(t)])
                    group.append(t[:cut])
                res.append(group)
        return res
