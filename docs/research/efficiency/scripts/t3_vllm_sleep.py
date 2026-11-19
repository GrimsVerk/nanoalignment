"""vLLM on sm_120: generate throughput + sleep/wake cycle (what TRL colocate uses)."""
import sys, time, torch, os
from vllm import LLM, SamplingParams

model = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
util = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

def used():
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 1e9

def main():
    print("GPU used before", round(used(), 2))
    t0 = time.perf_counter()
    llm = LLM(model=model, gpu_memory_utilization=util, enable_sleep_mode=True, max_model_len=4096, dtype="bfloat16", kv_cache_dtype=os.environ.get("KVD", "auto"))
    print(f"init {time.perf_counter()-t0:.0f}s, GPU used {used():.2f} GB")
    prompts = [f"Write a bash one-liner that counts lines in file{i}.txt and explain it." for i in range(int(os.environ.get("NP", 64)))]
    sp = SamplingParams(max_tokens=int(os.environ.get("MT", 256)), temperature=1.0, ignore_eos=True)
    for rnd in range(2):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"round {rnd}: {ntok} tok in {dt:.2f}s = {ntok/dt:,.0f} tok/s")
        t0 = time.perf_counter()
        llm.sleep(level=1)
        print(f"  sleep(1) {time.perf_counter()-t0:.2f}s, GPU used {used():.2f} GB")
        t0 = time.perf_counter()
        llm.wake_up()
        print(f"  wake_up {time.perf_counter()-t0:.2f}s, GPU used {used():.2f} GB")
    print("OK")

if __name__ == '__main__':
    main()
