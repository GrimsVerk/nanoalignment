"""2-step Unsloth GRPO smoke test (LoRA, fast_inference vLLM, standby) on sm_120. Not training."""
import os, sys, time
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

name = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
t0 = time.perf_counter()
model, tok = FastLanguageModel.from_pretrained(name, max_seq_length=1024, load_in_4bit=False,
                                               fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.5)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, use_gradient_checkpointing="unsloth",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
print(f"load {time.perf_counter()-t0:.0f}s", flush=True)
ds = Dataset.from_list([{"prompt": [{"role": "user", "content": f"Say a bash command to list file{i}."}]} for i in range(32)])
def reward(completions, **kw):
    return [float("ls" in (c[0]["content"] if isinstance(c, list) else c)) for c in completions]
args = GRPOConfig(output_dir="out_unsloth", max_steps=2, per_device_train_batch_size=8,
                  num_generations=8, max_completion_length=128, logging_steps=1, report_to=[], learning_rate=1e-5,
                  bf16=True, save_strategy="no")
tr = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward], args=args, train_dataset=ds)
t0 = time.perf_counter()
tr.train()
print(f"GRPO 2 steps {time.perf_counter()-t0:.1f}s, peak torch alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
f, t = torch.cuda.mem_get_info()
print(f"GPU used now {(t-f)/1e9:.2f} GB\nOK", flush=True)
