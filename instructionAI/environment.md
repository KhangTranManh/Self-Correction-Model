# Environment

## Hardware is not stable across sessions — don't assume a specific box

This project has moved across multiple rented/free GPU platforms in the course of
development: a per-hour rented VPS with root SSH (V100-SXM2-32GB, then later a fresh
container of the same VPS type), a Kaggle Notebook (T4 x2, no native SSH, no sudo),
an RTX 3090 24GB root-SSH container used for the latest train/export/serve run, and a
JupyterHub deployment (L40S, 46GB VRAM, no sudo, no SSH). None of these are
assumed to persist — always re-verify GPU model, access method, and installed packages
at the start of a session rather than trusting what a previous session found. The one
constant is the project code itself (this repo) and, ideally, `data/processed/
phase1_sft.jsonl` if it was successfully carried over from a previous box.

## Python / package manager

Varies by platform — check before assuming:
- Rented VPS-style boxes (root access): Python usually lives in a conda base env, NOT
  at the system `python3` (e.g. `/opt/conda/bin/python`). Using the system
  `/usr/bin/python3` by mistake produces `ModuleNotFoundError` for every third-party
  package, since none of this project's deps are installed there.
- Kaggle / JupyterHub: whichever kernel/conda env is active in the notebook (check with
  `which python3`, `conda env list`). Installing packages while a specific conda env is
  active only affects that env — it does not touch other envs (`base`, etc.) on the
  same box.

## Unsloth is the current default — GPU compute capability must be >= 7.5

`train_sft.py`, `generate_attempts.py`, and `evaluate_self_correction.py` currently
import `unsloth.FastLanguageModel` directly. This requires Turing or newer (T4, L40S,
A100, 4090, H100...) — compute capability 7.5+.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```
Check this FIRST on any new box before running the pipeline.

Install order:
```bash
pip install unsloth
pip install -r requirements.txt   # everything else; do NOT add version pins here
```
Unsloth pulls in a compatible `torch`/`bitsandbytes`/`trl`/`transformers` set on its own.
Pinning those separately (as the V100 setup below does) causes dependency conflicts —
this was hit in practice when reusing V100-era pins on a newer GPU.

### If you land on a V100-class GPU (compute capability 7.0) again

Unsloth (and Axolotl) on their current releases hard-require `torch>=2.3`/`torch>=2.11`,
and torch dropped CC 7.0 kernel support around that line — the failure mode is silent
and dangerous: `torch.cuda.is_available()` still returns `True`, but actual tensor ops
either error out or (worse) run incorrectly. This was verified empirically, twice
(Unsloth's transitive torch upgrade, and Axolotl 0.18.0's hard pin to
`torch<=2.12.1,>=2.11.0` + `bitsandbytes==0.49.1`, both confirmed via `pip install
--dry-run` before touching the live env).

If this happens, **revert `train_sft.py`, `generate_attempts.py`, and
`evaluate_self_correction.py`** to the plain `transformers + peft + trl` pattern (no
Unsloth import — `AutoModelForCausalLM` + `BitsAndBytesConfig` +
`peft.get_peft_model`/`prepare_model_for_kbit_training`; see git history for the exact
version used before Unsloth was reintroduced), and pin:
```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers>=4.44.0,<5.0' 'trl>=0.9.6,<0.10' 'bitsandbytes<0.44,>=0.43.0' peft accelerate datasets openai huggingface_hub python-dotenv pyyaml sympy
```

**Quoting gotcha** (applies on any box, any stack): version-constrained pip args
(`'transformers>=4.44.0,<5.0'`) MUST be quoted in shell commands. An unquoted `>` is
interpreted as shell output redirection, silently creating a junk file and installing
an unconstrained (too-new) package version. This caused a real incident (torch got
silently bumped this way).

### Smoke test after any dependency change (any GPU)

```bash
python3 -c "
import torch
print(torch.__version__, torch.cuda.is_available())
x = torch.randn(4,4, device='cuda') @ torch.randn(4,4, device='cuda')
torch.cuda.synchronize()
print('real compute ok', x.shape)
"
```
A clean run with no `UserWarning` about "does not include kernels for this GPU" means
the environment is sound. On some containers (seen on a Kaggle T4 box), a correct
driver is present but `LD_LIBRARY_PATH` doesn't include the NVIDIA userspace libs by
default in a fresh shell (SSH session env differs from the Jupyter kernel's own env) —
if `nvidia-smi`/torch can't find `libnvidia-ml.so` despite `/dev/nvidia*` existing,
`find / -iname 'libnvidia-ml.so*'` to locate it and `export LD_LIBRARY_PATH=<dir>:
$LD_LIBRARY_PATH` (persist in `~/.bashrc` for the session).

## API provider config

`.env` (project root) holds: `DEEPSEEK_MODEL`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`,
`HF_TOKEN`. Despite the `DEEPSEEK_*` naming, these point at whatever OpenAI-compatible
provider is currently configured — currently a proxy at `https://api.vilao.ai/v1`
serving GLM-5.2 (model id `op/z-ai/glm-5.2`). Both DeepSeek's own API and this GLM proxy
expose a `reasoning_content` field on the chat completion message (separate from
`content`) — this is real chain-of-thought the model produces, used as the "Thinking"
trace in training data. Confirm this field still appears if the provider changes again.

`.env` must be created directly on whichever machine is running the pipeline — it is
never copied/synced between machines (deliberately excluded from every transfer
method used: `scp`, zip upload, notebook embedding). A past mistake was saving it to
the home directory (`~/.env`) instead of the project root (`~/AGI/.env` or
`AGI/.env` relative to wherever the notebook/shell's cwd is) — `config.py` only looks
in the project root, so a misplaced `.env` fails with a "not set" error even though the
file exists.

`test_deepseek_connection.py` is the cheap way to validate `.env` — no GPU needed, one
API call, checks schema. Run it after any `.env` or provider change, before spending
GPU-hours on `generate_attempts.py`/`train_sft.py`.

## vLLM export and serving

Keep serving dependencies in a separate environment from Unsloth training. The latest
verified deployment used `/workspace/AGI/.venv_vllm` with vLLM 0.28, while training
remained in `/workspace/AGI/.venv_train`. Mixing their torch/transformers dependency
sets is unnecessary and makes a working training environment easy to break.

The attempted pre-quantized serving path was not reliable in this environment:

- `vllm-bnb-plugin` 0.0.3 called an API removed by vLLM 0.28
  (`WeightsMapper.get_rename_mapper`);
- after a compatibility shim, the quantized-base + runtime-LoRA path failed in CUDA's
  LoRA kernel; and
- these failures were serving-stack problems, not evidence that the trained adapter
  was invalid (the adapter independently passed reload/forward validation).

The working path is `src/export_merged_for_vllm.py`: load the original BF16 Qwen base,
apply `outputs/phase1_lora`, merge safely, save four safetensor shards, upload to
`Kxck/Self_Correction_v1`, and serve the standalone merged checkpoint with native vLLM.
This used about 16.4 GiB VRAM during export and 22.5/24.6 GiB while serving at
`--max-model-len 4096`; a 24GB RTX 3090 is sufficient but has little serving headroom.

CUDA-runtime-only containers may have no `/usr/local/cuda` and no `nvcc`. vLLM 0.28's
FlashInfer sampler then tries to JIT-compile during warm-up and exits. Use the supported
native sampler switch:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Kxck/Self_Correction_v1 \
  --served-model-name Kxck/Self_Correction_v1 \
  --host 0.0.0.0 --port 8999 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --dtype bfloat16 --api-key "$VLLM_API_KEY"
```

Always protect the endpoint with an API key, suppress command-argument logging when it
could expose that key, and rotate any key that appeared in a failed startup log. Store
the key in a permission-restricted file or process environment, never in tracked config.
The rented host/port mapping is ephemeral; probe authenticated `/v1/models` locally and
through the external mapped port before declaring the service ready.

See `results.md` for the model manifest, measured behavior, and canonical evaluation
artifact names.

## Access methods encountered (varies by platform, don't assume SSH is available)

- **Root SSH VPS rental**: standard `ssh -p <PORT> root@<HOST>`. May be a brand-new
  container on reconnect even at the same IP:port — symptom: SSH warns "REMOTE HOST
  IDENTIFICATION HAS CHANGED". If so, the disk is fresh (code, packages, `.env`,
  `data/processed/`/`outputs/` all gone) and must be recreated: `ssh-keygen -R
  "[<HOST>]:<PORT>"` to clear the stale host-key entry, re-register the existing local
  key via one-time password login, `scp` the project over (excluding `.env`), reinstall
  deps, recreate `.env` directly on the box.
- **Kaggle Notebook**: no SSH, no sudo, root user. Workaround used: install
  `openssh-server` + inject a public key into `authorized_keys`, then tunnel port 22
  out via `pyngrok` (needs a free ngrok account + authtoken). The resulting
  `tcp://X.tcp.ngrok.io:PORT` tunnel is fragile — observed random disconnects
  ("Connection closed by remote host" / "Connection refused" after ~tens of minutes);
  a training process survived tunnel drops in practice (didn't receive SIGHUP), but a
  *new* long-lived monitoring connection through the same tunnel is not reliable —
  prefer short reconnect-and-check commands over one persistent watch session, and
  always redirect long-running commands to a log file on the box itself (`nohup ...
  > train.log 2>&1 & disown`) rather than relying on the SSH pipe to capture output,
  since that output is lost if the tunnel drops mid-run.
- **JupyterHub, non-root, no sudo, no sshd binary and no apt**: no way to install
  system SSH. Workaround: `conda install -c conda-forge openssh -y` (installs `sshd`
  into the active conda env, no root needed) + generate host keys and a custom
  `sshd_config` under `$HOME` (not `/etc`) bound to a non-privileged port (e.g. 2222)
  with `UsePAM no` and `StrictModes no` to avoid needing root — then tunnel via ngrok
  as above. If even this isn't feasible, the fallback that always works regardless of
  platform restrictions is running the pipeline directly from notebook cells (`!cd AGI
  && python -m src.train_sft`) — this needs no remote-control channel at all since the
  notebook's own kernel already has the GPU; only code/data transfer must be solved
  (e.g. upload a zip of the project next to the notebook and `!unzip` it in a cell).
- Whichever method: `~/work/...`-style directories in JupyterHub-style deployments are
  often the persistent-volume mount (survives container restart) — prefer that location
  over ephemeral paths when choosing where to place the project.
