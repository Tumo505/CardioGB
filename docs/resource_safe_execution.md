# Resource-safe real-data execution

The original full-stage graph runs used approximately 10.4 GB of the 12.2 GB
reported GPU memory continuously and could make the laptop unresponsive. Full
experiments now use a bounded, recoverable protocol.

## Safety controls

- Source graphs are split by tissue section into deterministic spatial patches
  of at most 4,000 spots.
- Training samples two patches per stage transition per epoch.
- Validation uses deterministic representative patches.
- Final evaluation streams every held-out patch and concatenates predictions
  before calculating one metric per biological transition.
- PyTorch may allocate at most 65% of GPU memory.
- CUDA caches are released after every model or ensemble member.
- A 20-second cooldown follows each trained unit.
- Long GPU models also pause for 15 seconds after every five completed epochs. The
  trainer releases unused CUDA cache before each pause so laptop cooling can catch up.
- Benchmark, ablation, and ensemble commands train at most one new model/member
  per invocation by default.
- Every completed unit has its own metrics, split, checkpoint, and manifest
  entry, so a restart resumes at the first missing unit.

A CardioGB safety smoke test recorded 1,982,012,416 bytes (1.85 GiB) peak
PyTorch allocation, versus about 10.4 GiB observed during the prior full-stage
run.

## Safe continuation commands

Each command below completes at most one new learned unit and then exits.

~~~powershell
$env:PYTHONPATH='src'
python scripts\run_multiseed_benchmark.py   --data data\processed\zebrafish_states.npz   --output-dir results\real_batched_multiseed   --epochs 200   --seeds 20260815 20260816 20260817 20260818 20260819   --max-new-models 1
~~~

~~~powershell
python scripts\run_real_ablations.py   --data data\processed\zebrafish_states.npz   --rank-data data\processed\zebrafish_states_rank_mean.npz   --output-dir results\real_batched_ablations   --reference-dir results\real_batched_multiseed   --epochs 200   --seeds 20260815 20260816 20260817 20260818 20260819   --max-new-models 1
~~~

~~~powershell
python scripts\train_ensemble.py   --data data\processed\zebrafish_states.npz   --output-dir results\real_batched_ensemble   --members 5   --epochs 200   --seed 20260815   --max-new-members 1
~~~

Do not combine metrics from the archived unbatched runs with the bounded-patch
study. They use different optimization sampling schemes.
