# Full-training configurations

Stage 03 formal baseline configuration has been created:

- config: `configs/full/stage03_b0_ce_512_seed42.yaml`
- split: fixed `5597/1399` train/val files in `splits/`
- model: SegFormer-B0
- crop size: `512`
- loss: plain Cross Entropy
- quick flag: `quick_pilot=false`
- trainer stop rule: current Trainer stops by `max_steps`
- schedule: `56000` steps, approximately `40` epochs at `1400` optimizer steps per epoch

RTX 4060 is suitable for environment checks, 20-step smoke tests, and formal
training when time allows. Full training is expected to take much longer than the
quick pilots. RTX 4090 is better suited for quickly completing full 56000-step
training.

Do not directly edit the formal baseline while keeping the same `experiment_id`.
Any optimization or ablation must copy the config to a new file and use a new
`experiment_id`.
