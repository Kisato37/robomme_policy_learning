# Preregistered N/S/P/SP configurations

| ID | History YAML | Symbolic prefix | Perceptual memory | Runtime symbolic source |
|---|---|---:|---:|---|
| N | none | no | no | none |
| S | `symbolic-grounded-subgoal.yaml` | GroundSG | no | Oracle or Qwen, same weights |
| P | `perceptual-framesamp-modul.yaml` | no | FrameSamp–Modulator | none |
| SP | `dual-grounded-framesamp-modul.yaml` | GroundSG | FrameSamp–Modulator | Oracle or Qwen, same weights |

All formal runs use batch 64, 80,000 steps, action horizon 20, AdamW with gradient clipping 1.0, 10,000 warmup steps, peak learning rate 5e-5, 100,000 decay steps, EMA 0.999, 8-device FSDP, and seeds 42/43/44. The final step 79,999 is selected without benchmark-based checkpoint selection.

The formal launch writes one immutable JSON-as-YAML snapshot per model/seed under `configs/`, including the resolved checkpoint directory, base checkpoint path, dataset path, and local metrics path.
