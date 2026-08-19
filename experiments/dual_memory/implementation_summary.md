# Dual-memory implementation summary

## Minimal architecture change

- The perceptual branch remains the official `representation_type=perceptual`, `frame_sampling`, `integration_type=modulation` path with budget 512, 16 tokens per sampled frame, and one front-view history stream.
- The hybrid configuration adds an independent `use_symbolic_prompt` switch and the official `grounded_subgoal` prompt to the VLM prefix. It does not let GroundSG select frames and does not let perceptual memory generate a subgoal.
- S and SP use the same dataset GroundSG selection, augmentation, tokenizer, prompt template, and 128-token effective limit. P and SP use identical FrameSamp tensors and Modulator configuration.
- N, S, P, and SP all initialize through the same π0.5 base checkpoint loader. No released S/P weights are merged into SP.

## Runtime and evidence changes

- Evaluation accepts exactly one symbolic source (`oracle`, `qwenvl`, or `none`) and keeps S/SP policy weights identical across Oracle and Qwen.
- Per-episode results and infrastructure failures are separate append-only JSONL files protected by file locks and duplicate keys.
- Qwen uses an exact content-addressed cache keyed by image bytes, task, subgoal history, prompt version, adapter/base hashes, generation settings, and initial-video hash. Raw/parsed responses, tokens, latency, cache/reuse reason, and coordinate validity are logged.
- The GPU gate records checkpoint missing/unexpected keys, input shapes, Action/VLM/memory gradient norms, frozen SigLIP gradient absence, and fixed-noise symbolic/perceptual interventions.

## Safety corrections

- Official GroundSG coordinates are treated as `(y, x)` integer pixels in the 256×256 front image and reversed only for OpenCV drawing.
- The shared S/SP ±8-pixel training augmentation is clipped to `0..255` to prevent invalid edge coordinates.
- Formal results, manifests, configs, checkpoint metadata, audits, and caches use exclusive creation or immutable verification.
