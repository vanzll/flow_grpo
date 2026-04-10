# Offline Prior DiT Change Log

Date: 2026-04-10

Scope:
- Only the offline Prior DiT training path was changed.
- Files affected:
  - `scripts/train_prior_dit_offline.py`
  - `config/prior_dit_offline.py`

Requested changes applied:

1. Batch size increased
- `offline.train_batch_size`: `16 -> 64`
- `offline.val_batch_size`: `16 -> 64`

2. Weight transform removed exponential
- `offline.weight_transform`: `"exp" -> "identity"`

3. Signed advantage used directly as loss weight
- `offline.score_source`: `"advantages"`
- `offline.positive_only`: `False`
- `offline.normalize_weights`: `False`
- `offline.min_weight`: `0.0`
- `offline.score_clip`: `0.0`

Result:
- Loss now uses raw signed `advantage` as the sample weight.
- Positive advantage pushes the model toward the sample.
- Negative advantage pushes the model away from the sample via the signed weighted MSE objective.

4. Regularization disabled
- `prior_dit.v_reg_weight`: `0.01 -> 0.0`

5. Gradient clipping removed
- Training loop no longer calls `clip_grad_norm_`.
- Logged `grad` is now the raw gradient norm before the optimizer step.

6. Disable small output-head initialization for offline training
- Added configurable output-head init in `flow_grpo/prior_dit.py`.
- Offline config now sets:
  - `prior_dit.small_init_output = False`
  - `prior_dit.output_init_std = 1e-4`

Result:
- Offline Prior DiT no longer forces the final projection head to start near zero.
- This removes the explicit bias toward the zero-velocity-field initialization for the offline experiment.

Operational note:
- `epoch 0` evaluation is still skipped.
- Terminal training progress still shows:
  - `loss`
  - `mse`
  - `reward`
  - `weight`
  - `grad`
- Each batch now also prints a dedicated log line:
  - `Epoch {epoch} batch {step}/{total}: loss=... mse=... reward=... weight=... grad=...`
