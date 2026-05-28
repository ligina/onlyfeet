# Clean-P4 Unified Composite Wrapper

This artifact is an engineering wrapper around two separately trained Clean-P4 final models.
It is not a jointly trained unified model and does not change model weights, data, or reported task results.

## Outputs

- `act_output`: activity recognition from IMU-only input.
- `surface_output`: walking-only surface recognition from image+audio inputs.

## Saved model

- `models/clean_p4_final/unified_composite_cleanp4/unified_composite_cleanp4.keras`

## Evaluation summary

- Activity evaluation status: `completed`
- Surface evaluation status: `completed`

### Activity

- Accuracy: `0.999634`
- Macro-F1: `0.999634`
- N eval: `2735`
- Original vs composite max abs diff: `0`

### Surface

- Accuracy: `0.950578`
- Macro-F1: `0.945858`
- N eval: `951`
- Original vs composite max abs diff: `0.00301384925842`
