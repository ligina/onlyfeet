# Split Rules

## Stage 1: Model Selection
- Train split: P1/P2
- Validation split: P3
- Used for modality selection, fusion comparison, and hyperparameter/model selection.

## Stage 2: Final Held-Out Evaluation
- Train split: P1/P2 + P3
- Test split: P4
- P4 is not used for full model search. Only 12 pre-selected candidate models are evaluated on P4.

## Surface Recognition
Surface recognition uses walking-only samples, because foot-ground interaction is mainly observable during walking. Therefore, surface datasets are walking-only subsets of the activity datasets within the same train/test partition.

## Leakage Check
Sanity checks confirmed:
- Stage 2 activity train/test folder overlap: 0
- Stage 2 surface train/test folder overlap: 0

The overlap between activity and surface datasets within the same split is expected because surface recognition is built from walking-only samples.
