# Clean-P4 Supplementary Checks

These outputs are supplementary diagnostics only. They do not replace the main Clean-P4 window-level evaluation and do not constitute new training experiments.

## Data Sources

### Activity
- Dataset path: `D:\thl_thesis\onlyfeet\datasets_m14_rgb64_stage2\activity\dataset_test.npz`
- Dataset card: `D:\thl_thesis\onlyfeet\datasets_m14_rgb64_stage2\activity\dataset_card.json`
- Prediction source: `D:\thl_thesis\onlyfeet\models\clean_p4_final\stage2_activity_imu_single_seed42_cleanp4\eval_predictions.csv`
- Existing final model metrics source: `D:\thl_thesis\onlyfeet\models\clean_p4_final\stage2_activity_imu_single_seed42_cleanp4\eval_metrics.json`
- Class names: walk, standing, sitting
- Windows: 2735
- Unique folders: 105

### Surface
- Dataset path: `D:\thl_thesis\onlyfeet\datasets_m14_rgb64_stage2\surface\dataset_test.npz`
- Dataset card: `D:\thl_thesis\onlyfeet\datasets_m14_rgb64_stage2\surface\dataset_card.json`
- Prediction source: `D:\thl_thesis\onlyfeet\models\clean_p4_final\stage2_surface_image_audio_concat_seed42_cleanp4\eval_predictions.csv`
- Existing final model metrics source: `D:\thl_thesis\onlyfeet\models\clean_p4_final\stage2_surface_image_audio_concat_seed42_cleanp4\eval_metrics.json`
- Class names: asphalt, PVC, sand, gravel, grass
- Windows: 951
- Unique folders: 36

## Results

### Activity
- Existing Clean-P4 window-level final model accuracy: 99.96%
- Existing Clean-P4 window-level final model macro-F1: 99.96%
- Folder-level majority-vote accuracy: 100.00%
- Folder-level majority-vote macro-F1: 100.00%
- Folders used: 105
- Window-level majority-class baseline: majority class `walk`, accuracy 34.77%, macro-F1 17.20%
- Folder-level majority-class baseline: majority class `walk`, accuracy 34.29%, macro-F1 17.02%

### Surface
- Existing Clean-P4 window-level final model accuracy: 95.06%
- Existing Clean-P4 window-level final model macro-F1: 94.59%
- Folder-level majority-vote accuracy: 100.00%
- Folder-level majority-vote macro-F1: 100.00%
- Folders used: 36
- Window-level majority-class baseline: majority class `gravel`, accuracy 28.18%, macro-F1 8.79%
- Folder-level majority-class baseline: majority class `gravel`, accuracy 27.78%, macro-F1 8.70%

## Interpretation

- Folder-level majority vote reduces the influence of multiple correlated windows from the same recording folder by aggregating predictions before computing metrics.
- The majority-class sanity baseline checks whether performance can be explained by class imbalance alone.
- These checks support interpretation of the Clean-P4 evaluation, but they do not remove all limitations of the controlled collection protocol.
- These diagnostics do not replace the main Clean-P4 window-level evaluation.

## Warnings and Uncertainties

- Existing Clean-P4 prediction files were found and used; no inference or training was run.
- Folder-level prediction ties are resolved by selecting the smallest class index.
- activity: no true-label inconsistencies, prediction ties, or CSV/NPZ label mismatches were detected.
- surface: no true-label inconsistencies, prediction ties, or CSV/NPZ label mismatches were detected.
