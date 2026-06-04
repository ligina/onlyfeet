# Sample Difference Diagnostics

This supplementary diagnostic uses existing Clean-P4 test split datasets and existing evaluation prediction CSV files only. No model training or new inference was performed.

Low window-accuracy folders are flagged at window accuracy < 0.90. This threshold is a diagnostic flag, not a statistical test.

## Activity Test Split

### Dataset Distribution

- Windows: 2735
- Unique folders: 105
- Windows per folder: min 4, max 29, mean 26.05, median 29.00

| Class | Windows | Folders |
| --- | --- | --- |
| sitting | 911 | 34 |
| standing | 873 | 35 |
| walk | 951 | 36 |

### Folder-Level Prediction Diagnostics

- Folder-level majority-vote accuracy: 100.00% (105/105 folders)
- Mean folder window accuracy: 99.86%
- Lowest folder window accuracy: 85.71% (data/P4/gravel/standing/P4_gravel_stand_0007, true standing, majority predicted standing)
- Highest folder window accuracy: 100.00% (data/P4/PVC/sitting/P4_pvc_sit_0001)

Folders flagged for low window accuracy:

| Folder | True label | Majority prediction | Windows | Window accuracy | Majority correct |
| --- | --- | --- | --- | --- | --- |
| data/P4/gravel/standing/P4_gravel_stand_0007 | standing | standing | 7 | 85.71% | 1 |

### Class-Level Folder Diagnostics

| Class | Folders | Windows | Mean folder window acc. | Min folder window acc. | Max folder window acc. | Folder majority acc. |
| --- | --- | --- | --- | --- | --- | --- |
| sitting | 34 | 911 | 100.00% | 100.00% | 100.00% | 100.00% |
| standing | 35 | 873 | 99.59% | 85.71% | 100.00% | 100.00% |
| walk | 36 | 951 | 100.00% | 100.00% | 100.00% | 100.00% |

## Surface Test Split

### Dataset Distribution

- Windows: 951
- Unique folders: 36
- Windows per folder: min 4, max 29, mean 26.42, median 29.00

| Class | Windows | Folders |
| --- | --- | --- |
| PVC | 174 | 6 |
| asphalt | 179 | 7 |
| grass | 181 | 7 |
| gravel | 268 | 10 |
| sand | 149 | 6 |

### Folder-Level Prediction Diagnostics

- Folder-level majority-vote accuracy: 100.00% (36/36 folders)
- Mean folder window accuracy: 95.50%
- Lowest folder window accuracy: 55.17% (data/P4/asphalt/walking/P4_asphalt_walk_0002, true asphalt, majority predicted asphalt)
- Highest folder window accuracy: 100.00% (data/P4/PVC/walking/P4_pvc_walk_0001)

Folders flagged for low window accuracy:

| Folder | True label | Majority prediction | Windows | Window accuracy | Majority correct |
| --- | --- | --- | --- | --- | --- |
| data/P4/asphalt/walking/P4_asphalt_walk_0002 | asphalt | asphalt | 29 | 55.17% | 1 |
| data/P4/asphalt/walking/P4_asphalt_walk_0003 | asphalt | asphalt | 29 | 86.21% | 1 |
| data/P4/asphalt/walking/P4_asphalt_walk_0004 | asphalt | asphalt | 29 | 82.76% | 1 |
| data/P4/asphalt/walking/P4_asphalt_walk_0005 | asphalt | asphalt | 29 | 82.76% | 1 |
| data/P4/asphalt/walking/P4_asphalt_walk_0006 | asphalt | asphalt | 29 | 72.41% | 1 |
| data/P4/sand/walking/P4_sand_walk_0002 | sand | sand | 29 | 86.21% | 1 |

### Class-Level Folder Diagnostics

| Class | Folders | Windows | Mean folder window acc. | Min folder window acc. | Max folder window acc. | Folder majority acc. |
| --- | --- | --- | --- | --- | --- | --- |
| PVC | 6 | 174 | 100.00% | 100.00% | 100.00% | 100.00% |
| asphalt | 7 | 179 | 82.27% | 55.17% | 100.00% | 100.00% |
| grass | 7 | 181 | 97.04% | 93.10% | 100.00% | 100.00% |
| gravel | 10 | 268 | 100.00% | 100.00% | 100.00% | 100.00% |
| sand | 6 | 149 | 97.13% | 86.21% | 100.00% | 100.00% |

## Bounded Interpretation

This analysis examines whether Clean-P4 performance is concentrated in a few folders or classes. Folder-level diagnostics reduce the influence of correlated overlapping windows by aggregating predictions within recording folders, but they do not eliminate all temporal-correlation, participant-split, or protocol limitations.

The class distribution diagnostics show the balance of windows and folders across classes. Window imbalance and folder-count imbalance can affect interpretation because aggregate window-level metrics give more influence to classes or folders with more windows. These diagnostics are therefore supplementary and do not replace the main Clean-P4 evaluation.

Participant metadata and collection-location differences should be discussed as limitations, but the small number of participants and limited locations do not support treating participant or location as statistically testable factors here. The results should not be used to claim broad real-world generalization, location-invariant surface recognition, or proof that the surface model learned pure material properties.

## Generated Files

- `sample_difference_diagnostics.md`
- `activity_folder_diagnostics.csv`
- `surface_folder_diagnostics.csv`
- `class_distribution_summary.csv`
- `folder_distribution_summary.csv`
