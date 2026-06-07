# OnlyFeet 最终实验发布版

[English](README.md) | [中文](README.zh-CN.md)

本仓库包含 **OnlyFeet 本科毕业论文项目** 的最终实验发布材料，包括训练脚本、评估脚本、结果报告、已选最终模型、日志、数据集与划分说明、原始阶段归档材料和环境信息。

项目内部将最终 P4 留出评估协议称为 **Clean-P4**。当前论文主结果证据是 **Clean-P4 final evaluation**。Clean-P4 修正了早期 Stage 2 中 P4 被用作 Keras `validation_data` 的问题：训练期验证只来自 Stage 2 训练数据，P4 只在训练和 checkpoint 选择完成后用于最终评估。

本项目研究基于 OnlyFeet 足部安装采集原型的 **多模态人体活动识别** 和 **步行场景下的地面材质识别**。可用模态包括：

* IMU
* RGB 图像
* 音频
* Time-of-Flight (ToF)
* 磁力计

本发布版的目标是，在受控参与者划分下评估不同任务中的模态有效性和融合行为。本仓库不声称部署就绪、已测量延迟、能耗、电池续航，也不声称具有广泛真实世界鲁棒性。

---

## 数据集可用性

公开的 OnlyFeet 数据集归档托管在 Hugging Face：

[OnlyFeet Dataset on Hugging Face](https://huggingface.co/datasets/ligina592/onlyfeet/tree/main)

代码和实验脚本托管在本 GitHub 仓库中，大型数据集归档单独托管在 Hugging Face。这样的分离方式可以保持 Git 仓库轻量，并避免在 Git 中存储大型二进制数据。

本 GitHub 仓库不直接存储大型 raw/final dataset archive。仓库中可包含 `dataset_card.json` 和 `build_stats.json` 等数据集元数据，用于可追溯性；完整上传的数据集归档应从 Hugging Face dataset repository 获取。

当前 Hugging Face 数据集内容包括 `data_makeup.zip`，大小约 1.13 GB。Hugging Face 页面显示的 license 为 MIT。用户应从 Hugging Face 下载数据集归档，并按项目预期的本地 data layout 放置或解压，然后再运行数据集准备或训练脚本。

```bash
# Download the dataset archive from Hugging Face, then extract it into
# the expected local data directory before running scripts.
huggingface-cli download ligina592/onlyfeet data_makeup.zip --repo-type dataset --local-dir data_external/onlyfeet
```

---

## 1. 仓库结构

```text
onlyfeet/
├── archive_manifest.txt
├── analysis_outputs/
├── datasets_m14_rgb64_stage1/
├── datasets_m14_rgb64_stage2/
├── data_docs/
├── environment/
├── logs/
│   ├── logs_clean_p4_final/
│   ├── stage2/
│   ├── robustness_trainnorm/
│   └── fusion_strategy/
├── models/
│   ├── clean_p4_final/
│   └── final_stage2/
├── requirement/
├── reports/
├── scripts/
├── README.md
└── README.zh-CN.md
```

主要目录说明：

* `scripts/`：数据集准备、模型训练、评估、结果收集、鲁棒性测试、folder-level 评估、sanity check 和 Clean-P4 重跑脚本。
* `analysis_outputs/`：补充分析输出，包括 Clean-P4 supplementary checks 和样本差异诊断。这些文件是辅助诊断，不替代 Clean-P4 主结果证据。
* `datasets_m14_rgb64_stage1/` 与 `datasets_m14_rgb64_stage2/`：包含生成后的 M14 RGB64 Stage 1/Stage 2 数据集元数据卡和 build statistics。本发布版不包含完整训练重跑所需的大型 `.npz` 数组文件。
* `reports/`：Stage 1、原始 Stage 2、最终表格、鲁棒性、folder-level、融合策略、non-overlap 和 sanity check 报告。原始 Stage 2 协议下的报告应按 deprecated/original 边界解释。
* `models/`：Clean-P4 最终模型和旧的原始 Stage 2 traceability 模型。
* `logs/`：关键训练和评估日志，Clean-P4 最终重跑日志位于 `logs/logs_clean_p4_final/`。
* `data_docs/`：参与者定义和数据划分规则。
* `requirement/`：论文要求或提交参考材料，包括 `requirement/Ziang Liu.pdf`。
* `environment/`：Python 版本、依赖、GPU 环境导出文件和存储信息。

---

## 2. Clean-P4 主证据边界

Clean-P4 是论文主最终结果证据。在 Clean-P4 中：

* Stage 2 训练数据用于拟合模型参数。
* 内部验证集只从 Stage 2 训练 NPZ 中划分。
* `ModelCheckpoint`、`ReduceLROnPlateau` 和 `EarlyStopping` 监控内部验证指标，而不是 P4 指标。
* P4 只在训练和 checkpoint 选择完成后加载。
* P4 仅用于最终评估。

原始 `final_stage2` 输出保留用于可追溯性，但它们属于 deprecated/original 证据，因为原始 Stage 2 训练曾将 P4 作为 Keras `validation_data`，使 P4 派生的验证指标可能影响 checkpoint 选择、early stopping 或学习率调度。

---

## 3. 论文主最终结果

论文主最终证据由两个任务专用 Clean-P4 最终模型组成。

| 任务 | Clean-P4 最终模型 | 模态 | 融合方式 | 四舍五入后的 P4 结果 |
|---|---|---|---|---|
| 活动识别 | `stage2_activity_imu_single_seed42_cleanp4` | IMU | single | 约 99.96% accuracy 和约 99.96% macro-F1 |
| 步行地面材质识别 | `stage2_surface_image_audio_concat_seed42_cleanp4` | image, audio | concat | 约 95.06% accuracy 和约 94.59% macro-F1 |

这些结果应解释为 M14 数据集上受控 subject-held-out P4 性能。活动识别结果很高，但不应描述为完美。地面材质识别结果较强，但不应泛化到非步行动作或开放真实场景。

Clean-P4 模型位置：

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/
```

关键支持文件：

```text
models/clean_p4_final/
logs/logs_clean_p4_final/
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

---

## Clean-P4 补充诊断

以下补充诊断已在 Clean-P4 证据链下重新计算：

* 非重叠窗口评估：
  * Activity: 2,735 -> 1,417 windows, 99.93% macro-F1
  * Surface: 951 -> 493 windows, 95.53% macro-F1

* Surface 缺失模态 ablation：
  * Normal image+audio: 94.59% macro-F1
  * No image: 29.25% macro-F1
  * No audio: 19.17% macro-F1

* 模型复杂度：
  * Activity IMU specialist: 99,139 parameters
  * Surface image+audio specialist: 280,613 parameters
  * Composite wrapper: 379,752 parameters

这些诊断是补充证据，不替代 Clean-P4 主最终模型。

报告位置：

```text
Non-overlap report: reports/clean_p4_final/non_overlap/
Surface robustness report: reports/clean_p4_final/robustness_surface/
Model complexity report: reports/clean_p4_final/model_complexity/
Composite wrapper report: reports/clean_p4_final/unified_composite_cleanp4/
```

解释边界：Non-overlap 结果支持高性能并非完全由重叠窗口冗余造成。Surface ablation 表明图像和音频都很重要，但当前模型对完整单模态失效并不鲁棒。Composite wrapper 是工程打包产物，不是 jointly trained multitask model。

---

## 4. 项目目标

本项目评估 OnlyFeet 足部安装采集原型上的多模态识别能力，包含两个主要任务：

1. **常规活动识别**

   * walk
   * standing
   * sitting

2. **步行场景下的地面材质识别**

   * asphalt
   * PVC
   * sand
   * gravel
   * grass

地面材质识别仅基于步行样本，因为足-地交互线索主要在步行过程中出现。本发布版的地面识别结果不应解释为覆盖 standing、sitting、楼梯运动或其他未测试活动的通用地面识别。

---

## 5. 数据集摘要

清洗后的 M14 数据集包含：

```text
Total folder-level recordings: 1,166
P1/P2: 691 folders
P3: 369 folders
P4: 106 folders
```

生成的窗口数据集包含字段：

```text
imu_win
tof_win
mag_win
audio_win
img_win
y_act
y_env
folder
participant
start_ms
```

### 特征表示说明

准备好的数据集中，原始 IMU 窗口字段为 `imu_win`，形状为 `[50, 6]`，对应 accelerometer 和 gyroscope 三轴。模型加载时，训练和评估脚本会额外拼接两个派生 magnitude 通道：accelerometer norm 和 gyroscope norm。因此模型侧 IMU 输入形状为 `[50, 8]`。这是预期行为，不表示数据集和模型不匹配。

该扩展由训练和评估脚本中的 `add_imu_magnitudes()` 执行。

---

## 6. 参与者定义与实验设计

参与者定义：

* `P1/P2`：早期录制中来自 Ziang 和另一名学生的合并组。由于电量耗尽和部分录制缺失，一些记录后续无法可靠拆分为 P1 或 P2。
* `P3`：不同日期录制的独立参与者。
* `P4`：不同日期录制的独立参与者，用作 Stage 2 和 Clean-P4 的最终 held-out participant test set。

实验分为 Stage 1 模型搜索、原始 Stage 2 traceability 和 Clean-P4 最终评估。

Stage 1：

```text
Train: P1/P2
Validation: P3
```

Stage 1 训练 114 个模型，包括 10 个单模态 baseline 和 104 个多模态 fusion 模型。该阶段用于模型、模态和融合策略选择，不是最终论文主证据。

Clean-P4 最终评估：

```text
Training source: Stage 2 dataset_train.npz
Internal validation: folder-level split from Stage 2 dataset_train.npz
Final test: Stage 2 dataset_test.npz, corresponding to P4
```

P4 在 Clean-P4 中不用于训练、验证、checkpoint 选择、early stopping 或学习率调度。

Stage 2 数据规模：

```text
Activity recognition:
Train: 29,788 windows, 1,058 folders
Test: 2,735 windows, 105 folders

Walking-only surface recognition:
Train: 10,299 windows, 371 folders
Test: 951 windows, 36 folders
```

---

## 7. 训练与评估架构

数据集准备阶段将原始多模态 recording folders 转换为 window-level NPZ 数据集。每个 2 秒窗口包含同步模态表示。Activity 标签用于 walk、standing 和 sitting；surface 标签只在 walking windows 上评估。

模型搜索阶段使用 P1/P2 训练、P3 验证，比较单模态 baseline 和多模态 fusion candidate。该阶段只用于模型和模态选择，不作为最终论文证据。

最终训练阶段将 P1/P2 和 P3 合并为训练数据，并重新训练选定的最终配置。Clean-P4 下，内部验证只来自训练数据的 folder-level internal split。

最终 held-out evaluation 中，P4 只在训练和 checkpoint 选择完成后加载，并只用于最终评估。两个最终任务专用模型分别为：

* Activity specialist: IMU-only
* Surface specialist: image+audio concat, walking-only

补充诊断与主最终模型分开报告。Non-overlap evaluation 检查高性能是否主要来自重叠窗口；surface modality ablation 检查 image/audio surface model 对各模态的依赖；model complexity 报告参数量和文件大小；composite wrapper 是工程打包产物，不是 jointly trained model。

---

## 8. 训练模型与输出

Clean-P4 主最终模型：

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/
```

每个 Clean-P4 模型目录包含：

```text
best_model.h5
final_model.h5
metrics.json
train_config.json
history.json
training_log.csv
eval_metrics.json
eval_predictions.csv
eval_classification_report.txt
eval_confusion_matrix.csv
eval_confusion_matrix.png
eval_confusion_matrix_normalized.png
```

Composite wrapper：

```text
models/clean_p4_final/unified_composite_cleanp4/
```

The composite wrapper is an engineering packaging artifact and is not a jointly trained multitask model. 它包装两个最终任务专用模型，不应作为单独科学结果报告。

Deprecated/original Stage 2 模型：

```text
models/final_stage2/stage2_activity_imu_single_seed42/
models/final_stage2/stage2_surface_image_audio_concat_seed42/
```

这些目录保留用于可追溯性，不是论文主最终证据。

---

## 9. 主要报告

原始 final-evaluation 表格位于：

```text
reports/final_eval/
```

Clean-P4 主最终证据应优先查看：

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/train_config.json
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/eval_metrics.json
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/train_config.json
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/eval_metrics.json
logs/logs_clean_p4_final/
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

原始 traceability 报告可查看：

```text
reports/final_eval/table_final_selected_models.csv
reports/final_eval/table_stage2_final_candidates_sorted.csv
reports/final_eval/table_robustness_trainnorm.csv
reports/final_eval/table_early_mid_late_extra.csv
reports/final_eval/table_folder_level_summary.csv
reports/nonoverlap_windows/summary_nonoverlap_windows.csv
reports/sanity_checks/folder_overlap_summary.csv
reports/sanity_checks/dataset_split_summary.csv
```

---

## 10. 脚本

主要脚本位于：

```text
scripts/
```

关键 Clean-P4 脚本：

```text
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

补充 Clean-P4 诊断脚本：

```text
scripts/run_cleanp4_supplementary_checks.py
scripts/94_eval_clean_p4_non_overlap.py
scripts/95_eval_clean_p4_surface_robustness.py
scripts/96_collect_clean_p4_model_complexity.py
scripts/create_clean_p4_composite_model.py
```

输出位置：

```text
reports/clean_p4_final/non_overlap/
reports/clean_p4_final/robustness_surface/
reports/clean_p4_final/model_complexity/
models/clean_p4_final/unified_composite_cleanp4/
reports/clean_p4_final/unified_composite_cleanp4/
```

典型 Clean-P4 主模型重跑命令：

```bash
bash scripts/run_32_stage2_clean_p4_final_models.sh
```

完整重跑需要准备好的 Stage 2 NPZ 数据集。

---

## 11. 环境

环境文件位于：

```text
environment/
```

包含：

```text
python_version.txt
pip_freeze.txt
nvidia_smi.txt
storage_info.txt
```

主要实验运行在配备 RTX 5090 GPU 的云机器上。

---

## 12. 重要限制与推荐解释

Clean-P4 最终 P4 结果很高：

```text
Activity: about 99.96% accuracy and about 99.96% macro-F1
Walking-only surface: about 95.06% accuracy and about 94.59% macro-F1
```

这些是论文主最终结果。不要用旧的原始 Stage 2 100.00% activity 或 99.26% walking-only surface 值替代它们。

Sanity checks 显示 Stage 2 train/test folder overlap 为 0。但数据采集环境仍相对受控，参与者数量有限，P1/P2 是合并组，窗口样本存在时间相关性，surface recognition 只覆盖 walking-only 样本，且场地特定视觉或声学线索无法完全排除。

Clean-P4 folder-level evaluation 未重新计算，因此不纳入 supplementary diagnostics。Window-level Clean-P4 non-overlap、surface robustness 和 model-complexity diagnostics 位于 `reports/clean_p4_final/`。

推荐解释：

1. Activity recognition 在该受控 M14 任务中强烈依赖 IMU，Clean-P4 final activity model 是 IMU-only 模型，并在 P4 activity windows 上表现很强。
2. Walking-only surface recognition 在选定最终模型中受益于 image+audio concat fusion。
3. Clean-P4 是主最终证据边界：P4 只用于最终评估，训练期验证来自 Stage 2 training data。
4. 原始 Stage 2 和原始诊断输出只作为 traceability 使用，除非已在 Clean-P4 下重新运行。
5. 最终结果应以受控 participant-held-out performance 解释，而不是完整真实世界泛化。
