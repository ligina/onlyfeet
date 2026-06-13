# OnlyFeet 论文实验结果

[English](README.md) | [中文](README.zh-CN.md)

## 项目概述

本仓库包含本科毕业论文 **《Multimodal Human Activity Recognition Using a
Foot-Mounted Edge Sensor Platform》** 的实验脚本与轻量化结果摘要。

论文的最终证据是 `thesis_results_complete_no_dataset/` 中的三折
留一可恢复参与者组交叉验证（LOPGO）结果。**LOPGO 是论文最终评估协议。**
仓库其他位置可能仍保留早期 P4-only 或 Clean-P4 材料，用于历史记录或结果
追溯，但它们不是论文的最终主结果。

项目基于 OnlyFeet 足部安装传感平台评估两个任务：

- 活动识别：walking、standing、sitting。
- 仅步行状态下的地面材质识别：asphalt、PVC、sand、gravel、grass。

## 最终评估协议

最终评估包含三个可恢复参与者组：

- `P1P2`
- `P3`
- `P4`

每一折完整留出一个参与者组作为测试集，并使用其余参与者组进行训练。由于
早期 P1 和 P2 记录在可恢复元数据中无法可靠分离，因此将其合并为 `P1P2`
组。该协议是三折 LOPGO，而不是严格的四受试者 LOSO。

完整模型搜索使用随机种子 42，共运行 342 个实验：

```text
3 folds x 2 tasks x 57 model configurations
```

选定最终模型后，又使用随机种子 42、43 和 44 在全部三折上进行了稳定性
评估。

## 最终主要结果

| 任务 | 最终模型 | 平均 macro-F1 | 关键折级观察 |
| --- | --- | ---: | --- |
| 活动识别 | Gated IMU+audio | 94.81% | P1P2 最难，为 86.13%；P4 最高，为 99.89% |
| 仅步行地面材质识别 | Audio+image concatenation | 97.26% | 各折 macro-F1 范围为 95.81% 至 98.00% |

活动类别为 walking、standing 和 sitting。地面类别为 asphalt、PVC、sand、
gravel 和 grass。地面材质识别仅在步行样本上评估，不能解释为覆盖所有活动
状态的通用地面识别。

权威结果表：

```text
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table1_final_selected_models.csv
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table2_per_fold_selected_models.csv
```

## 合理性诊断

- 标签打乱检查：活动识别平均 macro-F1 下降 52.76 个百分点，地面材质识别
  下降 74.74 个百分点。
- 非重叠窗口诊断：活动识别平均 macro-F1 为 94.79%，地面材质识别为
  97.26%。
- 非重叠结果降低了“直接 50% 窗口重叠是高分主要来源”的担忧，但不能消除
  参与者、会话、地点或实验协议混杂，也不能证明真实世界独立性。

标签打乱结果记录于：

```text
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table3_label_shuffle_sanity.csv
```

## 仓库内容

最终轻量化证据包结构如下：

```text
thesis_results_complete_no_dataset/
├── experiments_lopgo/
├── reports_lopgo_summary/
│   └── thesis_tables/
├── scripts/
├── splits_lopgo/
└── README_RUNBOOK.md
```

重要内容包括：

- `thesis_results_complete_no_dataset/reports_lopgo_summary/all_runs.csv`：
  已完成实验的索引。
- `thesis_results_complete_no_dataset/reports_lopgo_summary/cv_summary_by_config.csv`：
  跨折配置汇总，包含单模态与多模态配置。
- `thesis_results_complete_no_dataset/reports_lopgo_summary/selected_final_3seed_results.csv`：
  最终模型在种子 42、43、44 下的结果。
- `thesis_results_complete_no_dataset/reports_lopgo_summary/label_shuffle_results.csv`：
  标签打乱实验结果。
- `thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/`：
  轻量化论文结果表。
- `thesis_results_complete_no_dataset/splits_lopgo/lopgo/`：三折 LOPGO
  划分定义和摘要。
- `thesis_results_complete_no_dataset/splits_lopgo/random_folder/`：
  补充随机文件夹划分定义。
- `thesis_results_complete_no_dataset/scripts/`：划分生成、数据准备、训练、
  实验编排和结果聚合脚本。
- `thesis_results_complete_no_dataset/experiments_lopgo/`：轻量化逐实验配置、
  摘要、JSON 指标和小型 CSV 报告。
- `thesis_results_complete_no_dataset/README_RUNBOOK.md`：详细协议、证据文件
  说明和复现注意事项。

## 有意排除的内容

以下内容有意不存入 Git：

- 原始传感器、图像和音频数据。
- 准备后的 `.npz` 数据集。
- `.h5`、`.keras` 等训练模型文件。
- 完整的逐窗口预测 CSV。
- 大型运行日志与训练日志。
- PNG 图、NPY 矩阵及其他较大的生成产物。
- 缓存目录和备份文件。

这些排除规则用于控制仓库体积、遵守数据处理与隐私要求，并通过脚本、划分
定义、实验配置和轻量化结果摘要支持结果复核，而不是依赖大型二进制产物。

## 如何使用本仓库

1. 首先查看
   `thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/`
   中的最终报告数值。
2. 查看
   `thesis_results_complete_no_dataset/reports_lopgo_summary/cv_summary_by_config.csv`、
   `thesis_results_complete_no_dataset/reports_lopgo_summary/selected_config_fold_details.csv`
   以及逐实验 JSON 指标，以获取更多追溯信息。
3. 阅读 `thesis_results_complete_no_dataset/README_RUNBOOK.md`，了解详细协议
   和文件映射。
4. 将包含的脚本用作复现流程参考。

仅凭 Git 中的轻量化证据包无法完成完整重跑。完整重跑还需要原始数据或准备
后的数据集，以及相应的本地软件和计算环境。

## 结果解释边界

这些结果是在受控环境下获得的证据，不能视为开放世界鲁棒性或部署就绪的
证明。重要限制包括固定采集地点、规模较小且相对同质的参与者群体、合并的
`P1P2` 组、受控的活动执行方式，以及可能存在的会话或地点特定视觉与声学
线索。
