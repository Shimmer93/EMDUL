# 在 EMDUL 复现 MM-Fi + HmPEAR

文中以 `logs/` 开头的路径指向实验机器上的权重和日志。使用新克隆的仓库时，可以按下方「运行」一节训练，再用自己生成的第二阶段 checkpoint 测试。

MM-Fi + HmPEAR 训练、跨域到 mmBody，固定两阶段配方、三次随机初始化的 MPJPE 为 **15.47 ± 0.68 cm**（均值 ± 样本标准差），比论文 **14.89 cm** 高 **0.58 cm**。最接近的一次为 **14.97 cm**，高 **0.08 cm**。同一份权重在 MM-Fi 上为 10.07 cm。下表为 15 个 ITOP 关节的全量测试，单位为厘米。[论文表 2](https://arxiv.org/html/2603.14507v3#S6.T2) · [各次权重、原始指标和汇总](logs/crossdomain_20261125/refinement_seeds.json)

| 训练设置 | MM-Fi MPJPE | MM-Fi PA-MPJPE | mmBody MPJPE | mmBody PA-MPJPE |
| --- | ---: | ---: | ---: | ---: |
| 论文表 2，HmPEAR | 10.06 | 7.01 | 14.89 | 11.11 |
| 五帧基线，种子 3407 | 10.02 | 6.92 | 16.47 | 11.89 |
| 关闭四项 LiDAR 增强，八轮，种子 3407 | 10.02 | 6.92 | 15.23 | 11.35 |
| 关闭四项增强并低学习率续训，种子 3407 | 10.07 | 6.97 | 15.20 | 11.14 |
| 同一两阶段配方，种子 3408 | 10.04 | 6.95 | 16.25 | 11.75 |
| 同一两阶段配方，种子 3409 | 10.07 | 6.97 | **14.97** | 10.95 |
| 两阶段配方，三次均值 ± 样本标准差 | 10.06 ± 0.02 | 6.96 ± 0.01 | **15.47 ± 0.68** | 11.28 ± 0.42 |

三次均用种子 3407 选出的同一批 97 段标注，以相同测试种子对导出的学生权重做全量评估。仅训练种子变化，影响网络初始化、数据顺序和训练中的随机变换。学生与教师从零开始，两阶段各自由 HmPEAR 留出集选取权重。比较训练方案时看过 mmBody 结果，训练参数只由源域数据更新。域内 MPJPE 在三次之间相差 0.03 cm，跨域相差 1.27 cm。[种子 3409 的学生权重](logs/crossdomain_20261125/refinement_seed3409/student.ckpt) · [初始化来源及校验值](logs/crossdomain_20261125/refinement_seed3409/student_provenance.json) · [该权重 MM-Fi 测试](logs/crossdomain_20261125/refinement_seed3409_mmfi/metrics.csv) · [mmBody 测试](logs/crossdomain_20261125/refinement_seed3409_mmbody/metrics.csv)

## 起点与历史权重

最初的 EMDUL [五帧基线](cfg/emdul/mmfi_hmpear_5frames.yml) 从零训练八轮，两套网络都随机初始化。验证集选中最后一轮，累计 11,880 次参数更新。MM-Fi 上得到 **10.02 / 6.92 cm**，mmBody 上得到 **16.47 / 11.89 cm**。与论文相比，域内 MPJPE 低 0.04 cm，跨域 MPJPE 高 1.58 cm。独立加载导出的学生权重重测 MM-Fi 后，两个指标舍入到 0.01 cm 均一致。

- [本次训练的推理权重](logs/reproduction_20261124/mmfi_hmpear_student.ckpt)
- [完整训练 checkpoint](logs/mmfi_hmpear_reproduction/native8/P4Transformer-epoch=7-val_mpjpe=0.0688.ckpt)
- [训练曲线及 MM-Fi 测试原始值](logs/mmfi_hmpear_reproduction/native8/metrics.csv) · [mmBody 测试原始值](logs/mmfi_hmpear_reproduction/native8_mmbody/metrics.csv) · [导出权重重测](logs/mmfi_hmpear_reproduction/native8_retest/metrics.csv)

同一份本地 CVPR 权重在 EMDUL 中重测得到 **9.99 / 6.96 cm** 和 **14.87 / 11.10 cm**。这组权重来自原有实验。导出的学生模型在 EMDUL 中通过全量重测，来源和校验值保存在 [reference_provenance.json](logs/reproduction_20261124/reference_provenance.json)。本次新训练与旧权重复测的指标分别写入 [results.json](logs/reproduction_20261124/results.json)，各项指标指向原始 CSV。

## 跨域差距的训练对照

2025-10-27 的开发实验在 TensorBoard 中记录了 mmBody **14.89185 / 11.10916 cm**，全程 100 轮、24,000 次参数更新。原实验选取训练池最前面的 97 段 MM-Fi 标注序列，共 25,673 帧。八轮基线随机抽取的 97 段共有 23,761 帧，仅 10 段重合。[运行参数、原始指标和标注子集](logs/crossdomain_20261125/protocol.json)

两套 HmPEAR 数据的 368 段、42,983 帧原始 XYZ 全部相同。开发版预处理保存的骨架等于原始骨架转为 ITOP 后去掉每段末帧。按开发版本 `809e4b3` 和当时保存的参数回放，抽查的 32 个训练窗口中，加噪点、缩放、旋转和坐标抖动均未改变模型读取的主点云。原因是同一增强先处理另一份点云，且每次都从相同的随机种子开始。旧版在窗口所有帧均达到 128 点时才稀疏采样。该训练集上主输入的有效保留比例约为 95.69%。八轮基线逐帧判定点数，并从 12.5% 至 100% 随机取保留比例。固定同一批 684 个 HmPEAR 源域窗口与流筛选结果，以每帧最多 256 点计，开发版抽样平均每帧保留 149.39 点，八轮基线为 110.30 点。[增强回放及原始数据一致性记录](logs/crossdomain_20261125/reference_augmentation_probe.json)

八轮基线的加噪点标准差设为 0.2 m，噪声中心在空间中随机取值。论文附录表 9 给出 0.02 m，中心取自骨架。基线还在补齐到 256 点之后对每一个副本独立抖动。用包含 32 个不同点的输入验证，先抖动再重复补点会保留 32 个不同 XYZ，先补点再抖动则变成 256 个不同 XYZ。这些是配置和实现差异，每一项的独立跨域收益仍需单独测量。[论文附录表 9](https://arxiv.org/html/2603.14507v3#S9.T9) · [补点与抖动顺序验证](logs/crossdomain_20261125/augmentation_order_probe.json)

对八轮基线追加六轮仅使用源域数据的训练，同时增大方位角扰动、提高静态回波保留率和 LiDAR 监督权重。HmPEAR 验证 MPJPE 从 6.88 降至 6.36 cm，验证选中的同一份学生权重在 MM-Fi 上为 **10.16 / 6.90 cm**，在 mmBody 上为 **17.59 / 12.54 cm**。这组改动加大了跨域差距。[续训配置](cfg/emdul/mmfi_hmpear_crossdomain.yml) · [MM-Fi 日志](logs/crossdomain_20261125/density/metrics.csv) · [mmBody 日志](logs/crossdomain_20261125/density_mmbody/metrics.csv)

八轮消融实验只关闭 LiDAR 主输入的加噪点、缩放、旋转和坐标抖动，保留原来的平移、流筛选和稠密帧抽样。与八轮基线固定同一个种子、同样的 97 段标注数据、11,880 次更新和测试流程。两套网络从零训练，HmPEAR 验证选中第 8 轮。MM-Fi 为 **10.02 / 6.92 cm**，mmBody 为 **15.23 / 11.35 cm**。四项增强合并关闭后，该种子上跨域 MPJPE 改善 **1.24 cm**，与论文相差 **0.34 cm**。各项增强的单独贡献仍未测量。[配置](cfg/emdul/mmfi_hmpear_5frames_lidar_ablation.yml) · [MM-Fi 日志](logs/crossdomain_20261125/augmentation_ablation/metrics.csv) · [mmBody 日志](logs/crossdomain_20261125/augmentation_ablation_mmbody/metrics.csv) · [学生权重](logs/crossdomain_20261125/augmentation_ablation/student.ckpt)

单独加载导出的学生权重重测，两套测试集的四项指标均复现到小数点后两位。让八轮基线与关闭四项增强的权重在每一帧上使用完全相同的随机补点后，mmBody 的 14 段测试序列均改善，按帧加权平均降低 1.24 cm。[导出权重重测 MM-Fi](logs/crossdomain_20261125/exported_ablation_mmfi/metrics.csv) · [重测 mmBody](logs/crossdomain_20261125/exported_ablation_mmbody/metrics.csv) · [逐序列配对结果](logs/crossdomain_20261125/paired_ablation_mmbody.csv)

在该消融配置上只将 LiDAR 稀疏采样换为开发版的整窗口判定与 95.69% 保留率，从零训练同样八轮，MM-Fi 为 **10.02 / 6.93 cm**，mmBody 为 **15.39 / 11.18 cm**。PA-MPJPE 改善了 0.17 cm，MPJPE 却升高了 0.16 cm。[抽样配置](logs/crossdomain_20261125/sampling_ablation.yml) · [MM-Fi 日志](logs/crossdomain_20261125/sampling_ablation/metrics.csv) · [mmBody 日志](logs/crossdomain_20261125/sampling_ablation_mmbody/metrics.csv)

改用按当时保存的开发代码推导出的 MM-Fi 标注子集，以 HmPEAR 遍历一轮作为 epoch，单卡 batch 32，从零训练 24 轮。HmPEAR 验证选中第 23 轮，MM-Fi 为 **11.86 / 8.61 cm**，mmBody 为 **16.71 / 12.33 cm**。这组实验同时更改标注子集、源域取样方式、流筛选帧和训练安排，不能用它单独估计标注子集的影响。就标注覆盖而言，按 MM-Fi 的环境分组统计，前缀子集的 97 段全部来自环境 1–2，随机子集在环境 1–2、3、4 的计数分别为 57、20、20。实际训练和测试采用 RDN-P3 划分。[配置](cfg/emdul/mmfi_hmpear_source_order.yml) · [MM-Fi 日志](logs/crossdomain_20261125/historical/metrics.csv) · [mmBody 日志](logs/crossdomain_20261125/historical_mmbody/metrics.csv)

在八轮关闭四项增强的权重上，重建优化器，以 1.6e-5 起始学习率追加六轮，保留同一套标注、无标签数据、LiDAR 转换和验证流程。累计 20,790 次参数更新。MM-Fi 为 **10.07 / 6.97 cm**，mmBody 为 **15.20 / 11.14 cm**，两项跨域指标分别高于论文 0.31 和 0.03 cm。导出的学生权重通过两套完整测试集重测。[续训配置](logs/crossdomain_20261125/ablation_refinement.yml) · [权重及来源](logs/crossdomain_20261125/ablation_refinement/student_provenance.json) · [MM-Fi 重测](logs/crossdomain_20261125/exported_refinement_mmfi/metrics.csv) · [mmBody 重测](logs/crossdomain_20261125/exported_refinement_mmbody/metrics.csv)

用相同点样本逐帧比较，续训后的模型在 mmBody 全部 14 段序列上均优于原来的 16.47 cm 基线。与旧权重相比，续训模型的鼻部和右肩误差分别高 1.70 和 1.55 cm，右腕和左腕误差分别低 1.65 和 1.50 cm。剩余的平均差距在关节之间并不均匀。[逐序列误差](logs/crossdomain_20261125/paired_refinement_mmbody.csv) · [逐关节误差](logs/crossdomain_20261125/paired_refinement_joints.csv)

固定关闭四项增强的八轮配方，将 HmPEAR 训练及验证中的流筛选换成开发实现的组合，使用位移起始帧的点和骨架、仅由点云确定的边界角点、空筛选时有放回补点。结果为 MM-Fi **10.00 / 6.91 cm**，mmBody **15.61 / 11.46 cm**。流筛选的局部实现已在运动窗口与静止窗口用固定随机种子逐点对照。[转换配置](logs/crossdomain_20261125/fpf_source.yml) · [MM-Fi 日志](logs/crossdomain_20261125/fpf_source/metrics.csv) · [mmBody 日志](logs/crossdomain_20261125/fpf_source_mmbody/metrics.csv)

在相同八轮预算下，只将关闭四项增强的模型改用竖直高度特征（索引 1），MM-Fi 为 **10.05 / 6.93 cm**，mmBody 为 **15.48 / 11.06 cm**。保留高度特征，在 LiDAR 流程中启用按骨架中心加噪点、补点前抖动、绕竖直轴旋转及随机缩放，MM-Fi 为 **10.04 / 7.00 cm**，mmBody 为 **15.36 / 11.09 cm**。两组均从随机权重开始，由 HmPEAR 选中第 8 轮。[仅改高度的测试](logs/crossdomain_20261125/height_only_mmbody/metrics.csv) · [高度与噪声调整的测试](logs/crossdomain_20261125/height_noise_mmbody/metrics.csv) · [双域结果与配置](logs/crossdomain_20261125/results.json)

### 完整训练预算

完整预算配置用 seed 3407 随机选取与八轮基线相同的 97 段 MM-Fi 标注，其余 875 段用作无标签数据。HmPEAR 按开发数据有效长度遍历，每轮取 30,738 个监督窗口，独立验证集共 11,649 个窗口。每卡 batch 64、双卡全局 batch 128，训练集丢弃不足一个 batch 的尾部，恰好每轮 240 次更新。学生和教师随机初始化，优化器为 AdamW，预热 20 轮，总计 100 轮。LiDAR 采用已实测的开发版主输入行为，包括起始帧流筛选、整窗口稀疏采样和固定小幅平移。现代 EMDUL 的时间一致性损失仍可更新教师。训练轮次由 HmPEAR 验证误差选择。[完整配置](cfg/emdul/mmfi_hmpear_full.yml) · [样本数及流筛选校验](logs/crossdomain_20261125/full_protocol.json)

100 轮共 24,000 次更新。第 91 轮（21,840 次更新）取得最低 HmPEAR 验证 MPJPE 6.10 cm。以同一 checkpoint 单卡测试，MM-Fi 为 **11.83 / 8.38 cm**，mmBody 为 **16.41 / 10.96 cm**。mmBody 的对齐姿态误差低于论文，未对齐 MPJPE 仍高 1.52 cm。这组结果说明恢复更新次数和历史 LiDAR 抽样不足以恢复两域主指标。[MM-Fi 单卡结果](logs/crossdomain_20261125/full100_mmfi/metrics.csv) · [mmBody 单卡结果](logs/crossdomain_20261125/full100_mmbody/metrics.csv) · [权重来源](logs/crossdomain_20261125/full100/student_provenance.json)

在每条非空的 MM-Fi 源域训练或验证序列取八个内部窗口，冻结权重后比较预测误差。仅被旧实验选作有标签数据的 87 条序列上，旧权重为 0.86 cm，新 100 轮权重为 11.21 cm。仅被新实验选中的 85 条非空序列上，两者分别为 9.59 和 0.88 cm。两边都未选的 787 条上分别为 9.69 和 11.62 cm。这一差异印证了两次训练实际拟合的标注序列不同。这里读取标签仅用于事后计算误差。[源域逐序列结果](logs/crossdomain_20261125/source_fit_by_sequence.csv)

### 按历史训练流程重新初始化

另做一组对照，使用当时保存的训练参数和 `809e4b3` 版数据加载器、增强及损失，学生和教师均由本仓库的 P4Transformer 随机初始化。训练种子设为 3407，历史主程序未记录种子。核对两个实现的全部初始参数后，在两个五帧点云窗口上逐元素比较，FP32 预测完全一致。完整训练采用同一份预处理后的 HmPEAR 数据、前缀 97 段 MM-Fi 标注和双卡全局 batch 128。无标签加载器从整个 MM-Fi 源域训练池采样，其中包括这 97 段。历史实现对两帧教师预测停止梯度，时间损失不更新教师。学生通过伪标签损失更新。这个对照保留了上述与论文描述、当前实现的差异。[训练入口](logs/crossdomain_20261125/replay_training.py) · [运行与测试脚本](logs/crossdomain_20261125/run_replay.sh) · [初始化及前向核对](logs/crossdomain_20261125/replay_smoke.log)

100 轮共 24,000 次更新。第 10 轮 HmPEAR 验证 MPJPE 为 9.90 cm，旧实验同轮为 9.92 cm。本次验证最优为第 78 轮的 6.55 cm（18,720 步），旧实验最优为第 100 轮的 6.50 cm。只导出本次第 78 轮学生的 65 个参数张量，用 EMDUL 的同一五帧测试流程得到 MM-Fi **9.94 / 6.93 cm**、mmBody **15.34 / 11.64 cm**，跨域 MPJPE 比论文高 **0.45 cm**。沿用历史标注划分与训练实现后，此种子仍有余差。[新权重与来源](logs/crossdomain_20261125/replay100/student_provenance.json) · [MM-Fi 全量测试](logs/crossdomain_20261125/replay100_mmfi/metrics.csv) · [mmBody 全量测试](logs/crossdomain_20261125/replay100_mmbody/metrics.csv)

额外固定在第 100 轮取权重，与旧实验使用相同的 24,000 次更新。MM-Fi 为 **9.95 / 6.92 cm**，mmBody 为 **15.37 / 11.67 cm**。将取权重时间固定到原实验的末轮，跨域 MPJPE 仍有 0.48 cm 差距。[固定末轮 MM-Fi](logs/crossdomain_20261125/replay100_final_mmfi/metrics.csv) · [固定末轮 mmBody](logs/crossdomain_20261125/replay100_final_mmbody/metrics.csv)

## 原配置对照

用户指定的 [mmfi_hmpear.yml](cfg/emdul/mmfi_hmpear.yml) 也从零训练了 13 个完整 epoch，保留原有 100 轮学习率安排。前 13 轮中 HmPEAR 验证最优的是第 13 轮。在 MM-Fi 上得到 **11.12 / 7.24 cm**，使用仓库 mmBody 原生五帧测试配置得到 **18.48 / 12.78 cm**。五帧八轮实验完成后结束这项对照，训练状态已保存（19,305 步，含优化器及学习率调度器）。[MM-Fi 实测](logs/mmfi_hmpear_reproduction/aggregate13_mmfi/metrics.csv) · [mmBody 实测](logs/mmfi_hmpear_reproduction/aggregate13_mmbody/metrics.csv) · [续训权重](logs/mmfi_hmpear_reproduction/scratch100/last.ckpt)

## 为什么两份仓库会有差距

1. 学生原来只直接接受 LiDAR 监督，教师只直接接受 MM-Fi 监督，教师的两帧伪标签预测又包在 no_grad 中。修复后的学生和教师都接受两种有标签数据，时间一致性损失可更新教师，伪标签监督在目标端停止梯度。改变 MM-Fi 标签或 LiDAR 标签的梯度实验可见 [gradient_audit.txt](logs/reproduction_20261124/gradient_audit.txt)。
2. 任务开始时的工作树把 ref_pipeline 指向验证流程，近期多次训练跳过了 LiDAR 转换。修复配置恢复了噪声点、运动筛选和随机扰动。五帧配置还在补点前按附录表 9 对稠密帧抽样。时间损失改用末帧点云与末帧预测配对，并按欧氏距离（米）判定动态和静态关节。旧 CUDA 实现返回平方距离。
3. 旧权重的输入通道有版本依赖。在同一份 2025-10-09 LEMT 权重、相同的测试序列与处理流程下，第 4 列 Doppler 得到 11.20 cm，第 3 列坐标得到 22.44 cm。LEMT 在 2025-10-10 才提交了通道切换。主配置用竖直坐标（索引 1），五帧开发配置用索引 2。配置必须跟权重一起保存。
4. 两份仓库使用相同的数据文件，但 LEMT 旧实验取已排序训练序列的前 10%，与当前 seed 3407 的随机子集只重合 10 / 97 条。部分 LEMT 运行还用 MM-Fi 测试集挑选 checkpoint。当前训练用独立的 HmPEAR 留出集选模。
5. MM-Fi 的邻帧聚合和 mmBody 的原生五帧输入是两种评估协议。历史 LEMT 的三帧切片实际拼了两帧，EMDUL 主配置拼三帧，五帧配置直接使用每帧的点云。历史权重若跨协议测试，误差可能大幅变化。

这些实验直接验证了梯度路径和输入通道的作用。历史训练的差异包括多项同时改变的设置，单项改动的训练收益不能据此分摊。

## 运行

在本仓库根目录激活 emdul 环境。数据为 data/mmfi.pkl、data/hmpear.pkl 和 data/mmbody.pkl。本次单卡为 RTX 4090 D，PyTorch 2.6.0，Lightning 2.5.5。以下命令指定新的日志版本，避免覆盖原始结果。

复跑这套两阶段配方时，第二阶段需要第一阶段验证最优的完整 checkpoint，其中包括教师与学生参数。第二阶段以新优化器开始。示例采用训练种子 3409，改用 3407 或 3408 可以复跑另两组。三组配置中的标注子集种子均固定为 3407。

```bash
conda activate emdul
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
SEED=3409
python main.py -c cfg/emdul/mmfi_hmpear_5frames_lidar_ablation.yml -g 1 -w 6 -b 16 -e 64 \
  --seed "$SEED" --pin_memory --exp_name crossdomain_repeat --version "stage1_$SEED"
STAGE1=(logs/crossdomain_repeat/stage1_"$SEED"/P4Transformer-epoch=*.ckpt)
python main.py -c cfg/emdul/mmfi_hmpear_5frames_refinement.yml -g 1 -w 6 -b 16 -e 64 \
  --seed "$SEED" --pin_memory --checkpoint_path "${STAGE1[0]}" --exp_name crossdomain_repeat --version "stage2_$SEED"
```

最初 16.47 cm 基线的训练命令如下。

```bash
conda activate emdul
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
python main.py -c cfg/emdul/mmfi_hmpear_5frames.yml -g 1 -w 6 -b 16 -e 64 \
  --pin_memory --exp_name mmfi_hmpear_reproduction --version native8_repeat
```

原配置的 100 轮训练，将上条命令的 -c 换为 cfg/emdul/mmfi_hmpear.yml，并更换版本名即可。--resume 指向 last.ckpt 时恢复包括优化器在内的训练状态。--checkpoint_path 用于从权重初始化或测试。

以下命令重测上述两阶段训练导出的推理模型。基线与旧 CVPR 权重也适用相同的五帧测试流程。

```bash
CKPT=logs/crossdomain_20261125/refinement_seed3409/student.ckpt
CFG=cfg/emdul/mmfi_hmpear_5frames_refinement.yml
python main.py --test -c "$CFG" --checkpoint_path "$CKPT" --seed 3407 -g 1 -w 6 -e 64 \
  --exp_name mmfi_hmpear_reproduction --version verify_mmfi
python main.py --test -c "$CFG" --checkpoint_path "$CKPT" --seed 3407 -g 1 -w 6 -e 64 \
  --test_data_path data/mmbody.pkl --test_split test \
  --exp_name mmfi_hmpear_reproduction --version verify_mmbody
```

MM-Fi 有标签 / 无标签 / 测试分别包含 97 / 875 / 108 条不重叠的序列。HmPEAR 的 train + val 提供 31,194 帧监督，HmPEAR 的 test 用于验证选模。最终测试覆盖 MM-Fi 24,895 帧、mmBody 28,109 帧。本地仓库和论文附录 9.2 均使用真实骨架中心做归一化，包括无标签及测试窗口。这是解释绝对位置误差和部署能力时必须保留的评估条件。[训练与评估协议](logs/reproduction_20261124/protocol.md)
