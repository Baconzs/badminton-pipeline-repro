# BST 逐拍击球类型识别

专业分析现在包含一个可选的 [BST: Badminton Stroke-type Transformer](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer) 适配层。它使用当前管线已有的三类信号：

- 双人 YOLO COCO-17 二维骨架；
- TrackNet 的实测羽球点；
- 四点标定投影得到的球员场上位置。

适配层不会把 BST 的类别置信度当作动作好坏分数。BST 结果首先进入独立的击球类型报告；只有它与球路规则一致并通过更严格的 Motion Coach 门槛时，才会成为精确球种建议的类型来源。BST 弱证据不会覆盖独立可靠的球路规则。

## 与 Motion Coach 的联动

专业任务按 **BST → Motion Coach** 的顺序运行，并用 `frame + rally_id + hit_index` 对齐结果。类型报告和训练建议使用两套不同门槛：

| 用途 | Top-1 | Top-1/Top-2 差值 | 姿态覆盖 | 球点覆盖 |
| --- | ---: | ---: | ---: | ---: |
| BST 类型报告 `type_ready`（击球已确认） | ≥60% | 不要求 | ≥8% | ≥8% |
| Motion Coach `coach_ready` | ≥85% | ≥25 个百分点 | ≥85% | ≥55% |

`type_ready` 同样要求击球事件已确认；60% 只是类型报告的其中一项门槛。Motion Coach 进一步要求击球方与模型侧别一致、类别可安全映射，并且 BST 与独立球路规则的 `family` 完全一致（`classification.status="agreement"`）。只有上述条件全部成立，该拍才可能参与建议聚合。

以下结果禁止驱动精确球种建议：

- BST 补充了规则无法确认的类型（`adopted`）；
- BST 与球路冲突（`conflict`）；
- 侧别不一致，或达到基础类型门槛后仍无法安全映射。

策略区分“缺失证据”和“明确反证”：

- BST 卡片缺少事件确认、Top-1 低置信、Top-1/Top-2 区分度不足、覆盖率不足，或 BST 与规则同族但未达到 Motion Coach 专用门槛时，记录为 `rule_fallback`。BST 不驱动文案，独立可靠的球路规则仍可进入相同的姿态和真实球点门槛；管线中的真实击球事件本身仍必须独立确认。
- `conflict`、`side_mismatch` 和高置信 `unsupported` 是明确反证，整拍仍被拦截。
- `adopted` 不能生成精确球种建议；若击球事件、归属、真实球点和阶段姿态独立达标，类型无关的准备/上手时序/恢复信号仍可参与三拍聚合。

所有 BST 结果仍显示在独立技术面板中，便于检查 Top-3 和覆盖率。

通过单拍门槛也不会立即产生卡片。系统按准备衔接、前场球、上手球、后场球和击球恢复等兼容动作组聚合相同信号，至少在 3 个独立击球中重复后，才显示固定“问题 / 训练”结构的 `status="reliable"` 建议。Motion Coach 不展示逐拍分类或证据不足提示。

## 配置

如果使用本仓库内已经准备好的 `third_party/BST-Badminton-Stroke-type-Transformer/` 和 `models/` checkpoint，直接运行下面这一条即可；`run_web_ui.sh` 会自动设置匹配的仓库、权重、模型和序列参数：

```bash
./run_web_ui.sh
```

BST 的研究代码和权重不会由脚本联网下载。若使用外部仓库或另一份 checkpoint，再按上游 README 准备文件，并在启动前设置路径（路径只在服务器进程内使用，不会出现在浏览器接口）：

```bash
export BADMINTON_BST_REPO=/opt/BST-Badminton-Stroke-type-Transformer
export BADMINTON_BST_WEIGHTS=/opt/models/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt
export BADMINTON_BST_DATASET=shuttleset_25
export BADMINTON_BST_MODEL=BST_0
export BADMINTON_BST_POSE_STYLE=JnB_bone
export BADMINTON_BST_SEQ_LEN=100
./run_web_ui.sh
```

脚本会优先使用你显式设置的 `BADMINTON_BST_*` 变量；只需替换 `BADMINTON_BST_REPO` 和 `BADMINTON_BST_WEIGHTS`，其余参数按 checkpoint 实际配置调整即可。启动后访问 `/api/health`，确认 `bst.configured` 为 `true`。

本仓库的 `requirements_repro.txt` 已列出 `positional-encodings` 和 `torchinfo`；如果只安装了最小运行环境，也可以手动执行 `python -m pip install positional-encodings torchinfo`。上游 checkpoint 目前按 Python 3.11/PyTorch 2.x 发布，若严格加载失败请按上游 README 对齐 PyTorch/CUDA 版本。

可用类别配置：

- `shuttleset_25`：合并 ShuttleSet，25 类；
- `shuttleset_35`：完整 ShuttleSet，35 类；
- `badmintondb_18`：BadmintonDB，18 类（类别顺序从上游 CSV 读取）。

`BST_0` 的 forward 不使用球员位置；`BST`、`BST_CG`、`BST_AP`、`BST_CG_AP` 使用位置输入。checkpoint 的架构、类别数、序列长度和 `pose_style` 必须与环境变量完全匹配。若 `weights/` 下存在多个 checkpoint，建议始终显式设置 `BADMINTON_BST_WEIGHTS`，避免误选。

适配器按上游预处理执行：球点按视频宽高归一化，关节按每个 bbox 对角线并中心对齐，脚点投影到 6.10 m × 13.40 m 球场后归一化。BST 面板显示 Top-3、类型置信度、姿态/球点覆盖率和原视频窗口；60% 只是类型报告门槛之一，且不能据此生成 Motion Coach 建议。长视频超过安全上限时会在报告中显示已处理/总击球数，可通过 `BADMINTON_BST_MAX_EVENTS` 调整。

公开权重主要在 ShuttleSet/BadmintonDB 的职业广播域训练；手机机位、不同镜头高度、业余球员和未见类别都可能造成域偏移。当前固定窗口是产品化的近似输入，不等价于上游逐拍裁剪标注，因此模型输出应作为候选而不是正式统计。

## 未配置时的行为

没有 BST 仓库、checkpoint 或其可选依赖时，专业任务仍会完成。报告会显示“未配置 BST”，Motion Coach 使用严格的球路与二维姿态规则，并且仍须通过单拍证据门槛和至少 3 拍同类重复门槛，不生成伪造的模型类型或质量分数。

## 为什么暂不把“动作好坏”接成一个模型分数

公开数据的质量标签含义并不相同：FineBadminton-20K 的 `quality` 更接近战术出球效果；BadminSense 有单拍专家评分但样本小、以 IMU/音频为主且是 CC BY-NC-ND；MultiSenseBadminton 主要覆盖高远球/反手平抽和技能等级。它们都不能直接给单目比赛视频的肘肩时序、击球点和回位质量提供通用可靠分数。因此当前 UI 只显示经过严格门控、同类多拍重复的二维证据建议；后续若有本项目教练标注，应按击球类型分别训练技术质量头，并把战术球质和技术动作质量分开。
