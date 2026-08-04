# Motion Coach：可靠建议契约

Web 版 Motion Coach 采用“宁缺毋滥”的输出策略。单拍姿态规则只产生内部信号，页面不会展示逐拍评价；只有同一球员、同一兼容动作组、同一问题在多拍中稳定重复，才生成面向训练人员的建议。

它不是技术评分器，也不判断握拍、拍面、旋转、三维击球高度或伤病风险。输入仍来自当前管线可核验的信号：已确认击球、TrackNet 真实球点、标定后的球员脚点、YOLO COCO-17 二维姿态，以及可选的 BST 击球类型。

## 生成流程

1. `web_app/stroke_coaching.py` 在击球前约 0.5 秒至击球后约 0.9 秒的稀疏时间线上，提取准备、触球、动力链和恢复指标。腕部位移按逐帧躯干长度归一化，肘、膝和肩髋变化按真实帧间隔换算；动力链峰值只保留为内部证据，不直接生成动作结论。
2. 精确球种信号必须满足可靠性门槛，才有资格参与汇总：击球类型明确且置信度不低于 72，接触窗口至少 3 个可用姿态样本，姿态置信度不低于 0.80，出球段至少包含 6 个真实球点；实际触发问题的膝、支撑面、触球距离或肘腕指标本身也必须在 3 帧中有效，单帧异常不能投票。
3. 击球者必须来自可靠侧别或可见接触姿态。仅靠回合交替补出或纠正的击球方可以用于统计，但不能进入动作建议。
4. 如果该拍匹配到 BST，低置信、区分度不足或覆盖不足只表示 BST 没有提供可用证据，不会否决独立可靠的球路规则。明确的类型冲突、侧别冲突和高置信但无法安全映射的类型仍会拦截建议。
5. 精确球种建议仍要求可靠 `family`。`preparation_support`、`overhead_preparation` 和 `recovery_stability` 使用独立阶段门槛，可在精确球种未知时投票；它们仍必须满足已确认击球、可靠归属、至少 3 个接触姿态样本、阶段姿态置信度不低于 0.80、至少 6 个真实球点及各自阶段样本数。
6. `summarize_reviews()` 按兼容动作组聚合：准备衔接、前场球、上手球、后场球和击球恢复。`clear/drive_clear/drop/smash` 的上手伸展可以合并，前场和后场的同类支撑问题也不会因精确球种拆散；物理击球仍按 `rally_id + hit_index` 去重。
7. 至少 3 个独立可靠击球重复同一问题，才输出一条 `status="reliable"` 的建议。浏览器只渲染可靠聚合建议，其他状态和单拍诊断不会显示成训练文案。

因此，一段视频得到“可靠建议 0 条”是正常结果，表示当前证据尚不足以支持中肯的动作建议。

## BST 专用门槛

BST 的 60% 类型报告门槛只用于独立的击球类型面板。BST 若要成为精确球种建议的类型来源，必须同时满足：

- 击球事件已确认；
- 击球方、模型侧别与姿态归属一致；
- BST 类型可安全映射，并与球路规则的 `family` 一致（`status="agreement"`）；
- Top-1 概率不低于 85%；
- Top-1 与 Top-2 的概率差不低于 25 个百分点；
- BST 序列姿态覆盖率不低于 85%；
- BST 序列球点覆盖率不低于 55%。

`conflict`、`side_mismatch` 和高置信 `unsupported` 是明确反证，会阻断该拍。`adopted` 不能驱动精确球种建议，但可见且独立达标的类型无关阶段信号仍可投票。`low_confidence`、覆盖不足、margin 不足以及同族但未达到专用门槛的结果会记录为 `rule_fallback`：BST 诊断继续保留，可靠球路规则照常接受相同的姿态/轨迹门槛。

## Web 报告 Schema（v2）

```json
{
  "version": 2,
  "status": "ready|insufficient_evidence|unavailable",
  "notice": "string",
  "players": [{
    "id": "near|far",
    "label": "下方球员|上方球员",
    "sample_count": 8,
    "review_count": 5,
    "gated_review_count": 2,
    "evidence_quality": 82,
    "recommendations": [{
      "status": "reliable",
      "priority": "high|medium",
      "family": "overhead",
      "signal": "overhead_extension",
      "problem": "持拍臂连续未充分伸展。",
      "action": "提前到位，在身体前上方拉开触球空间。",
      "repeat_count": 3,
      "title": "上手球 · 依据 3 拍",
      "detail": "兼容旧客户端的简短描述",
      "metric": "3 次重复上手伸展二维信号",
      "evidence": [{
        "frame": 180,
        "seconds": 7.2,
        "timecode": "00:07",
        "window_start": 6.65,
        "window_end": 8.4,
        "real_points": 8
      }]
    }]
  }]
}
```

面向训练人员的固定结构只有两项：

- `problem`：已在同类多拍中重复出现的可见问题；
- `action`：一条可直接练习的动作提示。

`title/detail/metric` 暂时保留用于服务端兼容。HTTP 边界会删除文件路径，限制字符串长度和证据数量，并在 v2 中丢弃所有非 `reliable` 建议。前端还会拒绝缺少 `problem/action` 或包含“待确认、待复核、信息不足、可能”等犹豫措辞的卡片。

## 可审计的内部分类字段

逐拍记录可保留 `classification` 供测试和离线审计，但不作为 Motion Coach 卡片展示：

```json
{
  "source": "bst|rule",
  "status": "agreement|adopted|rule_fallback|conflict|side_mismatch|unsupported|unavailable",
  "advice_eligible": true,
  "coaching_gate": {
    "eligible": true,
    "reason_code": "ready",
    "source": "bst_agreement",
    "bst_confidence": 92.1,
    "bst_margin": 31.4,
    "pose_coverage": 0.91,
    "ball_coverage": 0.68
  }
}
```

`bst_confidence` 和 `bst_margin` 只说明类型分类的确定程度，不是动作优良程度。

阶段特征保存在逐拍内部 `pose.phase_metrics`，包括准备阶段腕高/膝角/支撑宽度、触球窗口躯干归一化腕速、肘/膝/肩髋变化峰值与时间，以及恢复窗口的有效区间和首次连续稳定时间。`pose.phase_gate` 记录持拍臂来源、样本数、姿态置信度和失败原因；最终逐拍记录另有 `evaluation.phase_gate`，把已确认击球和真实球点门槛合并进去。动力链顺序不直接显示为“好坏分数”。

击球归属另有独立的 `attribution_gate`。`rally_alternation_inferred` 和 `rally_alternation_corrected` 固定输出 `eligible=false / reason_code="hitter_attribution_unverified"`；即使球路、姿态和 BST 都达标，也不能参与建议聚合。

## 独立规则模块

`scripts/coaching/motion_coaching.py` 仍是一个不依赖 Web 渲染的 COCO-17 纯规则模块，适合离线实验。它的 `suggestions`、`suppressed_checks` 和置信度字段不等同于 Web v2 的可靠建议契约；接入产品页面前必须经过上述单拍门控与同类多拍聚合，不能直接展示。

## 开源方案参考边界

实现采用了开源运动分析中通用、可独立实现的结构：二维姿态与球场位置融合、以真实击球事件锚定短时间窗、按准备—触球—恢复分段，以及把结构化证据交给固定文案层。BST 只负责击球类型；MIT 许可的 [`tennis-vision-analysis`](https://github.com/Baconzs/tennis-vision-analysis) 提供了“姿态 + 球场 + 时序”组合方式的参考。

本工作区的 `shuttleposereview` 展示过 timing/chain/recovery 启发式，但仓库未提供 LICENSE，也没有可验证的击球锚点、BST 联动或教练质量标注。本实现没有复制其代码、阈值或评分公式，而是基于当前管线信号独立实现尺度归一化时序指标。`TennisExpert` 一类语言建议层只适合消费已经通过门控的结构化信号，因此当前产品继续使用固定“问题 / 训练”模板，不让语言模型推测动作优良程度。
