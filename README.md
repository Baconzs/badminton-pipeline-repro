# 羽毛球比赛视频智能分析系统

**简体中文** · [English](README_EN.md)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ychenfen/badminton-pipeline-repro/pulls)

把一段普通的羽毛球比赛视频，转换成带**运动员轨迹、移动速度、累计跑动距离、羽毛球飞行轨迹**的可视化分析视频。当前默认输出普通速度分析视频；电影级"子弹时间"/慢动作特效可通过 Web UI 的“子弹镜头特效”按钮或命令行开关显式开启。

整套系统基于 **TrackNet（球检测） + YOLOv8s-pose（球员姿态） + ByteTrack（多目标跟踪） + 透视矫正**，在 Apple Silicon Mac 上跑通过完整流程。

![效果演示](docs/images/demo.gif)

---

## 这个项目解决了什么

市面上"开源羽毛球分析"项目通常只能做以下其中一项：
- 只检测球，没有球员分析
- 假设俯视机位（真实比赛视频几乎都是斜拍）
- 写死 Windows 路径，Mac/Linux 跑不通
- 阈值硬编码，在真实视频上静默失败（球检测率 0%）

这个仓库是**完整跑通、所有 bug 都修过、Mac 优先**的版本。每个修复都记录在 [HANDOVER.md](HANDOVER.md) 里。

---

## 效果展示

**输入**：原始比赛视频（960×544，21 fps，30 秒短样本）

![原始第一帧](docs/images/01_input_frame.jpg)

**输出**：叠加分析后的视频，左侧统计面板 + 右上 Mini Court 俯视轨迹

![完整叠加效果](docs/images/04_overlay_full.jpg)

**统计面板**（每个球员 4 项核心数据：当前速度 / 回合距离 / 回合最高 / 总距离）

![面板放大](docs/images/05_panel_close.jpg)

**Mini Court**（俯视图轨迹：黄色 = 上半场球员，粉色 = 下半场球员，青色 = 球）

![Mini Court 放大](docs/images/06_minicourt_close.jpg)

---

## 整体架构

```
原始视频.mp4
    │
    ▼  Step 1: TrackNet ─────────── 球检测（专门追小目标的连续帧热力图模型）
带球轨迹的视频 + 球坐标 CSV
    │
    ▼  Step 2: Overlay ──────────── YOLOv8s-pose + ByteTrack + Homography
叠加分析的视频
    │
    ▼  Step 3: FX（可选）────────── 子弹时间冻帧 + 慢动作 + 虚拟轨道相机
普通速度分析视频（默认）
```

三段独立运行，每段可以单独迭代。改一次面板字号、调一次跳变阈值、加一个新特效，都不需要从头重跑。

---

## 快速开始（30 秒短视频跑通）

### 1. 克隆仓库（含 LFS 大文件）

```bash
git lfs install     # 没装的话先 brew install git-lfs
git clone https://github.com/ychenfen/badminton-pipeline-repro.git
cd badminton-pipeline-repro
```

模型权重优先使用 `weights/TrackNet_official_best.pt`（官方 TrackNetV3，130 MB）；
旧的 `weights/TrackNet_best.pt` 保留作兼容回退。大文件通过 Git LFS 下载。

### 2. 装依赖

```bash
python3 -m pip install --user --index-url https://pypi.org/simple \
    numpy opencv-python pandas Pillow torch ultralytics tqdm \
    pycocotools parse lap
```

`pycocotools`、`parse`、`lap` 是 TrackNet/ByteTrack 的隐藏依赖，原 requirements 没列全，必装。

### 3. 标球场 4 角点

整个流程**唯一需要人工**的环节。运行：

```bash
python3 scripts/tools/select_court.py short.mp4
```

弹窗里**按顺序**点 4 下：左上 → 右上 → 右下 → 左下（球场长方形 4 角，不是球网）。点完按 q，终端会输出像 `--court_points "352,342,628,343,944,527,52,532"` 这样的字符串。

样本视频 `short.mp4` 的标准答案：

```
352,342,628,343,944,527,52,532
```

效果如图（黄色四边形贴合球场边线）：

![球场角点标注](docs/images/02_court_corners.jpg)

### 4. 一键跑通分析流程（FX 默认关闭）

```bash
./run_all_mac.sh \
  --input-video short.mp4 \
  --court-points "352,342,628,343,944,527,52,532" \
  --yolo-device mps
```

默认只执行 TrackNet 与 Overlay，最终视频保持普通速度。命令行可显式加入子弹时间/慢动作；Web UI 打开“子弹镜头特效”按钮即可对专业分析启用相同效果：

```bash
./run_all_mac.sh ... --enable-bullet-time-fx
# 或：BADMINTON_ENABLE_BULLET_TIME_FX=1 ./run_all_mac.sh ...
```

对于这段 1080p 羽毛球转播，脚本默认使用标准 TrackNet 解码和两个互补的非重叠时序相位。这样比在检测到 CUDA 后自动切换到高清分块更可靠：高清分块在高吊球时容易将球员、横幅或观众误认为羽球。`--tracknet-high-res` 仍可用于显式 A/B 测试，`--tracknet-tile-overlap 0.25` 用于调节其实验性分块重叠比例。

默认的 `nonoverlap` 模式会再跑一个错开半个序列长度的 TrackNet 相位（官方权重为 `offset=4`），输出到 `tracknet_official_result_phase_half/`。Overlay 以两相位的逐帧一致性、连续冲突片段和稀疏姿态框做候选级融合；短的人体/背景分支会在 Kalman 前被屏蔽，音频只验证已有视觉转折。可用 `--no-tracknet-dual-phase` 关闭第二相位。双相位不能恢复真正飞出画面上沿的球，这类帧会保持 `missing`，避免把背景或人体轨迹伪装成检测。

Overlay 阶段会在干净的输入画面上依次执行短缺口插值、常速度 Kalman 平滑、球场几何/静态背景过滤和镜头切换重置，并输出击球候选、自动回合编号、主画面发光拖尾和 Mini Court 拖尾。当几何过滤后的 TrackNet 真实覆盖率低于 35% 时，流程会评估 CPU 三帧运动差分检测；只有它至少形成 2 条严格轨迹、30 个观测并且实测点多于 TrackNet 时才会替换 TrackNet。经典检测先保留 120px 保守区域作为精度基线，再用 480px 顶部扩展执行高吊球专用的二次扫描；只有到达高空且满足更严格外观、加速度和弹道门槛的轨迹才会补入。轨迹断线只会切断可视拖尾，Kalman 仅在镜头切换时硬重置，因此短断检仍可插值，而不会把一次飞行误当成多个回合。可用 `--classical-trigger-coverage` 调整评估阈值，或用 `--no-classical-ball-fallback` 关闭经典检测。

渲染视频旁会生成 `<视频名>_ball_tracking.csv`，其中保留每帧的原始来源和置信度、`Source`（`model`/`classical`/`interp`/`kalman`/`missing`）、经典轨迹 ID、检测模式、击球坐标、球在击球帧是否实测、击球来源和自动回合编号。击球检测会先用双向二次曲线拼接同一飞行，再以分段二次拟合排除普通抛物线顶点；宽带音频只佐证已有的轨迹转折或端点，不再独立制造击球点。稀疏人体姿态仅用于标注击球方和人体框硬否决，姿态腕点不会成为球坐标；落入人体框、缺少邻近实测球轨迹或与轨迹位置不一致的事件会直接删除。自动回合以镜头切换为硬边界，默认允许最长 5 秒的球断检，因此高吊球/遮挡不会把同一回合拆开。

参数含义：
- `TRACKNET_VIS_THRESH=0.20` — TrackNet 二值化阈值；0.20 是官方权重在这段转播上的稳妥默认值。降低到 0.15 只建议做 A/B 实验，因为弱响应更容易跳到人体或背景（详见下文 §6）
- `--yolo-device mps` — YOLOv8 走 M 系列芯片 GPU 加速
- `--classical-trigger-coverage 0.35` — TrackNet 真实覆盖率低于此值时评估经典检测
- `--no-classical-ball-fallback` — 禁用低覆盖率自动经典视觉回退（默认启用）

跑完输出在 `~/yumaoqiu_repro/`：
- `tracknet_official_result/short_ball.csv` — 球的逐帧坐标
- `tracknet_official_result_phase_half/short_ball.csv` — 错开时序窗口的第二相位坐标
- `end1_ball_tracking_official_fused.mp4` — 普通速度叠加分析视频（默认播放产物）
- `end1_ball_tracking_official_fused_ball_tracking.csv` — 可审计的轨迹、击球和回合 sidecar
- `end1_ball_tracking_official_fused_fx.mp4` — 历史兼容文件名；FX 关闭时是普通速度视频，显式开启后才包含子弹时间特效

播放器兼容性优先使用同目录的 `_h264.mp4` 文件（例如 `end1_ball_tracking_official_fused_h264.mp4`）。

### 5. 看 demo

仓库已附带跑好的 demo：

```bash
open demo/short_overlay_demo.mp4
```

## Web UI（本地 ShuttleVision 工作台）

仓库现在附带一个轻量 Flask Web UI，适合在本机选择视频、标定球场并提交后台分析任务。它不需要 Node.js 或前端构建工具，页面由 `web_app/static/` 提供，推理仍复用本仓库的 TrackNet/Overlay 脚本。

### 启动

在已安装依赖的 Python 环境中运行：

```bash
python3 -m pip install -r requirements_repro.txt
./run_web_ui.sh
```

然后打开 <http://127.0.0.1:7860>。启动脚本会自动使用仓库内已准备的 BST 代码和 checkpoint（如果两者都存在）；如果机器上的默认 `python3` 不是项目环境，可以显式指定：

```bash
BADMINTON_UI_PYTHON=/data/miniconda3/envs/py311/bin/python ./run_web_ui.sh
```

也可以直接调用模块并修改监听端口：

```bash
python -m web_app.server --host 127.0.0.1 --port 7860 \
  --python "$(command -v python3)"
```

局域网内访问时，把监听地址显式改为所有网卡，并在服务器防火墙放行该端口：

```bash
./run_web_ui.sh --host 0.0.0.0 --port 7860
# 其他设备访问：http://<运行服务的电脑局域网 IP>:7860
```

默认仍监听 `127.0.0.1`，避免未经授权暴露上传接口；不要把服务直接暴露到公网。

上传和任务结果默认保存在 `runtime/web_ui/`（该目录适合加入 `.gitignore`）。可用环境变量调整本地部署：

| 变量 | 用途 | 默认值 |
|---|---|---|
| `BADMINTON_UI_DATA_ROOT` | 上传、任务和产物根目录 | `runtime/web_ui` |
| `BADMINTON_UI_INPUT_ROOTS` | 允许“导入服务器路径”的目录，多个目录用系统路径分隔符 | 仓库根目录 |
| `BADMINTON_UI_MAX_UPLOAD_BYTES` | 单次上传上限 | 20 GiB |
| `BADMINTON_UI_PYTHON` | `run_web_ui.sh` 使用的 Python | `python3`（缺少依赖时尝试项目 conda 环境） |

### 使用流程和可选功能

1. 点击“选择视频”上传 MP4/MOV/MKV/AVI/WEBM/M4V；服务端会先验证视频可解码并生成封面。
2. 在第一帧画布上按 **TL → TR → BR → BL** 点击球场四角。坐标会按原视频分辨率提交，不能点球网或观众区。
3. 可分别勾选“自动回合切割”、“鹰眼辅助复核”、“专业比赛分析”和“子弹镜头特效”。
   - 四项都关闭：只导入和预览，不启动模型。
   - 只开回合切割：运行基础 TrackNet 球路分析，生成回合短片。
   - 只开鹰眼辅助复核：运行基础球路分析，并只审查每个回合**最后一拍之后**的终局轨迹。系统先检测“落地前运动 → 速度骤降 → 至少 3 帧稳定”的真实事件；若落地后球被遮挡但原视频出现持续静止的小白球，会用多段背景差分恢复该目标，并给出 `OUT`/复核和边线距离范围。离屏、插值、跨缺口分支、球员身体锁定和后续击球不会直接生成落点。
   - 开专业分析：运行双相位 TrackNet、YOLOv8-pose、BST 击球类型识别、击球/球员移动/热力图 Overlay。Motion Coach 分析准备、触球、可见动力链和恢复阶段；同类问题至少在 3 个独立高质量击球中重复时，才生成简洁的“问题 / 训练”建议。点击时间码可回看证据；没有可靠建议时页面显示 0 条，不用低置信内容填充。
   - 开子弹镜头特效：自动带上专业分析，并在输出视频中加入冻结画面与慢动作；渲染时间会增加。
   - 可自由组合：各功能复用同一个 sidecar；同时打开回合切割时会额外导出短片。
   - 若长视频由同一固定全景机位组成，可额外打开“跨硬切镜头继续分析”；只有所有片段仍适用同一组球场四点时才建议使用。
4. 任务在后台单 GPU 队列执行，页面会显示阶段、日志、摘要和可下载产物；浏览器优先播放 `_h264.mp4`。

### 当前限制

- 自动回合是“智能初筛”：`RallyID` 由有效球轨迹、已验证击球、约 5 秒断检和镜头切换聚类得到，不是严格的发球→死球语义分类器。短片带有默认前后缓冲，仍建议在时间轴上复核起止点。
- 一键脚本和 Web UI 默认在第一个硬镜头切换后停止固定球场标定。多机位、回放、特写或换场视频需要先切成固定全景片段，或为每个场景单独标定；同一标定下可谨慎开启“跨硬切镜头继续分析”。
- 专业移动轨迹和热力图目前主要烘焙在分析视频中；sidecar CSV 提供逐帧球轨迹、击球、回合编号和球速摘要，并不是完整的球员坐标/热力图 JSON。
- Motion Coach 不展示逐拍候选。单拍须满足击球确认、击球方归属、真实球点和二维姿态门槛；仅靠回合交替推断的击球方、单帧肢体异常或重复击球记录都不能投票。相同问题按准备衔接、前场球、上手球、后场球和击球恢复等兼容动作组聚合，仍须至少 3 个独立可靠击球。低置信、低覆盖或区分度不足的 BST 结果视为缺失证据，不会否决独立可靠的球路规则；明确的类型冲突、侧别冲突和高置信但无法安全映射的类型仍会拦截。BST 只有在 Top-1 ≥85%、Top-1/Top-2 差值 ≥25 个百分点、姿态覆盖 ≥85%、球点覆盖 ≥55% 且与球路规则一致时，才成为精确球种建议的类型来源。单机位不能可靠判定握拍、拍面、旋转、真实球高、绝对球速、伤病或绝对动作分，因此证据不足时直接不输出建议。
- “鹰眼辅助复核”是单机位二维落点复核，不是正式比赛 Hawk-Eye。它只把 `model`/`classical` 的真实终局观测作为轨迹证据，不使用插值、Kalman 预测、离屏桥接或 `terminal_endpoint_proxy`。当 TrackNet 在遮挡后漏掉静止球时，会用原视频的多段背景差分寻找持续、紧凑的新白色目标，并排除场线、广告和人体残差；恢复结果只发布边界、`OUT`/复核和距离范围，不输出厘米级伪精度。没有落地证据时明确显示“未观察到可靠落地事件”；距线 20 cm 内、超出边线 1 m、标定/镜头不稳或启用了跨硬切继续时会降级为视频复核。参考的开源方案和许可证边界见 [docs/hawkeye_open_source_review.md](docs/hawkeye_open_source_review.md)。请以原视频和裁判/教练复看为准，不能用于正式判罚。
- TrackNet 和 Overlay 对长视频很耗时，服务端默认串行处理以避免 GPU/内存争用。音频击球检测会解码整段音频，小时级视频可能占用大量内存；资源不足时应缩短视频或关闭音频辅助。
- 浏览器上传的是文件内容，不会把用户电脑上的任意绝对路径暴露给服务器。“导入服务器路径”仅允许配置目录内的普通视频文件；默认服务只监听 `127.0.0.1`。
- 上传视频最好先规范化为恒定帧率（CFR）H.264 MP4。VFR、旋转元数据或不兼容的编码可能造成帧坐标和切片时间轻微漂移。

---

## 三段详解

### Step 1 — TrackNet（球检测）

**它解决的问题**：羽毛球只有几个像素、飞得快、容易模糊，单帧 YOLO 之类的检测器经常漏。

**它的思路**：官方权重一次处理 8 帧连拍，输出 8 张概率热力图（每个像素值 = 这里是球的概率）。利用连续帧的运动信息识别出模糊的球。类比：你看一张静态照片可能看不出蚊子在哪，但连续帧能看出“有什么东西在那一带飞过”。

**输出**：

![TrackNet 输出帧](docs/images/03_tracknet_output.jpg)

视频上小圆圈是模型识别出的球轨迹。同时生成 CSV：

```csv
Frame,Visibility,X,Y
0,1,455,202
1,1,455,202
4,1,481,122
...
```

`Visibility=1` 表示这一帧检测到球，X/Y 是球在画面里的像素坐标。

**关键参数**：

| 参数 | 含义 | 推荐 |
|---|---|---|
| `--tracknet_file` | 模型权重 | `weights/TrackNet_official_best.pt` |
| `--device` | 推理设备 | `auto`（Mac CPU；NVIDIA cuda） |
| `--large_video` | 流式 dataloader | 长视频必须加 |
| `--eval_mode` | `nonoverlap` / `weight` | `nonoverlap` 快 8 倍 |
| `TRACKNET_VIS_THRESH`（环境变量） | 二值化阈值 | **0.20**（0.15 仅 A/B） |

### Step 2 — Overlay（球员检测 + 数据叠加）

整个项目 90% 的工程量在这一段，干 5 件事：

1. **YOLOv8s-pose** 检测每帧球员 + 17 个人体关键点（脚踝精确定位"足点"）
2. **ByteTrack** 维持球员 ID 跨帧不串
3. **Homography** 透视矫正：把斜拍画面里的梯形球场拉成俯视长方形
4. **MotionStats** 算每个球员的瞬时速度 / 累计距离 / 最高速度
5. **绘制叠加层**：左侧面板 + 右上 Mini Court + 球员骨架 + 轨迹线

**透视矫正示意**：

```
画面里的梯形                标准球场坐标系（俯视）
                              (0, 0) ─────── (6.1, 0)
   TL ──── TR                    │              │
    \      /                     │              │
     \    /     ──→ Homography ─→│              │
      \  /                       │              │
   BL ──── BR                    │              │
                              (0, 13.4)─── (6.1, 13.4)
```

OpenCV 一行调用：

```python
H, _ = cv2.findHomography(court_quad, dst_rectangle)
```

之后任何脚点像素坐标都能投影到球场米制坐标，距离/速度计算跟机位无关。

**为什么要球员脚踝不用 bbox 中心**：bbox 中心是身体中心，离地有 1 米多高；脚踝在地面上，投影更准。`estimate_foot_point()` 优先取左右脚踝平均，置信度低时退到 bbox 底中点。

### Step 3 — FX（可选的子弹时间特效）

该阶段默认跳过，以免长视频被冻帧/慢动作改变时间轴。显式使用 `--enable-bullet-time-fx`，或在 Web UI 打开“子弹镜头特效”按钮后，才会生成电影《黑客帝国》Neo 躲子弹那样的单机位轻量版效果：

- 选定一些时间点（"子弹时刻"，可手动 / 均匀分布 / 自动峰值检测）
- 在该时刻**冻帧** 28 帧，期间用虚拟相机做小幅度旋转 + 缩放
- 冻帧后接 **40 帧慢动作**（每帧重复 6 次，可选插值）
- 然后回到正常播放

完全不依赖检测结果，纯粹是对 Step 2 的输出做后期。

---

## 参数速查表

### 跑别的视频要改什么

1. `--input-video` 路径
2. 重新跑 `select_court.py` 标 `--court-points`
3. 如果机位完全不同（前场低位 / 侧场），可能要调 `--court_length_m`（全场 13.4，半场 6.7）

### 全长视频时间预估（M4 Pro）

| 阶段 | CPU | MPS（GPU） |
|---|---|---|
| TrackNet（13344 帧） | ~3 小时 | 暂不支持，需改代码 |
| Overlay | ~30 分钟 | ~10 分钟 |
| FX | ~5 分钟 | 同上 |

**最大瓶颈是 TrackNet**，用 PyTorch MPS 后端能压到 30 分钟。改造方案见 [HANDOVER.md](HANDOVER.md) §10 Task P1.1。

---

## 常见问题

**Q：球完全检测不到（Visibility 全 0）**
A：先使用脚本默认的 `TRACKNET_VIS_THRESH=0.20`，确认权重和输入视频正确；仍然全 0 时再显式 A/B 测试 0.15。详见 [HANDOVER.md](HANDOVER.md) §6.1.4。

**Q：球员速度显示 24 m/s（比博尔特还快）**
A：跳变阈值过松导致 ID 串变被记进 max_speed。已修复为 `8.0 × dt + 0.05` 自适应阈值。详见 [HANDOVER.md](HANDOVER.md) §6.2.7。

**Q：中文显示成方块**
A：字体回退已加了 macOS PingFang.ttc。如果还报错，确认 `/System/Library/Fonts/PingFang.ttc` 存在。

**Q：球场轮廓画歪了**
A：`select_court.py` 点的顺序错了，必须 TL → TR → BR → BL。可加 `--draw_court_polygon` 让 overlay 视频里画出绿色四边形检查。

**Q：`ModuleNotFoundError: pycocotools / parse / lap`**
A：在项目 Python 环境中重新执行 `python -m pip install -r requirements_repro.txt`；这些 TrackNet/ByteTrack 依赖已列入当前 requirements。若使用极简系统镜像，再单独确认编译工具链已安装。

更多问题排查：[HANDOVER.md](HANDOVER.md) §8。

---

## 项目结构

```
badminton-pipeline-repro/
├── README.md                       # 本文件（中文快速上手）
├── HANDOVER.md                     # 1500+ 行详细交接文档（含 AI agent 任务包）
├── README_MAC.md                   # macOS 启动笔记（原作者）
├── CHAIN_EVIDENCE.md               # 原作者解释为什么这条 pipeline 是"最可信"链
├── run_all_mac.sh                  # macOS 一键脚本
├── run_all.ps1                     # Windows PowerShell 一键脚本
├── requirements_repro.txt          # Python 依赖
├── short.mp4                       # 30 秒样本视频（LFS）
├── b13b2c0b...mp4                  # 全长 10 分 35 秒比赛视频（LFS）
│
├── weights/                        # 模型权重（LFS）
│   ├── TrackNet_official_best.pt   # 官方 TrackNetV3（8 帧/30 epochs，默认）
│   ├── TrackNet_best.pt            # 旧权重（4 帧/3 epochs，回退）
│   └── yolov8s-pose.pt             # 球员姿态（23 MB）
│
├── demo/
│   └── short_overlay_demo.mp4      # 跑好的成品 demo（LFS）
│
├── docs/images/                    # README 配图
│
├── web_app/                        # 本地 Flask Web UI 与后台任务队列
│   ├── server.py                   # HTTP API、上传、SSE 状态和文件服务
│   ├── pipeline_service.py         # 单 GPU 队列与 TrackNet/Overlay 调用
│   ├── rally_cutter.py             # RallyID 解析与 ffmpeg 切片
│   └── static/                     # Web 页面资源
├── run_web_ui.sh                   # Web UI 启动脚本
│
└── scripts/
    ├── tracknet_runtime/           # Step 1: TrackNet
    │   ├── predict.py              # 入口
    │   ├── model.py                # 网络结构
    │   ├── dataset.py              # 数据加载
    │   ├── test.py                 # 工具函数
    │   └── utils/general.py        # HEIGHT=288, WIDTH=512 等常量
    │
    ├── overlay/
    │   └── overlay_player_analytics.py   # Step 2: 1340+ 行核心
    │
    ├── fx/
    │   └── video_fx_bullet_time.py       # Step 3: 特效
    │
    └── tools/                      # 调试工具
        ├── select_court.py         # 交互式标球场角点
        ├── diag_tracknet.py        # 诊断 TrackNet heatmap 强度
        └── render_panel_preview.py # 单独渲染面板预览
```

### 可选：BST 逐拍击球类型识别

专业分析已接入 [BST: Badminton Stroke-type Transformer](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer) 适配层。配置上游研究仓库和 checkpoint 后，Web UI 会在独立面板显示每个击球的 Top-3 类型、置信度和姿态/球点覆盖率。Top-1 达到 60% 只是类型报告门槛之一；Motion Coach 使用 85% Top-1、25 个百分点区分度和更高覆盖率门槛，且只让与球路规则一致的 BST 成为精确类型来源。低置信或覆盖不足的 BST 仍用于诊断，但会回退到独立可靠的球路规则；明确冲突、侧别错误和高置信无法映射结果不会回退。未配置 BST 时专业任务仍正常完成。BST 只识别“是什么球”，不提供“动作好坏”评分。

配置示例和类别/权重匹配要求见 [docs/bst_integration.md](docs/bst_integration.md)。

---

## 这个项目踩过的坑（精简版）

| 问题 | 现象 | 修复位置 |
|---|---|---|
| TrackNet 球检测 0% | csv 全 0 | `predict.py:35` 阈值 0.5 → 环境变量 |
| 球员速度 24 m/s | 面板数字离谱 | `overlay_player_analytics.py:602` 阈值 1.2m → `8×dt+0.05` |
| 中文方块 | macOS 字体路径 | `overlay_player_analytics.py:18-27` 字体回退表 |
| 面板信息冗余 | 每球员 7 个数字 | 砍到 4 个核心 |
| 球场 quad 错位 | 距离速度全错 | `select_court.py` 重新人工标 |

完整变更清单：[HANDOVER.md](HANDOVER.md) §9。

---

## 后续工作

[HANDOVER.md](HANDOVER.md) §10 列了 12 个详细任务（P0-P3），每个都按"背景 / 目标 / 步骤 / 涉及文件 / 验证 / 已知陷阱"格式写好，AI agent（Codex / Cursor / Claude）可以直接挑一个任务从那里起步。

最优执行顺序：

```
P0.1 写默认值进 sh        → 15 min
P0.2 清理临时文件          → 已完成
P0.3 全长视频跑 baseline   → 3-4 小时
P1.1 TrackNet MPS 加速     → 2-4 小时
P1.2 缓存 detections.json  → 3-5 小时
P1.3 YOLO 跳帧检测         → 1-2 小时
P2.1 球轨迹补漏（卡尔曼）  → 3-4 小时
P2.2 击球点 + 自动回合分割 → 4-6 小时
P2.3 Mini Court 视觉增强   → 1-2 小时
P3.1 Flask Web UI           → 已提供本地 MVP
P3.2 多机位适配            → 1-2 天
P3.3 数据导出 + 热力图     → 4-6 小时
```

---

## 致谢

- TrackNet 模型来自 [TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- YOLOv8 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)
- ByteTrack 跟踪算法 [ByteTrack](https://github.com/ifzhang/ByteTrack)
- 样本视频出自 YouTube 频道 POGBADMINTON

---

## License

代码部分 MIT。模型权重和样本视频按各自原始来源的 license 使用，仅供学习研究。

---

## 引用

如果这个项目帮到了你的研究、论文、或产品，star 是最简单的支持方式。论文引用：

```bibtex
@misc{badminton_pipeline_repro,
  author       = {ychenfen},
  title        = {Badminton Match Video Analytics Pipeline},
  year         = {2026},
  howpublished = {\url{https://github.com/ychenfen/badminton-pipeline-repro}}
}
```

如果你做了改进或衍生项目，欢迎提 PR / Issue / Discussion。

---

## 关键词

羽毛球, 视频分析, 图像识别, 运动分析, 计算机视觉, 目标检测, 多目标跟踪, 球员追踪, 球轨迹, 透视变换, 单应性矩阵, 子弹时间, 慢动作, 体育数据分析, AI 教练, 羽毛球训练, 比赛复盘, 战术分析, OpenCV, PyTorch, YOLOv8, TrackNet, ByteTrack, Apple Silicon, M4 Pro, MPS, macOS, badminton, sports analytics, video analytics, computer vision, object tracking, shuttle detection, player tracking, court homography, bullet time, TrackNet, YOLOv8, ByteTrack, OpenCV, PyTorch, Apple Silicon, macOS.
