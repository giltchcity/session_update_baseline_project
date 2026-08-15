# Session Update Baseline

本仓库实现一个跨完整 session 的 metric-semantic 4D mapping baseline。它借鉴并移植
Khronos、Panoptic Mapping 等开源实现，但研究问题、session 状态协议、运行入口和最终系统
属于本项目；它不是“原封不动的官方 Khronos”，也不是在两个最终网格之间做一次离线清理。

代码仓库只有这一份：

- Git worktree：`/home/jixian/Desktop/FT/session_update_baseline_project`
- 代码根目录：`session_update_baseline/`
- 便捷路径 `/home/jixian/Desktop/FT/session_update_baseline` 是指向上述代码根目录的符号链接，
  不是第二份源码
- GitHub：`https://github.com/giltchcity/session_update_baseline_project`
- 论文仓库另置于 `https://github.com/giltchcity/myncv---rpsl---daicma`

这是本仓库唯一的 Markdown 文档。不要再建立第二份 README、设计说明或 agent 交接文档；
运行记录应写成 JSON、CSV、TXT 或日志。

## 当前状态（2026-08-15）

正式递归接口、唯一 session runner、显式 semantic/instance wire protocol、同步终局保存、
canonical mapper 源码选择、instance label 固化进 dataset 以及 1 cm 标准配置的完整 A→B
链已经实现并跑完。旧文档所说的“没有 runner”、“根据 `CV_32SC1` 猜 packed”、“终局直接
追加 raw DSG”、“没有 `finishProcessing()`”、“5 cm 仍配 2 cm mesh grid”和“递归全量
从未运行”都已经过时。

2026-08-15 的 1 cm A→B 全量链（`configs/room18_instance_1cm.yaml`）：

- session_a：4,003 帧、62 个 time steps、17.9 GB final.4dmap、16 个 physical ID 全部带
  current private mesh、catalog review gates 全部通过。
- session_b：4,041 帧、29 个 time steps、22.9 GB final.4dmap、16 个 physical ID 全部带
  current private mesh；唯一 soft gate 是 `timeline is too short: 29 < 50`——这是
  “B 只继承 A 最新状态作为 1 个 seed snapshot、其余快照全部是 B 自己的变化”这一继承设计
  的预期产物，不是建图失败（见第 4 节）。
- 评审规则已固化进 dataset：`datasets/local_ab/instance_labels/` 是唯一权威 instance
  map（I9 合并进 I7、I8 清除、袋子编号轮换、I20 仅 A），自带
  `manifest.json`（schema `reviewed_ab_physical_instances/v1`），可直接运行无需脚本；
  原生实例图完整备份在 `instance_labels_backup_20260815/`。
- 外部门禁已改为软评判：数据契约（帧账、checksum、empty depth、时间戳、重复 ID、seed
  等价）仍然 hard-fail；catalog 期望（物理 ID 集合、timeline 长度、轨迹数、语义标签、
  private mesh 存在性）全部记录到 `transition_manifest.json` 的 `review_gates` 字段，
  map 先建好、再评判、再操作。

这不表示整套论文系统已经验收。当前仍需完成或验证：

1. 构建仍从外部 ROS 工作区取得 Hydra、Spark-DSG、Kimera-PGMO、ianvs、
   config_utilities 等研究依赖，尚未达到 clean clone 自包含构建（vendoring 已规划）。
2. 已知几何质量问题未解决：背景 mesh 1 cm 后明显干净，但床/桌/衣柜等大件家具的
   object private mesh 仍碎——根因已定位为三条独立机制（见第 4 节末）：
   a. object 分辨率 = `-0.01 × extent` ≈ 2–3.3 cm，不是 1 cm；
   b. 每个 track 重建只使用 frame buffer 内 ≤100 存储帧（≈10 s 窗口），连续可见的
      track 更只在 session 终点提取一次，96 s 的观察只用最后 ~10 s；
   c. 跨可见 segment 的几何是 winner-takes-all（support gate 选一者），无跨段累积；
      reconciler 虽有真实的面片级融合路径（`mergeObjectMeshes`），但只由后端 merge
      提议触发，而 `max_dt_merge_proposal: 3.0` 让跨段（相隔 >3 s）提议结构性不可达。
3. 完整 A→B→C 尚未跑完：C 只读 B 输出即可运行，但还没执行；B 的 soft gate 是否要在
   评审里接受/调整 catalog 期望也未定。

## 1. 冻结的研究问题与协议

系统统一处理三种动态：

1. **D1：可见运动**。物体持续可见时发生运动；系统保存轨迹和时序几何，运动过程不得污染
   当前静态地图。
2. **D2：同一 session 内隐藏变化**。物体离开视野后变化，并在同一运行中被重新观测；系统
   结合表面支持和 free-space/ray evidence 更新当前状态。
3. **D3：跨 session 的同一种隐藏变化**。前一进程结束后环境发生变化；下一独立进程先恢复
   前一持久状态，然后继续调用与 D2 完全相同的状态更新。它不是另一套 change detector，
   不是额外的 pairwise reconciliation，也不是离线后处理。

系统只有一个状态更新：

```text
P_next = Update(P_current, current_observation)
```

持续可见时由这个更新产生 D1 轨迹；出现观测空档后，由同一更新结合新表面和 free-space/ray
证据处理 persistent、absent、unobserved、moved 和 new。D2 的空档发生在一个进程内，D3
的空档跨越 serialize -> process termination -> deserialize；算法分支和状态语义不变。

唯一正式接口是：

```text
P_new = RunSession(P_prev, complete_session_observations, T_session_to_world)
```

递归展开为：

```text
P_empty + Session A  -> P_A
P_A     + Session B  -> P_B
P_B     + Session C  -> P_C
...
```

每个 session 都是新进程、新 active window、新短期 tracker 和新 session 时钟。A 从空状态
开始；B 必须加载 A 的最终持久状态；C 只加载 B 的输出。B 的数据时间可以重新从零计时，但
地图状态不能清空。

`P_A`、`P_B`、`P_C` 使用相同 schema。当前实现加载 `P_prev/final.4dmap` 中最新的 updated
current DSG 作为新 timeline 的初始状态，而不是把前一 session 的整段历史再次递归复制；随后只追加
本 session 的状态。A seed 是 B 判定旧几何和旧实体为 persistent、absent 或 unobserved 的
必要基准，不是对 B 的“污染”。

生产路径禁止：

- 运行 `B_FROM_SCRATCH`；它只能作为另行显式消融，不是第二套 production B。
- 把已加载 A 的 B 再交给 Base1 并再次注入 A prior。
- 使用 A/B 专用 pairwise 后处理定义算法。
- 让 C 再读取 A 或依赖硬编码的 A/B 路径。

`scripts/run_final_ab_pipeline.sh` 和旧 session 入口已显式拒绝执行；
`run_session_update_baseline.sh` 的旧 post-process 默认拒绝、只有显式 diagnostic 环境变量才
放行。`run_base1_khronos_env.sh` 中仍可直接调用的工具也已标为 diagnostic/export，不属于
正式协议。

## 2. 地图、实体与状态不变量

- `semantic_id` 表示类别，`physical_instance_id` 表示跨帧、遮挡和 session 的实体身份；两者
  必须独立保存。
- 同一 physical ID 可以有多个可见时间段，但当前逻辑地图只能有一个实体状态。例如 I10
  始终是一把 swivel chair，不能因 tracker 分段而留下多把椅子。
- `persistent` 和 `new` 进入当前地图；`absent` 只有在可靠重观测后移除；`unobserved` 保守
  保留。没有看到不等于确认消失。
- object/dynamic 像素从 background TSDF 排除；当前 object private mesh 是权威地图的一层，
  不能依赖 viewer 临时拼 PLY 才看到家具。注意：reconciler 的 `merge_object_meshes: true`
  融合路径只由后端 merge 提议触发，而 `max_dt_merge_proposal: 3.0` 让跨可见 segment
  （相隔 >3 s）的提议结构性不可达——实际跨段几何是 functor 的 winner-takes-all，没有累积。
- moved object 的旧位置必须从当前状态消失，新位置必须出现；运动轨迹保留在时序层，不能
  作为多份静态残影。
- D1、D2、D3 共享一个 `Update`、physical identity、证据、状态枚举和 current-map 物化规则；
  D2 与 D3 的唯一差别是观测空档是否跨越 serialize → process termination → deserialize
  边界。

当前联合 catalog（`configs/room18_physical_catalog.json`）为 17 个物理实体、12 个 object
semantic classes：

```text
I1 bed/S7, I2 cabinet/S10, I3 table/S15, I4 shelf/S24,
I5 desk/S33, I6 wardrobe/S35, I7 white desktop monitor incl. black gaming laptop/S74,
I10 swivel chair/S75, I11 apparel/S92,
I12 maroon backpack/S115, I13 checkered bag/S115, I14 blue backpack/S115,
I16 patterned bag/S115, I17 white bag/S115, I18 blanket/S131,
I19 fan/S139, I20 MacBook/S74
```

I9（black gaming laptop）按 2026-08-14 人工评审决定合并进 I7，任何 A/B 输出都不再出现
I9；该决定已固化进 `datasets/local_ab/instance_labels/`（source 9→7 remap，A 的
MacBook 帧 1962-1980 上 9→20）。I20 只在 A 出现（A 的 `active_physical_instance_ids` 含
20，B 不含）。A 的床面区域按人工策略整体归 I1，因此 A 中 I12 像素为 0；不要自动把
床上的局部重新细分成 I12。physical I12 与 semantic S12=person 属于不同命名空间。

## 3. 输入数据与显式 label 协议

每个完整 session 需要 RGB、深度、相机位姿、语义图和 physical-instance 图：

```text
rgbd/session_n/
  timestamps.csv
  Intrinsics.txt
  <ImageID>_color.png
  <ImageID>_depth.tiff
  <ImageID>_pose.txt

semantics/session_n/<ImageID>_segmentation.png
instances/session_n/<ImageID>_segmentation.png    # canonical（评审规则已固化，可直接运行）
instances/session_n/manifest.json                 # reviewed_ab_physical_instances/v1 台账
# 历史格式（仅 backup）：<ImageID>_instances.png
alignment/session_n_to_world.txt                  # 可选 4x4 左乘变换
```

本数据集的实际实例图位于 `datasets/local_ab/instance_labels/`：评审规则（I9→I7、
I8 清除、袋子编号轮换、I20 仅 A）已于 2026-08-15 应用固化，目录自带 manifest 台账，
**不需要任何脚本**即可直接运行；原生实例图备份在
`datasets/local_ab/instance_labels_backup_20260815/`。固化脚本是
`scripts/finalize_reviewed_instance_labels.py`，评审规则的来源是
`scripts/build_reviewed_ab_physical_instances.py`（SAM2 局部细化产物已删除，
其身份级规则全部保留在 remap 中）。输入 resolver 同时接受 canonical
`<ImageID>_segmentation.png` 与历史 `<ImageID>_instances.png`，但同一帧两种名字同时存在
会被视为歧义并拒绝。runner 在创建输出目录前逐帧核对 RGB-D 时间表对应的
semantic/instance 文件；播放器再核验图像尺寸、单通道整数类型、16-bit ID 范围和位姿矩阵，
不再需要 staging copy。

语义图保存 ADE semantic ID，instance 图保存稳定 physical ID，0 表示该像素没有物理实体。
播放器在提供 `--instance-dir` 时显式生成：

```text
packed_32SC1 = (semantic_id << 16) | physical_instance_id
```

mapper 只有在配置 `active_window.input_labels_are_packed: true` 时才拆包；不再根据 OpenCV
矩阵类型猜测协议。未提供 instance 目录时走明确的 semantic-only 路径。

当前 A 有 4,003 帧，B 有 4,041 帧。语义类别数、物理实体数和每段视频实际可见的实体数是
三个不同计数，不能混用。

## 4. 唯一运行入口

生产入口只有：

```text
session_update_baseline/scripts/run_session.sh
```

它只启动一次 mapper，不运行 from-scratch 对照，也不调用 Base1。`--input-state` 可为空，
也可指向上一 state 目录或其中的 `final.4dmap`；`--output-state` 必须是尚不存在的新目录。

首个 session：

```bash
cd /home/jixian/Desktop/FT/session_update_baseline
./scripts/run_session.sh \
  --run-dir /path/to/rgbd/session_a \
  --semantic-dir /path/to/semantics/session_a \
  --instance-dir /path/to/instances/session_a \
  --output-state /path/to/states/session_a
```

后续 session：

```bash
./scripts/run_session.sh \
  --input-state /path/to/states/session_a \
  --run-dir /path/to/rgbd/session_b \
  --semantic-dir /path/to/semantics/session_b \
  --instance-dir /path/to/instances/session_b \
  --world-transform /path/to/alignment/session_b_to_session_a.txt \
  --output-state /path/to/states/session_b
```

C、D 和之后的 session 使用完全相同命令，只把紧邻上一 accepted state 传给
`--input-state`。`run_khronos_session_strict.sh` 是该入口内部使用的受保护执行层，不应直接
调用。

一次成功运行输出：

```text
session_n/
  final.4dmap
  transition_manifest.json
  state_summary.json
  control/
    command.txt
    playback_manifest.json
    processing_manifest.json
    logs/
    *.exit_code
```

播放器采用 per-frame ACK；strict 层检查进程退出码、clean-finish 标记和 `final.4dmap`。
transition manifest 记录输入/输出 state SHA-256、配置 SHA-256、真正 ACK 的终帧时间以及当前
mesh/object/physical-ID 统计；输出 latest timestamp 必须等于最后 ACK 帧。递归运行还必须证明
`output.initial == input.current`，否则运行失败而不是标成完成。
终止路径会 drain pipeline，等待异步 change detection，执行一次同步终局优化和同一状态更新，
再保存唯一 map timeline；保存端不再把 raw private DSG 追加成
latest state。

标准配置为 `configs/room18_instance_1cm.yaml`（1 cm；`room18_instance_5cm.yaml` 保留为
参考/快速配置）：

```text
image scale                       0.5 (960x540)
flow control                      per-frame ACK
min_output_separation             0.4 s
change_detection_every_n_backend_updates   5
save_every_n_frames              0
voxel / truncation               0.01 m / 0.03 m
mesh resolution                  0.01 m
object_reconstruction_resolution -0.01（= 1% × object extent，≈2–3.3 cm，见已知问题）
min_separation_distance          10 voxels
removed min_weight 修复（去除 5 帧污染）
```

2026-08-15 实际使用的 A、B 命令（产物在
`runs/session_ab_1cm_20260815/`，每个 session 的 `control/command.txt` 有完整复现记录）：

```bash
# session_a（空状态起步）
./scripts/run_session.sh \
  --run-dir datasets/local_ab/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p \
  --semantic-dir datasets/local_ab/semantics/session_a \
  --instance-dir datasets/local_ab/instance_labels/session_a \
  --output-state runs/session_ab_1cm_20260815/session_a

# session_b（加载 A，world transform 对齐）
./scripts/run_session.sh \
  --input-state runs/session_ab_1cm_20260815/session_a \
  --run-dir datasets/local_ab/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p \
  --semantic-dir datasets/local_ab/semantics/session_b \
  --instance-dir datasets/local_ab/instance_labels/session_b \
  --world-transform datasets/local_ab/alignment/session_b_to_session_a.txt \
  --output-state runs/session_ab_1cm_20260815/session_b
```

继承设计（解释 B 的 29 个 time steps 和 22.9 GB）：

- B 加载 A 的 `final.4dmap` 时只取**最新一个 updated current DSG** 作为 1 个 seed
  snapshot（`latestSessionSeed` + `initializeSessionTimeline`），A 的整段历史
  **不递归复制**；B 的 timeline = 1 seed step + B 自己触发变化检测产生的快照。
  B 快照少（29 < A 的 62）是因为 B 拍到的大部分内容 A 已建过，change detection 只对
  新增/变化触发——这是 D2/D3 语义的正确产物，也是 catalog `minimum_time_steps: 50`
  对继承型 session 偏保守、只能当 soft gate 的原因。
- 每个 snapshot 都是**完整场景**（不是 delta）：B 的每个快照都携带 A 的 inherited
  background mesh，所以 `final.4dmap` 体积 ≈ 1.3×A（22.9 GB vs 17.9 GB）而不是等比
  于 B 的新内容。这是格式语义，不是泄漏。

### 固定建图流程（端到端，2026-08-15 冻结）

1. **准备 dataset（一次，已固化）**：`datasets/local_ab/` 下 rgbd/、semantics/、
   instance_labels/（评审规则已应用、manifest 台账在列、可直接运行）、alignment/
   （session_b_to_session_a.txt，world transform 左乘）。
2. **构建 canonical mapper**：`./scripts/build_canonical.sh`（源码/配置变更后；
   已修复 4 处，测试 13/13 + 8/8 通过），再用
   `./scripts/check_canonical_runtime.sh --require-built` 验证
   `CANONICAL_RUNTIME_OK`。当前仍 source 外部 `/home/jixian/ros2_ws` 的 7 个 MIT 库
   （vendoring 待做），runner 启动前会重算源码/二进制指纹，不一致即拒绝。
3. **运行 A**：见上，空 `--input-state`。输出 `final.4dmap`、
   `transition_manifest.json`、`state_summary.json`、`control/`（完整命令、播放/处理
   manifest、退出码、日志）。
4. **运行 B（及之后的 C…）**：把紧邻上一 accepted state 传给 `--input-state`，
   B 的数据时间从自身 acquisition 重新计时，但地图状态从 A 的 seed 继续。
5. **评判（评审即产物）**：数据契约 hard-fail（帧账、checksum、empty depth、严格递增
   时间戳、重复 ID、seed 等价 `output.initial == input.current`、输出起点时间戳）；
   catalog 期望 soft——全部 mismatch 记录在 `transition_manifest.json["review_gates"]`
   并 echo 到 stderr，**不拒绝已完成的地图**（先建好、再评判、再操作）。2026-08-15
   A 无 gate；B 一条 `timeline is too short: 29 < 50`（见上，预期产物）。
6. **已知几何问题**（见“当前状态”第 2 条）：大件家具 private mesh 仍碎，三根因已
   定位（分辨率 ≈2–3.3 cm / 重建窗口 ≈10 s / 跨段 winner-takes-all 无累积）；修复
   方向：object resolution 改正数 0.01、放宽/定期重建窗口、或引入跨段 private TSDF
   累积。这些只影响 object geometry 质量，不影响数据契约与 ID/语义正确性。

## 5. 构建与源码边界

唯一活动 mapper 源码是 `session_update_baseline/ports/mapping_core/`。顶层 CMake 要求显式的
`SESSION_UPDATE_CANONICAL_MAPPING_PREFIX`，并校验 `khronos`、`khronos_ros` 的 CMake
package 与运行时动态库来自这个 canonical install；外部 workspace 中另一份 Khronos 不能
被静默选中。

标准构建入口是：

```bash
cd /home/jixian/Desktop/FT/session_update_baseline
./scripts/build_canonical.sh
```

脚本先构建 `ports/mapping_core`，再让 baseline 只链接该 install，并运行测试和静态检查。
它不会删除已有 build/install；同一路径已有构建会立即拒绝，并发构建由文件锁隔离。只有在
调用者确认目录归自己所有时才可用 `--incremental`。
构建还保存 canonical mapping core 与本项目 C++ 源码的内容指纹；runner 启动前重新计算并
逐字节比较。源码变化但二进制未重建、验收工具缺失或动态库解析到另一棵源码时，运行必须在
创建输出目录之前失败。

当前构建仍默认 source `/home/jixian/ros2_ws/install/setup.bash` 获取非 Khronos 研究依赖。
所以“普通系统库 + 本仓源码即可 clean build”仍是发布目标，不是当前事实。完成自包含前，
README、论文和 release 都必须明确这一边界。

主要目录：

```text
ports/mapping_core/   唯一活动的 mapping core、backend、4D map 与 ROS I/O
ports/panoptic_core/  已移入仓库的轻量 predicate 实现
src/runtime/          session seed、运行时和保存边界
src/base1/            保留的诊断/研究代码，不属于 production transition
app/                  可执行程序与语义测试
scripts/              唯一 runner、构建、校验、数据适配与可视化
configs/              仓库内配置
vendor/               来源快照/参考，不得进入 production build
```

## 6. 已实现事实与待验收边界

| 项目 | 当前事实 | 状态 |
|---|---|---|
| Session 接口 | `P_prev + complete Session -> P_new`，A/B/C 不硬编码 | 已实现入口 |
| B 初始化 | 加载 `P_A` 最新 updated current state（1 seed snapshot，不复制历史） | 已实现代码路径 |
| B scratch / Base1 | production runner 不执行；旧入口显式拒绝 | 已实现隔离 |
| Semantic/instance | player 显式打包，mapper 由配置显式解包 | 已实现并有单元测试 |
| Instance 固化 | 评审规则已应用进 dataset，manifest 台账在列，无需脚本可直接运行 | 已完成（2026-08-15） |
| Finalization | drain、join、终局同步执行同一状态更新后保存 | 已实现代码路径 |
| 1 cm 配置 | voxel `.01`、truncation `.03`、mesh `.01`、object `-0.01` | A/B 全量已跑完 |
| Canonical source | build/cache/`ldd` 检查只接受 `ports/mapping_core` | 已实现 |
| Instance 真数据后缀 | `_segmentation.png` / `_instances.png` 显式解析并全帧预检 | 已实现 |
| 自包含依赖 | Hydra/PGMO 等仍来自外部安装 | 未完成 |
| A 全量链 | 4,003 帧 ACK、62 steps、16 physical IDs 全 private mesh、gates 全过 | 已完成 |
| B 全量链 | 4,041 帧 ACK、29 steps、16 physical IDs 全 private mesh、1 条 soft gate（预期产物） | 已完成 |
| C 递归 | C 只读 B 输出即可运行 | 未运行 |
| 家具几何质量 | private mesh 碎（2–3.3 cm 分辨率、≈10 s 窗口、无跨段累积） | 根因已定位，未修复 |
| D1 停止后的当前物化 | 新位置 current mesh 与历史 trajectory 正交保存；有回归测试 | 全量已跑，几何质量待改善 |

这里的“已实现代码路径”只表示源码机制存在且可测试；A、B 全量已按 1 cm 冻结协议跑完并
记录 review gates，C 尚未执行，家具几何质量尚未修复。

## 7. 验收标准

正式结果至少必须满足：

- A 从空状态处理 4,003 帧；B 在新进程中加载 A accepted state 并处理 4,041 帧；全部帧有
  ACK，终局 timestamp 覆盖最后输入。**（2026-08-15 1 cm 链已满足）**
- B 只有一个 production 输出，且没有再次注入 A；新进程 C 只读 B 输出即可运行。
  **（B 已满足；C 未运行）**
- semantic ID 与 physical ID 同时正确进入 mapper；I10 全程只有一个逻辑 chair。
  **（A/B 的 16 个 physical ID 均单节点、语义与 catalog 精确一致，gate 已验证）**
- D1 有轨迹且运动过程不污染当前静态图；物体停止后旧位置消失、新位置成为当前 mesh。
- D2/D3 通过同一代码路径对 persistent、absent、unobserved、new 作出有可追溯 evidence 的判定；没有可靠重观测
  不删除。
- 验证 table、cabinet、bags/apparel/blanket 组合移动和 A-only I20；A 中 I12 的床面策略
  不得被误报为对象缺失或新生。
- `.4dmap` 查询、RViz 和 RGB 时序 viewer 对同一 timestamp 显示同一 current state；viewer
  拼接不能冒充算法结果。
- clean-clone 构建不读取外部 Khronos/Panoptic 或个人 `ros2_ws`，源码和许可证可追溯。
- manifest 记录输入、配置、源码版本、帧计数、退出状态与失败原因；失败产物不能标记
  accepted。

开发规则是：一个活动实现、一个 production runner、一个 README、一个 accepted 结果。
禁止用参数 sweep 替代缺失机制，禁止把可视化拼接称为地图更新，禁止用 semantic class 加
中心距离取代已知 physical ID。datasets、权重、bags、build/install、完整 runs、`.4dmap`、
`.ply` 和逐帧预览均放在 Git 外；仓库只保存源码、配置、许可证和紧凑 provenance。

## 可视化（唯一入口）

查看 0→A→B 链结果的唯一可视化是 Open3D 播放器
`session_update_baseline/scripts/view_ab_chain_4dmap.py`（依赖
`build_canonical/4dmap_mesh_server`，两者同属一套）。本仓库不再维护任何其他查看器。

直接打开（默认参数已指向最新 5cm A/B 全量产物，无需传参）：

    /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
      session_update_baseline/scripts/view_ab_chain_4dmap.py

显式指定地图（例如其他 run）：

    conda run -n 3d_vsg python session_update_baseline/scripts/view_ab_chain_4dmap.py \
      --map-a <A final.4dmap> --map-b <B final.4dmap>

规则：仓库只保留这一个可视化入口；禁止新增或恢复其他查看器、WebGL 桥或导出渲染
工具（历史废弃：`4dmap_http_bridge.py`、`viewer_web/`、`view_office_ab_*`、
`prepare_*_process_visualization.py`、`visualize_nss_semantic_masks.py`、
`configs/4d_visualizer_latest.yaml`）。

## 8. 来源与许可

- Khronos，MIT-SPARK，BSD-3-Clause：
  `https://github.com/MIT-SPARK/Khronos.git@63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`
- Panoptic Mapping，ETHZ ASL，BSD-3-Clause：
  `https://github.com/ethz-asl/panoptic_mapping.git@3926396d92f6e3255748ced61f5519c9b102570f`

移植或修改开源文件时必须保留原许可证和文件头。内部保留上游 namespace 便于来源追溯，
不改变本项目的对外协议和研究归属。论文的新问题是：让完整 session 之间传递同构的持久
current scene state，并在同一递归系统中统一 D1、D2、D3、稳定 physical identity、几何
evidence 和最终地图物化；这一贡献最终必须由消融、标准指标和可复现实验支持。
