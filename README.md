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

## 当前状态（2026-08-19）

接受结果：5 cm 0→A→B 链（默认配置 `configs/room18_instance_5cm.yaml`）：

- session_a：`runs/v4_a`（4,003 帧、62 time steps、5.2 GB）。
- session_b：`runs/v32_b`（4,041 帧、65 time steps、12 GB、`review_gates: {}` 全过、
  player/mapper 双 exit code 0）。B 加载 A 最新状态作为 seed，在线处理 A→B 的 D3
  变化，同时完整保留 B 自身的 D1/D2 能力；切换是逐轮证据驱动的原子 handoff，不是
  终局后处理。
- 基线：`runs/v5_pureb`（同一 B 输入、从零构建，KPI 对照用）。
- 测试：mapping core 15/15、baseline 8/8 全部通过。
- 2026-08-15 的 1 cm 链（`configs/room18_instance_1cm.yaml`）仍为历史记录：session_a
  4,003 帧/17.9 GB、session_b 4,041 帧/22.9 GB，配置与评审规则不变。
- 评审规则已固化进 dataset：`datasets/local_ab/instance_labels/` 是唯一权威 instance
  map（I9 合并进 I7、I8 清除、袋子编号轮换、I20 仅 A），自带
  `manifest.json`（schema `reviewed_ab_physical_instances/v1`），可直接运行无需脚本；
  原生实例图完整备份在 `instance_labels_backup_20260815/`。
- 外部门禁已改为软评判：数据契约（帧账、checksum、empty depth、时间戳、重复 ID、seed
  等价）仍然 hard-fail；catalog 期望（物理 ID 集合、timeline 长度、轨迹数、语义标签、
  private mesh 存在性）全部记录到 `transition_manifest.json` 的 `review_gates` 字段，
  map 先建好、再评判、再操作。v32_b 的 review_gates 为空——没有 soft gate。

这不表示整套论文系统已经验收。当前仍需完成或验证：

1. 构建仍从外部 ROS 工作区取得 Hydra、Spark-DSG、Kimera-PGMO、ianvs、
   config_utilities 等研究依赖，尚未达到 clean clone 自包含构建（vendoring 已规划）。
2. 已知几何质量问题未解决：背景 mesh 1 cm 后明显干净，但床/桌/衣柜等大件家具的
   object private mesh 仍碎——根因已定位为三条独立机制（见第 7 节末）：
   a. object 分辨率 = `-0.01 × extent` ≈ 2–3.3 cm，不是 1 cm；
   b. 每个 track 重建只使用 frame buffer 内 ≤100 存储帧（≈10 s 窗口），连续可见的
      track 更只在 session 终点提取一次，96 s 的观察只用最后 ~10 s；
   c. 跨可见 segment 的几何是 winner-takes-all（support gate 选一者），无跨段累积；
      reconciler 虽有真实的面片级融合路径（`mergeObjectMeshes`），但只由后端 merge
      提议触发，而 `max_dt_merge_proposal: 3.0` 让跨段（相隔 >3 s）提议结构性不可达。
3. 完整 A→B→C 尚未跑完：C 只读 B 输出即可运行，但还没执行。
4. registry 关闭 verdict 与 presence 区间未联动：逐 sample 证据关闭 fragment 时只把
   mesh 置空，presence 右端点仍开放，时间线视图可能出现“空 mesh + 开放 presence”的
   空壳节点（3D 无几何，时间条仍显示在场）。需要把关闭结论传导给 Reconciler 的
   presence 账；动的是 registry↔Reconciler 边界，单独评估后实施。
5. KPI 的朴素 recall 对 moved 对象会被 pure-B 自身的跨位置 union 压低（见第 2.4 节
   cabinet 实例）；`measure_ab_kpi.py` 已提供站点感知 recall（`siteR` 列）作为正确
   对照口径。

### 2026-08-18/19 进展

- **在线 inherited→B 切换（5a775ec, 8e8b3ed）**：B 加载 A 最新状态后，A 的旧表面按
  当前帧实测深度逐 sample 验证；矛盾证据成立即原子 handoff（旧 fragment 进 history、
  B 状态成为 CURRENT），不再等到 session 终点，也不再出现 old∪new union 帧。
- **实测深度证据（8e8b3ed）**：`PhysicalEvidenceStore` 每帧保存 quantized depth RLE，
  端点证据带 `measured_depth_m`，据此区分“同深度被替换”与“更近处遮挡”。
- **通用 moveability prior（83db26b）**：删除 C++ 里 `isHighMobilitySemantic`
  （bed/shelf/desk/wardrobe 低移动）和 1.0 m 台面高度 hack。moveability 改为
  观测 D1 历史 + 已关闭 fragment（搬家频率）+ 配置驱动的语义本体
  （`backend.high_mobility_semantic_labels`）。它只用于判断表面重叠是否可信共同观测，
  从不自行删除或合并任何东西。
- **跨位置绝不 union（83db26b）**：物化、session 吸收、顶层吸收、终局 A+B 补全四条
  路径全部加门控——可移动身份的两处几何只有实测共享表面才允许合并；不同位置的候选
  作为独立 hypothesis 归档为 closed history fragment，不合并不删除。unidentified
  端点不再投 absent 票（可能是丢失跟踪 ID 的同一物体）。
- **六类证据账本（83db26b）**：每轮每 ID 输出 `STATE_SLICE` 日志行（supported /
  free-space / replaced-by-other / replaced-by-background / occluded / unobserved），
  `scripts/diag_tools/parse_state_slices.py` 转成 CSV；v32 账本在
  `runs/v32_b/state_slices/`。
- **KPI 站点感知 recall（5c7d286）**：`measure_ab_kpi.py` 新增 `siteR` 列——moved
  对象只与 pure-B 中离 B 当前质心最近的站点簇比较，避免 pure-B 自身跨位置 union 污染
  基线。
- **测试对齐冻结契约（6de589d）**：两个 baseline 测试按证据驱动契约更新（无证据时
  inherited 保持 CURRENT、矛盾证据触发 handoff、逐 sample 替换投票关闭）。
- 数据保留（Git 外）：`runs/v4_a`（A）、`runs/v5_pureb`（干净 B）、`runs/v32_b`
  （接受的 0-A-B 结果，含 `state_slices/` 证据账本与 `kpi_vs_pureb.txt`）；其余历史
  run 已清理。

## 核心判据：RGB-D 是真实世界，map 是对真实世界的估计

本系统里“正确”只有一个定义：**同一时刻，RGB-D 看到什么，map 的 CURRENT 就应该显示
什么**。RGB-D 是真实世界；map 是我们对真实世界的估计。系统每一轮用当前 RGB-D 校正
map，让 map 和真实世界一样在线进化。判断对错不是“old/new 哪个更像”，而是“真实世界
已经变了，map 就必须跟着变”。

对旧 surface 的每个 sample 点 P 独立提问，用当前帧的实测值回答（`z_map` = P 在当前
相机位姿下的期望深度，`z_obs` = 该像素实测深度，`id_obs` = 该像素 physical ID，
`tol` = 传感器/位姿容差）：

| 条件 | 证据 | 含义 |
|---|---|---|
| 无有效 depth / P 不在 FOV | UNOBSERVED | 没看到，不投票 |
| `z_obs < z_map - tol` | OCCLUDED | 前面有东西，看不见，不投票 |
| `|z_obs - z_map| ≤ tol` 且 same ID | SUPPORTED | 旧表面还在 |
| `|z_obs - z_map| ≤ tol` 且 background | REPLACED_BY_BACKGROUND | 旧位置已变空背景 |
| `|z_obs - z_map| ≤ tol` 且 different ID | REPLACED_BY_OTHER | 旧位置被别的物体占据 |
| `z_obs > z_map + tol` | FREE_SPACE | ray 穿过旧表面，已经空了 |

同一位置同深度的不同 ID 是**替换**，不是遮挡——cabinet 占据 apparel 旧位置的一部分
时：被 cabinet 同深度占据的 sample 记 REPLACED_BY_OTHER，被 ray 穿过的 sample 记
FREE_SPACE，两者都是 absence 证据，apparel 旧 fragment 因此关闭；只有更近处的
cabinet 才记 OCCLUDED（只是挡住，不删）。

部分可见时只看被看到的 sample：SUPPORTED / FREE_SPACE / REPLACED_BY_OTHER /
REPLACED_BY_BACKGROUND 是 decisive votes；OCCLUDED / UNOBSERVED 不投票、不删除。
fragment 级判定：

```text
absence_votes > support_votes  -> 旧 fragment 被可靠否定，退出 CURRENT
support_votes > absence_votes  -> 仍在，继续补全
decisive votes == 0 或相等      -> 没看到或冲突，保持 unresolved，不动作
```

时间上还要看同一 sample 的多次观测：多帧多视角一致的 free/replaced 才可靠；一会儿
support 一会儿 free 的 sample 保持冲突，不参与最终投票。

状态切换语义（在线执行，不等待 finalization）：

```text
old confirmed absent + new READY    -> 原子 handoff（一帧内旧退历史、新进 CURRENT）
old confirmed absent + new BUILDING -> 旧 fragment 退出 CURRENT，新 fragment provisional
candidate 存在但 old 未解决          -> 两个 hypothesis 分开，绝不 union
identity conflict                   -> 不 union，不自动删
```

验证手段（都在 v32 上跑通）：

- `runs/v32_b/state_slices/state_slices.csv`：每 ID 每轮六类证据计数，直接回答
  “旧位置看到什么、什么时候没的、新位置建到哪一步、因为什么切换”；
- `scripts/diag_tools/diag31 <map> <inst> 0 N 1`：逐时间片 CURRENT mesh
  （顶点数/中心/span），检查切换帧有没有 union 帧或 span 爆炸；
- `scripts/measure_ab_kpi.py`（含 `siteR` 站点感知 recall）：与 pure-B 对照验收。

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

## 2. 研究目标：记忆只能带来正收益

D1 + D2 是一次完整 mapping session 内部的动态世界建模。

D1：observed dynamics——东西在机器人看着的时候动。要保留 trajectory、temporal
geometry / bbox / timestamp 等动态历史。

D2：hidden / out-of-view change——同一个 session 还在运行，但机器人没看着的时候东西发生
变化。重新看到时，需要判断 persistent / absent / unobserved / new，而不是虚构中间轨迹。

Khronos 本身就是基础：一个完整 dense D1+D2 mapper。D3 不是“又一种 change detection”，
D3 是 complete sessions 之间的时间边界：

```text
Session A
 ├─ D1: visible motion
 ├─ D2: hidden change
 └─ 得到完整 dynamic scene memory
            │
            │ A结束
            ▼
          D3（跨 session 的变化）
            │
            ▼
Session B
 ├─ load A 的 scene memory
 ├─ A → B 的变化
 ├─ 同时重新具有完整 D1 能力
 ├─ 同时重新具有完整 D2 能力
 └─ 再输出 scene memory
            │
            ▼
          D3
            │
            ▼
Session C ...
```

Session A：一次完整的动态建图运行，里面正常做 D1 + D2。D3：A 结束以后，机器人离开；
环境继续变化；之后新的独立 Session 读取 A 留下的 scene memory，处理这段跨 Session 的
变化。Session B：独立启动的新一次运行——不是只“更新旧地图”，而是读取 A 的 memory、
处理 D3，然后自己再次正常执行新的 D1 + D2。Session C：证明这套东西不是只做一次 A/B
pairwise comparison；B 结束后还能输出同类 memory，再让 C 继续。

目标是：

> **A→B 的动态建图，在 B 时刻的 CURRENT 地图必须至少和从零单独建 B 一样准确；同时因为
> 保留并正确利用了 A 的信息，完整的 persistent map 必须比单独 B 更丰富、更完整。**

这包含两个同时成立的要求。

### 2.1 B 时刻的 CURRENT 不能比 pure-B 差

```text
CURRENT(A→B) ≥ CURRENT(pure-B)
```

同样的 B RGB-D 输入下：B 单独能建出来的 table、cabinet、fan，A→B 也必须完整建出来；
不能因为继承了 A，反而把 B 的 table 削薄、延迟出现、来回消失；不能让 A 的旧位置污染
B CURRENT；不能 old mesh 和 new mesh 直接 union。

对移动物体，正确结果是：

```text
A old state → HISTORY
B new state → CURRENT
```

B 的新 CURRENT 至少要达到 pure-B 的质量。将来如果有可靠的 object-local canonical model，
A 的历史还可以帮助补全 B；但不能把 A 的旧 world-space mesh 直接搬过去冒充补全。

### 2.2 整个 persistent map 必须比 pure-B 更完整

这份“更完整”不等于把 A 的所有几何继续显示在 CURRENT，它来自三个地方：

1. **静态区域的多视角补全**：例如衣柜没动，A 看见正面 + B 看见背面 = A→B CURRENT
   比 pure-B 更完整。
2. **历史信息**：pure-B 只有 B；A→B 应该有 A 的旧状态、A→B 的移动/观测空档、B 的当前
   状态、D1 真实轨迹、D2/D3 temporal fragments。
3. **未被 B 重新观察的静态区域**：B 单独可能根本没看到某些角落，但 A 已正确建立；只要
   B 没有给出 contradiction，这些静态信息应该继续保留。

### 2.3 三条缺一不可的指标

```text
B 当前准确度：      Q_current(A→B) >= Q_current(pure-B)
历史与场景完整度：  I_persistent(A→B) > I(pure-B)
旧影：              Ghost_current(A→B) = 0
```

只做到“B 新 geometry 很完整但旧 table 还在 CURRENT”不算成功；只做到“旧 table 清干净
但 B 新 table 比 pure-B 烂”也不算成功；只做到“CURRENT 和 pure-B 一样但 A 的历史、静态
补全全部丢了”仍然不算 lifelong mapping。

### 2.4 当前验证状态（v32 基线）

接受的 v32_b 对 pure-B 的测量（记录在 `runs/v32_b/kpi_vs_pureb.txt`，
`scripts/measure_ab_kpi.py` 生成）：

- 15 个 object 中 14 个对 pure-B 的 surface recall = 100%、precision 100%，说明继承 A
  没有损坏 B 本来能重建出的表面；静态对象获得明显 A+B 多视角增益（wardrobe 43,221 /
  pure-B 17,496、shelf 82,638 / 41,742、desk 7,653 / 4,161、bed 23,712 / 16,731）。
- 移动对象切换帧没有 old∪new union：cabinet f8 纯旧 5808 → f9 纯新 2730，table
  f11 → f12，apparel 无 union 帧；v30 时代 cabinet f40 span 5.47 m 的跨位置并集消失。
- cabinet 的朴素 recall 只有 52.5%——根因是 pure-B 自身：它的 reconciler 把 cabinet 的
  X/Y/Z 三个位置 union 进一个节点（pure-B 自己的 span 也是 5.47 m）。这正是
  “不能把 pure-B 的 D2 残留当成正确结果”的实测例证。按站点感知 recall（`siteR`，只与
  离 B 当前质心最近的 pure-B 站点簇比较）cabinet = 98.5%，其余 moved 对象全部 100%。
- moved old-geometry-still-CURRENT 均值 6.8%：全部来自近距离移动的物理重叠（table
  移动 ≈0.4 m，其 2.16 m 宽的 A mesh 与 B 新位置自然重叠 37%），不是跨位置 union；
  真正的 ghost（旧位置 mesh 仍留在 CURRENT）为 0。

### 2.5 最终代码只应该服务这个循环

```text
读取上一时刻 persistent state
        ↓
用当前 B RGB-D 更新场景
        ↓
还在的静态表面：保留并补全
已经空掉的旧状态：退出 CURRENT，进入 HISTORY
移动物体的新状态：使用 B 观测建立 CURRENT
遮挡/没看到：不乱删、不乱合并
        ↓
保存比 pure-B 信息更多、但 CURRENT 同样准确的状态
        ↓
C 只加载 B，继续相同循环
```

以后判断任何算法修改，先回到唯一判据（RGB-D 看到什么，CURRENT 就显示什么），再问四个
问题：

1. 它会不会让 B CURRENT 比 pure-B 差？（moved 对象按第 2.4 节站点感知 recall 比较）
2. 它会不会把移动物体的历史旧位置继续放在 CURRENT？
3. 它有没有保留 A 带来的静态补全和时间历史？
4. 它有没有留下可验证的证据链（每轮六类证据计数 + 切换原因），而不是只给一个结果？

任何一条答案不对，这个修改就不能接受。要构建的不是单纯的“去 ghost 系统”，而是
**具有单调信息增益的 recursive dynamic mapping**：每次 revisit 都不降低当前地图准确度，
同时不断增加可靠的空间完整度和时间历史。

## 3. 地图、实体与状态不变量

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

## 4. 输入数据与显式 label 协议

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

## 5. 唯一运行入口

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

标准配置为 `configs/room18_instance_1cm.yaml`（1 cm）；**接受结果 v32_b 使用默认的
`configs/room18_instance_5cm.yaml`（5 cm）**，参数如下：

```text
image scale                       0.5 (960x540)
flow control                      per-frame ACK
min_output_separation             0.4 s
change_detection_every_n_backend_updates   5
save_every_n_frames              0
voxel / truncation               0.05 m / 0.15 m
mesh resolution                  0.005 m
object_reconstruction_resolution -0.01（= 1% × object extent，≈2–3.3 cm，见已知问题）
backend.high_mobility_semantic_labels [10, 15, 74, 75, 92, 115, 131, 139]
removed min_weight 修复（去除 5 帧污染）
```

`high_mobility_semantic_labels` 是 moveability 先验的语义本体部分（ADE 类别 ID：
cabinet/table/monitor/chair/apparel/bags/blanket/fan；静态家具 bed/shelf/desk/
wardrobe 刻意不在列）。它只是“表面重叠是否可信”的弱提示，从不自行删除或合并；先验的
另外两部分（观测 D1 历史、已关闭 fragment 的搬家频率）不依赖语义。

2026-08-19 接受的 5 cm A、B 命令（产物 `runs/v4_a`、`runs/v32_b`，每个 session 的
`control/command.txt` 有完整复现记录）：

```bash
# session_a（空状态起步）
./scripts/run_session.sh \
  --run-dir datasets/local_ab/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p \
  --semantic-dir datasets/local_ab/semantics/session_a \
  --instance-dir datasets/local_ab/instance_labels/session_a \
  --output-state runs/v4_a

# session_b（加载 A，world transform 对齐）
./scripts/run_session.sh \
  --input-state runs/v4_a \
  --run-dir datasets/local_ab/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p \
  --semantic-dir datasets/local_ab/semantics/session_b \
  --instance-dir datasets/local_ab/instance_labels/session_b \
  --world-transform datasets/local_ab/alignment/session_b_to_session_a.txt \
  --output-state runs/v32_b
```

2026-08-15 的 1 cm 链（`configs/room18_instance_1cm.yaml`，产物
`runs/session_ab_1cm_20260815/`）仍是历史记录，命令见当时的
`control/command.txt`。

继承设计（解释 B 的时间步与体积）：

- B 加载 A 的 `final.4dmap` 时只取**最新一个 updated current DSG** 作为 1 个 seed
  snapshot（`latestSessionSeed` + `initializeSessionTimeline`），A 的整段历史
  **不递归复制**；B 的 timeline = 1 seed step + B 自己触发变化检测产生的快照。
  B 快照少于 A（v32 的 65 steps 里大部分是继承后变化检测的产物）是因为 B 拍到的大部分
  内容 A 已建过，change detection 只对新增/变化触发——这是 D2/D3 语义的正确产物。
- 每个 snapshot 都是**完整场景**（不是 delta）：B 的每个快照都携带 A 的 inherited
  background mesh，所以 `final.4dmap` 体积大于 B 的新内容。这是格式语义，不是泄漏。

### 固定建图流程（端到端，2026-08-19 更新）

1. **准备 dataset（一次，已固化）**：`datasets/local_ab/` 下 rgbd/、semantics/、
   instance_labels/（评审规则已应用、manifest 台账在列、可直接运行）、alignment/
   （session_b_to_session_a.txt，world transform 左乘）。
2. **构建 canonical mapper**：`./scripts/build_canonical.sh`（源码/配置变更后；
   测试 mapping core 15/15 + baseline 8/8 通过），再用
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
   并 echo 到 stderr，**不拒绝已完成的地图**（先建好、再评判、再操作）。v32_b 的
   review_gates 为空。
6. **对照与验证**：`scripts/measure_ab_kpi.py`（含 `siteR` 站点感知 recall）对
   pure-B 对照；`scripts/diag_tools/diag31` 看逐时间片切换帧；
   `scripts/diag_tools/parse_state_slices.py <log> --out DIR` 把六类证据账本转成 CSV。
7. **已知几何问题**（见“当前状态”第 2 条）：大件家具 private mesh 仍碎，三根因已
   定位（分辨率 ≈2–3.3 cm / 重建窗口 ≈10 s / 跨段 winner-takes-all 无累积）；修复
   方向：object resolution 改正数 0.01、放宽/定期重建窗口、或引入跨段 private TSDF
   累积。这些只影响 object geometry 质量，不影响数据契约与 ID/语义正确性。

## 6. 构建与源码边界

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

## 7. 已实现事实与待验收边界

| 项目 | 当前事实 | 状态 |
|---|---|---|
| Session 接口 | `P_prev + complete Session -> P_new`，A/B/C 不硬编码 | 已实现入口 |
| B 初始化 | 加载 `P_A` 最新 updated current state（1 seed snapshot，不复制历史） | 已实现代码路径 |
| B scratch / Base1 | production runner 不执行；旧入口显式拒绝 | 已实现隔离 |
| Semantic/instance | player 显式打包，mapper 由配置显式解包 | 已实现并有单元测试 |
| Instance 固化 | 评审规则已应用进 dataset，manifest 台账在列，无需脚本可直接运行 | 已完成（2026-08-15） |
| Finalization | drain、join、终局同步执行同一状态更新后保存 | 已实现代码路径 |
| 1 cm 配置 | voxel `.01`、truncation `.03`、mesh `.01`、object `-0.01` | A/B 全量已跑完（历史） |
| Canonical source | build/cache/`ldd` 检查只接受 `ports/mapping_core` | 已实现 |
| Instance 真数据后缀 | `_segmentation.png` / `_instances.png` 显式解析并全帧预检 | 已实现 |
| 自包含依赖 | Hydra/PGMO 等仍来自外部安装 | 未完成 |
| A 全量链 | 4,003 帧 ACK、62 steps、16 physical IDs 全 private mesh、gates 全过 | 已完成 |
| B 全量链（接受） | v32_b：4,041 帧 ACK、65 steps、16 physical IDs、`review_gates: {}`、exit 0/0 | 已完成（2026-08-19） |
| 在线 inherited→B 切换 | 逐 sample 深度证据 → 原子 handoff，无 old∪new union 帧 | 全量已跑并验收 |
| 实测深度证据 | 端点证据带 `measured_depth_m`，区分替换/遮挡 | 已实现并有账本 |
| moveability prior | D1 历史 + 搬家频率 + 配置语义本体，无硬编码类别/高度 | 已实现（83db26b） |
| 跨位置 union 门控 | 物化/吸收/终局四条路径共享表面门控；候选归档不删除 | 全量已跑并验收 |
| 六类证据账本 | `STATE_SLICE` 每 ID 每轮六类计数 → CSV（`parse_state_slices.py`） | v32 账本在 `runs/v32_b/state_slices/` |
| KPI 站点感知 recall | `measure_ab_kpi.py` `siteR` 列，避免 pure-B ghost union 污染基线 | 已实现（5c7d286） |
| 单元/回归测试 | mapping core 15/15、baseline 8/8 | 全部通过 |
| C 递归 | C 只读 B 输出即可运行 | 未运行 |
| 家具几何质量 | private mesh 碎（2–3.3 cm 分辨率、≈10 s 窗口、无跨段累积） | 根因已定位，未修复 |
| D1 停止后的当前物化 | 新位置 current mesh 与历史 trajectory 正交保存；有回归测试 | 全量已跑，几何质量待改善 |
| presence 联动 | registry 关闭 verdict 未写回 presence 右端点 | 已定位，未实施 |

这里的“已实现代码路径”只表示源码机制存在且可测试；A、B 全量已按冻结协议跑完并记录
review gates（接受的 5 cm 链为 v32_b），C 尚未执行，家具几何质量与 presence 联动尚未
修复。

## 8. 验收标准

正式结果至少必须满足：

- A 从空状态处理 4,003 帧；B 在新进程中加载 A accepted state 并处理 4,041 帧；全部帧有
  ACK，终局 timestamp 覆盖最后输入。**（v32_b 5 cm 链已满足：`review_gates: {}`、
  exit 0/0）**
- B 只有一个 production 输出，且没有再次注入 A；新进程 C 只读 B 输出即可运行。
  **（B 已满足；C 未运行）**
- semantic ID 与 physical ID 同时正确进入 mapper；I10 全程只有一个逻辑 chair。
  **（A/B 的 16 个 physical ID 均单节点、语义与 catalog 精确一致，gate 已验证）**
- D1 有轨迹且运动过程不污染当前静态图；物体停止后旧位置消失、新位置成为当前 mesh。
  **（v32：cabinet/table/apparel 切换帧无 union，I20 已移除）**
- D2/D3 通过同一代码路径对 persistent、absent、unobserved、new 作出有可追溯 evidence 的判定；没有可靠重观测
  不删除。**（v32：每轮每 ID 六类证据账本在 `runs/v32_b/state_slices/`）**
- 验证 table、cabinet、bags/apparel/blanket 组合移动和 A-only I20；A 中 I12 的床面策略
  不得被误报为对象缺失或新生。**（v32 KPI：moved 对象站点感知 recall ≥98.5%、true
  ghost = 0）**
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

直接打开（默认参数已指向接受的 5 cm A/B 全量产物：`v4_a` → `v32_b`，无需传参）：

    /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
      session_update_baseline/scripts/view_ab_chain_4dmap.py

显式指定地图（例如其他 run）：

    conda run -n 3d_vsg python session_update_baseline/scripts/view_ab_chain_4dmap.py \
      --map-a <A final.4dmap> --map-b <B final.4dmap>

规则：仓库只保留这一个可视化入口；禁止新增或恢复其他查看器、WebGL 桥或导出渲染
工具（历史废弃：`4dmap_http_bridge.py`、`viewer_web/`、`view_office_ab_*`、
`prepare_*_process_visualization.py`、`visualize_nss_semantic_masks.py`、
`configs/4d_visualizer_latest.yaml`）。

## 9. 来源与许可

- Khronos，MIT-SPARK，BSD-3-Clause：
  `https://github.com/MIT-SPARK/Khronos.git@63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`
- Panoptic Mapping，ETHZ ASL，BSD-3-Clause：
  `https://github.com/ethz-asl/panoptic_mapping.git@3926396d92f6e3255748ced61f5519c9b102570f`

移植或修改开源文件时必须保留原许可证和文件头。内部保留上游 namespace 便于来源追溯，
不改变本项目的对外协议和研究归属。论文的新问题是：让完整 session 之间传递同构的持久
current scene state，并在同一递归系统中统一 D1、D2、D3、稳定 physical identity、几何
evidence 和最终地图物化；这一贡献最终必须由消融、标准指标和可复现实验支持。
