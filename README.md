# Session Update Baseline：跨会话动态建图

本项目在 Khronos 上实现持续的 metric-semantic 4D mapping。Khronos 提供单次 session 的输入处理、跟踪、TSDF、网格生成和 D1/D2；项目扩展维护稳定物理身份、状态有效期、观测证据、旧表面更新及跨进程接续。数学统一的对象是这些扩展共享的状态与证据，工程实现直接复用已有建图基础。

活动仓库：`/home/jixian/Desktop/FT/session_update_baseline_project`。源码位于 `session_update_baseline/`；`/home/jixian/Desktop/FT/session_update_baseline` 是该目录的符号链接。当前分支 `dev`，算法恢复基准为 `48a3033b7ef6203f84ef68b7b2a8c97a3eaa8799`。本文对应恢复后的实现与实际接入边界。

**本文件是唯一活动 Markdown，也是唯一原理说明。** 历史设计和成绩保存在 `archive/`；当前运行、审查、实验使用 JSON、CSV、TXT 和日志。代码、构建、运行与输出统一留在 WSL。

## 1. 系统目标与完整递推

D1 是持续可见的运动，D2 是同一运行内的隐藏变化，D3 是跨 session 空档的隐藏变化。每个 session 启动新进程和短期跟踪器，恢复前序地图后继续执行完整 D1/D2。

设会话输入为 \(Y_k=\{(t_i,I_i,D_i,L_i,T_i)\}_{i=1}^{n_k}\)，分别包含时间、颜色、深度、物理身份和相机位姿；语义类别另行保存。持久状态为 \(P_k\)。完整算法的组合为

\[
\boxed{
P_k=\operatorname{Save}\!\left[
\operatorname{Surface}\!\left(
\operatorname{Drain}\!\left[
\operatorname{Fold}_{Y_k}(\operatorname{Khronos}+\operatorname{IdentityEvidence},
                         \operatorname{Load}(P_{k-1}))
\right],Y_k\right)\right].
\tag{1}
\]

式中 `Khronos + IdentityEvidence` 表示已有在线执行链在确定的回调点组合项目扩展。`Drain` 排空处理队列、异步变化检测和最后到达的物体片段；`Surface` 用最终状态限定观测，再更新会话末表面；`Save` 提交下一进程读取的状态。

初值 \(P_0=\varnothing\)，同一运行入口完成 \(P_0\to P_A\to P_B\to P_C\)。下一会话读紧邻前序状态；旧 session 的完整输入视频和整条快照序列留在原产物中。

研究验收同时衡量三个目标：

\[
Q_{\rm current}(A\to B)\ge Q_{\rm current}(B),\quad
I_{\rm persistent}(A\to B)>I_{\rm persistent}(B),\quad
\operatorname{Ghost}_{\rm current}(A\to B)=0.
\tag{2}
\]

这是评价目标。准确度按真实 RGB-D/GT 检查，信息增益来自未重观测区域的保留、静态多视角补全和时间历史。移动物体旧状态进入 HISTORY，新位置进入 CURRENT；全过程和最终时刻分别检查。

## 2. 源码归属与 Khronos 边界

```text
session_update_baseline/
  session_core/
    include/                 项目状态、证据、几何与运行接口
    src/state/               持久物理身份和片段状态
    src/evidence/            实测物理证据与传感器统计
    src/surface/             会话末几何更新、帧归档、空间查询
    src/adapters/            物理射线查询、物体合并、静态选帧、D3 编排
    src/runtime/             SessionBackend、状态指纹、运行支持
  ports/mapping_core/        复用的 Khronos 与 ROS 接入
  ports/panoptic_core/       已移植的通用 predicate
  app/                      生产程序和检查程序
  scripts/                  构建、唯一 runner、数据协议、唯一查看器
  configs/                  运行配置
  src/base1/                历史诊断工具
  vendor/                   来源快照
```

`session_core` 是独立 CMake OBJECT 目标，对象文件链接进现有 mapper 库。算法函数体集中在该目录，兼容原有 namespace、成员声明和数据类型。Khronos 头文件保留项目所需的声明和状态成员；适配接入负责传递已有几何、证据及时间。

| 职责 | 使用的实现 |
|---|---|
| 输入、短期跟踪、活动窗口、在线 TSDF、对象提取 | Khronos 及现有适配 |
| 原生射线变化检测、reconciler、4D map | Khronos 及已验证的既有改进 |
| stable physical ID、片段有效期、继承状态裁决 | `session_core/src/state/` 与 `adapters/` |
| 真实像素证据、覆盖与缺失统计 | `session_core/src/evidence/` 与 `adapters/physical_ray_queries.cpp` |
| 会话末表面更新 | `session_core/src/surface/session_refusion.cpp` |
| 恢复、终结、保存、接续 | `session_core/src/runtime/` 与 `adapters/backend_session.cpp` |

原生 `Backend` 默认关闭项目会话扩展；项目 runner 在空状态和继承状态下均使用 `SessionBackend`。输入为空时开始新地图，输入有效时恢复前序地图。已有 D1/D2 改进按功能保留。

源码边界固定如下：新增项目算法写入 `session_core/`；项目内 `ports/mapping_core/khronos` 保留基础实现、原有有效适配及必要的接口声明/调用。外部 `/home/jixian/ros2_ws/src/khronos` 位于本仓库之外，作为源码参考；生产链接使用本项目 canonical 安装。外部 baseline 的历史修改保留原状，本次整理只读核查。具体文件及哈希见 `reports/rewrite_start_boundary_20260930.json`。

当前状态作为 **rewrite 开始点**：算法已按 `48a3033` 恢复，项目函数体已集中，A→B 短程运行已验证。后续重写从此检查点演进项目算法，Khronos 基础按上述边界复用。

上游比对使用官方 Git 对象 `MIT-SPARK/Khronos@63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`，直接读取该提交内容。最初 64 项 TODO 清理涉及其中 20 个原生函数，已恢复其实现；项目 TODO 的七个源文件/头文件也已从 `48a3033` 恢复，并保留新目录布局。精确清单见 `reports/khronos_upstream_boundary_20260930.json` 和 `reports/todo_restore_20260930.json`。

## 3. 身份、状态与时间

每个物理实体 \(\ell\) 有独立于语义类别和 DSG node ID 的身份。实体的运行状态写作

\[
X_\ell=(\mathcal F_\ell,c_\ell,n_\ell,b_\ell,\mathcal A_\ell,\eta_\ell).
\tag{3}
\]

\(\mathcal F\) 保存时序片段；\(c\) 为 current；\(n\) 为候选；\(b\) 为继承实体的本会话子状态；\(\mathcal A\) 是已摄取观察区间；\(\eta\) 保存观测统计及运动先验。一个片段包含几何、包围盒、语义、出生、最近支持、传感器确认和关闭时间。

时间有明确角色：观察时间来自传感器；状态开始时间限定观测有效域；裁决时间标识执行轮次。合并片段时出生/首次观察取最早，支持/确认取最晚。关闭时间为

\[
t_{\rm close}=\max(t_{\rm decision},t_{\rm last\ support}).
\tag{4}
\]

片段按 `(first,last,node_id)` 摄取；重复区间在入口识别。首个观察打开 current。带可见运动历史的观察关闭原片段并打开新片段。继承实体先把本会话观察送入 \(b\)，让旧位置与新位置分别接受证据。同一会话内，体素重叠且其后仍有表面支持的观察可加入 current，其余进入候选。

设本轮支持数 \(S\)、反证数 \(C\)、可靠缺失门 \(g\in\{0,1\}\)，有效反证 \(C^*=gC\)。候选与 current 的扩张包围盒分离指示为 \(D\)。普通 current 的关闭条件为

\[
J=[C^*>S]\lor[D\land S=0].
\tag{5}
\]

关闭时提升可用候选；有直接支持时更新确认时间。候选时间已到且物体具静态先验或共享表面时合入当前片段。

继承状态还使用本会话已建立的实体几何：实体具运动倾向、本会话可靠样本至少 30、且超过一半顶点距旧表面顶点大于 0.10 m 时，判定实体已在另一位置。此项与式 (5) 的有效反证共同驱动继承态退出；本会话 current 接任。运动倾向由配置语义、实际 D1 历史和已关闭片段提供。

终结按相同规则处理候选和本会话子状态，再把最后的状态物化进 DSG。几何合并执行当前基准的网格拼接；实体时间片段与几何存储分别保持各自含义。

源码：`session_core/src/state/persistent_object_state.cpp`、`src/adapters/physical_object_merge.cpp`、`src/adapters/backend_session.cpp`。

## 4. 同一测量的几何含义与物理证据

相机中心为 \(c_i\)，世界射线方向为 \(d_i(u)\)，径向测距为 \(r_i(u)\)。测量端点为 \(p_i(u)=c_i+r_i(u)d_i(u)\)。查询旧点 \(x\) 时

\[
q_i(x)=\|x-c_i\|,\qquad e_i(x)=r_i(\pi_i(x))-q_i(x).
\tag{6}
\]

光轴 Z 深度由相机模型转换到径向距离后比较。世界变换统一左乘；几何和测量使用同一坐标系。点证据使用式 (6)，体素积分使用其投影像素方向上的 projective 残差。

| 实测关系 | 物理实体证据 |
|---|---|
| 视野外或无有效深度 | unobserved |
| 测量在旧点前方 | occluded |
| 深度吻合且相同物理 ID | supported |
| 深度吻合且其他 ID / 背景 | replaced by other / background |
| 射线穿过旧点到达更远表面 | free space |

几何存在性允许任意身份的同位置测量提供支持；实体存在性同时检查物理身份。查询使用状态最后支持之后的实际测量；遮挡与视野外各自保留为观察条件。

物体样本按空间格去重，维护各样本的支持与穿透时间。可靠样本上的观空比例 \(f\) 由自身与总体历史拟合 \(p_{\rm present}(f)\)（Beta 密度），当前实现递推

\[
L^+=\max(0,L-w\log p_{\rm present}(f)),\quad
w=\min(1,N_{\rm fresh}/\max(1,N_{\rm reliable})),\quad
 g=[L^+>\log99].
\tag{7}
\]

均值/方差使用已观测统计的稳健估计，密度与比例按源码有界化；阈值 \(\log99\) 是当前参数。其输出授权式 (5) 的反证进入裁决。通过后重置累计量；传感器总体统计随会话保存。标签统计保留在证据记录中，当前覆盖门使用几何项。

关闭物体对应的背景副本由 `closed_object_background.cpp` 查询实际后续深度：背景表面自身时间参与查询，同位置实测表面继续提供支持，最后几何支持之后的穿透才产生背景变化。静止物体同时参与背景和 private mesh 的既有表示由此协同更新。

## 5. 状态授权的会话末观测域

令 \(t_\ell\) 为本会话新开的当前状态首次观察时间，恢复基准从 registry 的 `track_first_seen` 读取，缺值时取 `birth_time`。早于 \(t_\ell\) 的本身份像素退出当前状态的融合。其他旧像素保留实测端点，自由空间受到当前物体首交 \(\lambda\) 限制。

体素积分设截断距离 \(T\)、projective 残差 \(\delta\)，测量的合法贡献域为

\[
\delta>-T\quad\land\quad(\delta\le T\ \lor\ \widetilde q<\lambda-T).
\tag{8}
\]

\(\widetilde q\) 是体素沿采样射线的投影距离。\(\lambda\) 先查当前物体三角面，缺少合格命中时使用实测端点体素首交。式 (8) 保留观测表面的局部带和当前物体之前的自由空间。

合法贡献进入会话末 `PresentTsdf`：

\[
F^+(x)=\frac{W(x)F(x)+\min(1,\delta/T)}{W(x)+1},\qquad W^+(x)=W(x)+1,
\quad P=\operatorname{MC}(F=0).
\tag{9}
\]

只在八角均已积分的 cube 提取表面。这里的 `PresentTsdf` 是既有会话末模块；在线融合继续调用 Khronos。新面按面心到当前物体三角面的最近距离关联身份，一个物体体素内继承对应物理 ID，其余归背景。

表面公差使用 \(\tau(h,q)=\max(h,\sigma(q))\)，\(h\) 是层体素的一半；\(\sigma\) 由当前表面与测量的距离分箱残差估计。跨视图深度尺度 \(s\) 由残差的稳健中位数拟合，当前与历史尺度共同定义旧表面的位移误差窗口。

源码：`session_core/src/surface/session_refusion.cpp`、`present_tsdf.cpp`、`triangle_grid.cpp`、`frame_archive.cpp`。

## 6. 旧表面延拓与一次装配

当前恢复基准对带身份、来源和时间的旧表面执行以下明确判定。直接表面支持 \(H\)、有效穿透 \(F\)、遮挡 \(B\)、截断带内遮挡 \(B_T\) 均来自第 5 节观测域。投影邻域的命中取有效支持，穿透要求有效采样均穿透；自由空间同时检查 \(\lambda\)。

直接观测的延拓条件写为

\[
K_{\rm obs}(x)=
\begin{cases}
[H\ge F],&H+F>0,\\
[B\ge2B_T],&H=F=0.
\end{cases}
\tag{10}
\]

平票和零证据延续旧表面。对旧点 \(x\)，设当前表面最近点 \(y\)、距离 \(d\)、最近有效 reaching 相机 \(c\)、量程 \(q\)，历史正尺度最大值为 \(s_A\)，本会话正尺度为 \(s_B\)。错位副本条件为

\[
D(x)=[s_A+s_B>0]\land\operatorname{reached}(x)
\land[2h<d\le\tau(h,q)+(s_A+s_B)q]
\land[(y-x)\cdot(c-x)>0].
\tag{11}
\]

它检验视线方向、当前表面对应与尺度误差窗。几何存储中精确相同的点面可以共享表示；式 (11) 则会改变表面选择，属于算法判定。

同身份内部判定用于已进入视野且离当前表面超过一个物体体素的旧点：沿固定 64 个方向查同 ID 当前表面的首交，按面法向计入 \(N_{\rm in}\)、\(N_{\rm out}\)，取

\[
I(x)=[N_{\rm in}>N_{\rm out}].
\tag{12}
\]

有 shown memory 时，旧面保留位为

\[
K(f)=\operatorname{stateValid}(f)\,
K_{\rm obs}(\operatorname{center}f)\,[\neg D(\operatorname{center}f)]
\prod_{v\in f}[\neg I(v)].
\tag{13}
\]

`stateValid` 对背景成立；物体要求 current 存在且该状态始于本会话之前。旧图只有推理几何时，恢复基准的后备路径对其中 memory 顶点执行内部判定，再与新表面装配。

本会话在线几何还提供 fill 候选：面心 cube 存在未积分角，且面心有实测支持时保留。最终几何按 fill、present、保留旧面的顺序装配，身份决定输出槽；属性按来源保存，新增顶点从同槽原几何取属性并截时。物体输出槽为空时保留输入槽，背景直接提交。全部结果先计算到局部数据，再写入最终地图。

式 (10)—(13) 是当前恢复版的三个几何决策及其组合，源码继续保留这三个判定。此次操作恢复既有算法，统一源码位置与接入边界。后续算法修改以状态、观测域、估计或表面解释的变化为单位，并用实际输出验证。

## 7. 终结、序列化与递归接续

在线变化检测在每个独立片段上完成；其后按物理 ID 归并逻辑实体。终结最后一次归并后重新处理新摄取片段并刷新射线索引，然后执行会话末表面更新，最终时间与最后 ACK 输入一致。

恢复版持久包包含：

| 文件 | 保存的内容 | 下一会话用途 |
|---|---|---|
| `final.4dmap.zpk` | 本 session 的时间线及更新后最终图 | 评估、查看、旧格式恢复入口 |
| `chain_state.4dmap.zpk` | 表面更新前的最终推理 DSG | 下一会话在线变化推理的 seed |
| `shown_state.4dmap.zpk` | 更新后最终表面的一份快照 | 下一会话表面更新中的旧记忆 |
| `depth_scales.txt` | 历次深度尺度 | 式 (11) 的历史误差窗 |
| `sensor_statistics.txt` | 缺失证据的总体统计 | 式 (7) 的校准 |
| `transition_manifest.json`、`state_summary.json`、`control/` | 数据、版本、配置、帧账、退出状态、地图摘要 | 追溯和验收 |

加载时优先读取 chain，旧格式直接读取 final；把最新推理 DSG 作为一个初始快照，随后追加本会话状态。已有各 session 完整历史留在各自地图文件中。运行时 registry 从当前 OBJECTS 重建；其闭合片段和未决候选目前保存在本进程运行状态中。当前包的恢复范围是已物化 current、地图时间线和上述统计。

当前恢复版直接使用原有 map 与 sidecar，`session_surface.bin` 未加入此状态协议。D3 通过加载与接入现有变化检测实现，下一会话按同一入口继续。

## 8. 输入、构建与运行契约

每个 session 提供原始 RGB-D、相机内参、`world_T_camera`、语义图和稳定实例图。可选 `T_session_to_world` 左乘位姿。原始 Z 深度单位米。图像尺寸、帧表、时间和 ID 均在启动时检查。

```text
rgbd/session_n/{timestamps.csv,Intrinsics.txt,<id>_color.png,<id>_depth.tiff,<id>_pose.txt}
semantics/session_n/<id>_segmentation.png
instance_labels/session_n/<id>_segmentation.png
instance_labels/session_n/manifest.json
alignment/session_n_to_world.txt
```

标签协议为 `packed_32SC1 = (semantic_id << 16) | physical_instance_id`，由 `input_labels_are_packed` 显式启用。物理 ID 0 表示背景。canonical 与历史实例后缀各自明确，同帧同义文件出现歧义时由预检报告。

真实 AB 的既定标注规则继续有效：I9 并入 I7、I20 仅 A、A 床面整体归 I1、I12 为背包实例。规则已固化进输入实例图及 manifest。原始 Synthetic AB 保持 30 Hz、680×480、A 4186/B 3555 帧；相机位姿和 semantic/instance 使用既定 GT 输入，GT 物体网格只参与评估。

标准构建与检查：

```bash
cd /home/jixian/Desktop/FT/session_update_baseline
./scripts/build_canonical.sh
./scripts/check_canonical_runtime.sh --require-built
```

构建验证 `ports/mapping_core`、`session_core` 和运行层源码指纹、安装位置及动态库来源，使用锁避免并发构建。当前 Hydra/PGMO 等研究依赖由 `/home/jixian/ros2_ws/install/setup.bash` 提供；Khronos 链接到本项目 canonical install。已有构建按脚本支持的 `--incremental` 方式更新。

生产入口为 `scripts/run_session.sh`，示例中的数据路径按实际数据填写：

```bash
./scripts/run_session.sh \
  --run-dir /path/to/session/rgbd \
  --semantic-dir /path/to/session/semantics \
  --instance-dir /path/to/session/instances \
  --output-state /path/to/new_state
# 后续 session 同一命令增加 --input-state /path/to/previous_state
# 不同坐标系增加 --world-transform /path/to/session_to_world.txt
```

每次运行启动一个 mapper，输出到新的状态目录。每帧通过 ACK 确认，终帧、保存时间和进程退出码对应；继承运行核对 output.initial 与 input 的推理 current。pure-B 使用独立标记的消融运行；生产更新只消费一份前序状态。C 直接消费 B。

唯一地图查看入口为 `scripts/view_ab_chain_4dmap.py`，配套原生 `4dmap_mesh_server`。压缩地图按时间片读取，保留完整时刻列表。datasets、权重、build/install、大地图与运行产物放在 Git 外，Git 保存源码、配置、许可证和紧凑记录。

## 9. 本轮验证与后续质量判据

2026-09-30 已恢复项目 TODO 函数，完整编译 Khronos、ROS、`session_khronos_node` 和状态检查工具；Khronos 现有 17 项测试通过。生产入口连续处理 A 120 帧，再在新进程加载 A 处理 B 120 帧；两段逐帧 ACK 完整，B 初始推理状态与 A 一致，两次表面更新均 applied=1、保存退出码均为 0。最终压缩地图 A 264,061 字节、B 805,816 字节。实际短程运行、地图尺寸与帧账详见 `reports/restored_runtime_smoke_20260930.json`，产物位于 `/home/jixian/Desktop/FT/runs/restored_core_smoke_20260930/`。

本轮任务是源码归属审核、算法恢复和正常运行验证。地图质量验收继续使用真实在线 A→B→C 的物体/几何、D1/D2/D3、残影、中间状态与最终图指标，并与纯 B、有效历史版本逐项比较。离线快速评估用于定位原因，报告单独标注输入来源。算法改变先落实数学含义，再实现和运行；场景名、特定物体编号只用于数据或评估记录。

旧 V6 的全新在线执行器、重写 Khronos 融合和 D1/D2、同时维护两套参考档案等任务已移出活动设计。fin4 原文及其表面模型的比较放入归档审查记录。当前唯一活动说明采用上面的 Khronos 复用边界与恢复版公式。

历史设计归档于 `/home/jixian/Desktop/FT/archive/design_models_20260930/`；旧 README 和恢复前 TODO 保存在本仓库 `archive/design_consolidation_20260930/`。Windows 的同批设计稿归档于 `D:/3Study/ETH/FT/archive/design_models_20260930/`。V3 等原文均保留。

合成地图清理沿用既定规则：新一轮完整运行、评估并更新 CURRENT_VIEW 后，核对旧产物已被替代且无进程使用，再记录路径/尺寸并清理。保留原始输入、USD/GT、真实 ABC 地图、当前可查看结果、日志和指标。

## 10. 来源与许可

Khronos：MIT-SPARK，BSD-3-Clause，`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`。
Panoptic Mapping：ETHZ ASL，BSD-3-Clause，`3926396d92f6e3255748ced61f5519c9b102570f`。
移植文件保留原许可证和文件头；原 namespace 用于来源追溯。仓库：`https://github.com/giltchcity/session_update_baseline_project`；论文：`https://github.com/giltchcity/myncv---rpsl---daicma`。
