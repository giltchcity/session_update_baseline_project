# Session Update Baseline：跨会话动态建图

本项目在 Khronos 上实现持续的 metric-semantic 4D mapping。Khronos 提供单次 session 的输入处理、跟踪、TSDF、网格生成和 D1/D2；项目扩展维护稳定物理身份、状态有效期、观测证据、旧表面更新及跨进程接续。数学统一的对象是这些扩展共享的状态与证据，工程实现直接复用已有建图基础。

活动仓库：`/home/jixian/Desktop/FT/session_update_baseline_project`。源码位于 `session_update_baseline/`；`/home/jixian/Desktop/FT/session_update_baseline` 是该目录的符号链接。当前分支 `dev`，固定行为参照为 `48a3033b7ef6203f84ef68b7b2a8c97a3eaa8799`。本文定义正在演进的项目算法、推导和实际接入边界。

**唯一设计稿固定为本仓库根目录 `README.md`：`/home/jixian/Desktop/FT/session_update_baseline_project/README.md`。** 完整数学原理、推导、算法定义和实现依据集中在本文件，并随算法持续更新。

- 每次算法修改先更新这里的模型、推导与编号公式，再同步代码及“公式 → 源码行”映射；文档和实现作为同一修订审查、评测和提交。
- 本文持续回答原先有效算法的每种关键行为如何由共同原理产生，逐项覆盖身份、时序状态、观测归属、几何更新和跨会话接续。`48a3033` 为固定对比基准，`8e5f3d3/cb71761` 的有效行为与已有成绩保留为只读参照；各版本输入和测量口径分别核对。
- 设计完成由同版本代码的独立审查、行为对照及真实指标共同验收：完整重现好版本的有效行为，并达到既定效果要求。出现偏差时在本文件修正原理、同步实现，重走审查和评测。
- 始终在本文件原位演进。Git 保存修订历史；既有旧稿留在 archive。JSON/CSV/TXT 只记录映射、问题、审查和测量证据；审查者与执行者统一读取本文作为算法定义。

本文件同时是唯一活动 Markdown。代码、构建、运行与输出统一留在 WSL。

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

**rewrite 起点**为提交 `1d6ebc30afdb08f89279a0ccb6ad5f0b172b019e`：其中项目函数体已集中。本文和工作区代码从该检查点共同演进，Khronos 基础按上述边界复用。

上游比对使用官方 Git 对象 `MIT-SPARK/Khronos@63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`，直接读取该提交内容。最初 64 项 TODO 清理涉及其中 20 个原生函数，已恢复其实现；项目 TODO 的七个源文件/头文件也已从 `48a3033` 恢复，并保留新目录布局。精确清单见 `reports/khronos_upstream_boundary_20260930.json` 和 `reports/todo_restore_20260930.json`。

## 3. 身份、状态与时间

物理身份 \(\ell\) 在会话间保持稳定。Khronos 输出的可见片段是这个身份的观测，项目层将观测归入时序状态。一个状态拥有自己的几何、观测时间和证据；几何合并发生在同一状态内。每个实体的状态为

\[
X_\ell=(\mathcal F_\ell,c_\ell,n_\ell,b_\ell,\mathcal A_\ell,E_\ell).
\tag{3}
\]

\(\mathcal F\) 保存时序片段，\(c\) 指向当前片段，\(n\) 保存各自带时间和几何的待归属观测集合，\(b\) 独立处理本会话观测，\(\mathcal A\) 记录已摄取的提取来源，\(E\) 记录当前片段的最近证据。载入的旧片段与本会话重建分别接受测量；普通状态和 \(b\) 使用同一个局部转移函数。

### 3.1 片段的合并与关闭

同状态的观测组成一个带时间的几何集合。设两个片段为 \(f,g\)，共同世界坐标系中的合并运算 \(\oplus\) 为

\[
\begin{aligned}
G_{f\oplus g}&=G_f\cup G_g,&
 B_{f\oplus g}&=\operatorname{bbox}(G_f\cup G_g),\\
b_{f\oplus g}&=\min(b_f,b_g),&
 t^{\rm track}_{f\oplus g}&=\min(t^{\rm track}_f,t^{\rm track}_g),\\
s_{f\oplus g}&=\max(s_f,s_g),&
 v_{f\oplus g}&=\max(v_f,v_g),\\
n_{f\oplus g}&=n_f+n_g,&
 d_f&=\max(t_{\rm decision},s_f).
\end{aligned}
\tag{4}
\]

\(b,s,v\) 分别是首次观测、最近观测支持和本会话传感器确认时间，\(n\) 是重建帧计数。所有顶点先按各自包围盒转换到世界坐标，再转换到共同包围盒坐标。式 (4) 的几何集合在当前实现中以带来源的三角面序列表示；重复输入由提取来源账本排除。重叠帧区间可以产生不同的几何估计，各自保留；重建帧计数记录几何输入的使用量，置信度由独立传感器证据计算。关闭时间 \(d\) 给出离开时间的上界。传感器确认取实际支持时间与本轮处理时间的较小值，保持证据时间与执行时间各自的含义。


提取事件具有独立来源 \(a=(\nu,g)\)：\(\nu\) 是提取池生命周期内固定的 128 位来源标识，\(g\) 是成功接受请求的单调编号。请求处理水位只排序事件，几何的出生和支持来自实际输入帧。并行提取发布已完成的连续请求前缀；空结果占据完成槽，失败沿同一前缀传播。registry 在本轮物理测量前摄取原始事件，再物化输出几何。来源账本按物理身份保存并跨会话恢复：

\[
 \mathcal A_\ell^+=\mathcal A_\ell\cup\{a\},\qquad
 T_a=\begin{cases}T(\cdot,Y_a),&a\notin\mathcal A_\ell,\\
 \mathrm{identity},&a\in\mathcal A_\ell.\end{cases}
 \tag{4a}
\]

同一来源重复交付沿用原结果，不同来源即使帧区间重叠也各自表达一份几何重建。几何使用计数保存来源信息；式 (7) 的观测置信度按真实传感器来源计算。旧原始地图缺少请求标识时，来源由地图节点身份及输入时间区间构成。已物化 current 带显式标志，作为状态表达读取。旧地图先按 Khronos 的存在区间判定边界时刻是否在场，实际支持从观测字段读取；已离场几何进入闭合历史，原始重复来源只摄取一次。导入按物理身份收集所有在场候选，式 (5b) 选择实际支持最新的 current，其余几何保留为独立待归属片段；节点遍历顺序只承担相同支持时间下的稳定次序。


已配置的检测节奏给出决策事件序列 \(t_1,t_2,\ldots\)。项目后端释放图更新锁后，等待本次请求的裁决与发布完成，再推进下一后端包；事件保持原有帧序和检测间隔。每个状态转移按此序列折叠，生产入口将 `active_window.detach_object_extraction` 固定为 `false`，每个输出包在提交该包前完成对应提取前缀；线程执行时长只影响耗时：

\[
 P_{j+1}=T(P_j,Y_{\le t_j},G_{t_j}),\qquad
 \operatorname{publish}(P_{j+1},t_j)\prec\operatorname{advanceBackend}(j+1).
 \tag{4b}
\]

原生变化检测和融合继续处理既定输入。等待位于后端图锁之外，工作线程可取得该锁并执行裁决；项目调用点维持至多一个未完成的决策请求，原有工作器的合并分支在该路径上保持空闲。

### 3.2 同一个关系判定与状态转移

对当前片段 \(c\)，本轮支持票为 \(S\)，反证票为 \(C\)，第 4 节输出可信反证门 \(g\)。这里的 $S,C$ 分别计数当前查询窗口内实际 `(frame,pixel)` 支持来源集合和反证来源集合；同一来源支持一处、穿透另一处时可同时进入两集合，在损失差中各贡献一次。第 4 节的预测门则用每个可靠格的最新事件按来源归组，组内取反证逻辑与。前者描述窗口内的直接几何支持，后者提供状态退出的可靠性授权。令 \(C^*=gC\)。把“保留”和“退出”视为两个解释，逐票不一致损失分别为 \(C^*\) 与 \(S\)：支持票反对退出，反证票反对保留。独占关系 \(X\) 表示当前片段和已建立的新位置属于两个状态；同状态关系 \(L\) 表示两份几何可补全同一状态。于是

\[
\begin{aligned}
J&=[C^*>S]\lor X,\\
\rho(c,q,E)&=
\begin{cases}
\mathrm{replace},&J,\\
\mathrm{refine},&\neg J\land L,\\
\mathrm{separate},&\text{其余},
\end{cases}\\
T(c,q,E)&=
\begin{cases}
(\operatorname{close}(c),\operatorname{promote}(q)),&\rho=\mathrm{replace},\\
c\oplus q,&\rho=\mathrm{refine}\land A,\\
c,&\text{其余}.
\end{cases}
\end{aligned}
\tag{5}
\]

平票沿用当前状态；独占约束要求候选的最新实际支持晚于 current 的最新实际支持，并晚于闭合状态的支持水位。关闭与晋升共同使用这份时序资格。\(A\) 是观测加入当前片段的授权：局部待归属片段要求真实支持 \(S>0\) 且支持测量的时间覆盖候选片段出生时间；继承片段与本会话重建的同状态关系用于在线物化，在会话终结时提交合并。在线物化和终结共享 \(\rho\)，普通局部状态和本会话子状态共享 \(T\)。

与 `48a3033` 的行为对应集中在关系的测量定义中：

| 输入关系 | 当前定义 | 消费位置 |
|---|---|---|
| 局部退出 | 当前片段实测反证通过第 4 节覆盖判断，且 \(C^*>S\) | 同一局部转移 |
| 继承 \(X\) | 时间合格且具有运动先验的新片段，在其可靠采样域上由式 (5e) 选择继任解释 | 同一继承关系 |
| \(L\) | 静态先验成立，或式 (5e) 的同一表面查询找到相符样本 | 局部吸收、在线物化、终结合并 |
| 直接运动 | Khronos 已报告可见运动时关闭原静态片段；有新静态几何时开启继任片段，纯轨迹事件保留轨迹 | 观测摄取 |
| 观测补全 | 同会话共享表面，且观测区间内旧片段仍有支持 | 观测摄取 |

运动先验来自既定语义配置、直接运动历史或已关闭的状态。几何关联采用同一世界表面距离 \(d(x,G)=\min_{y\in G}\|x-y\|\)；三角形按完整面求最近点，点云使用点集。采样候选为三角面心或纯点云的输入点，每个采样格保留距格心最近的候选，等距依次按坐标、有效法向优先、法向字典序取值，重复面和顶点沿用同一格的代表。局部关联在地图分辨率取样；跨会话继任比较直接读取第 4 节已取得可靠证据的样本集 \(\mathcal R_q\)，保持样本数、分子和分母属于同一域。

把当前几何的容差管 \(G_c^{\delta}=\{x:d(x,G_c)\le\delta\}\) 作为该几何状态的空间表示。均匀抽取一个可靠样本，同状态解释在管外支付单位错误，继任解释在管内支付单位错误；最小经验风险由两种错误数决定：

\[
 a_c(x)=[d(x,G_c)\le\delta],\qquad
 J_{\rm continue}=\sum_{x\in\mathcal R_q}(1-a_c(x)),\quad
 J_{\rm successor}=\sum_{x\in\mathcal R_q}a_c(x).
 \tag{5e}
\]

固定基线协议取 \(\delta=0.10\,\mathrm m\)，以 30 个可靠空间样本作为继任比较的建立预算。运动先验和时间资格限定可选择继任的域；域内取损失较小者，平票延续。共享表面授权读取同一个 \(a_c\) 的正支持。该估计的对象是地图所表达的几何状态；旧表面直接实测退出仍由式 (5) 的观测损失决定。样本可靠性由实际身份观测建立，点到三角面的距离使旧大面内部的新重建点保持几何一致。

### 3.3 交接的守恒与证据归属

一次交接采用同一个关系 \(\rho\)。`replace` 关闭旧 current 并接入本会话 current；`refine` 按式 (4) 合并；`separate` 保存独立片段。三条路径共同转移本会话已经关闭的历史，以及待归属观测集合。分离关系保持观测的未决身份；物化新的 current 时优先选择实际支持时间最新的观测，其余观测保留各自来源。关闭片段均满足式 (4) 的时间界。由此

\[
\mathcal F^{+}=\mathcal F_{\rm old}\mathbin{\uplus}
\mathcal F_{\rm session,closed}\mathbin{\uplus}
\operatorname{settle}_{\rho}(c_{\rm session},n_{\rm session}),
\quad
E(f)\text{ 仅用于 }f.
\tag{5a}
\]

\(\uplus\) 表示保留片段身份的历史拼接；同状态合并的片段由式 (4) 保留它的几何和时间。本轮测量后若本会话 current 已更换，继任片段的可靠样本在下次实测时建立。历史转移保持已观察到的运动属性，后续背景副本检查可访问这些闭合片段。

待归属观测按独立片段保存，使用同一合并算子时才构造几何并集。令 \(\mathcal N_f\) 为待归属观测集合；已有 current 的吸收授权要求实际支持时间落在该观测区间内，并满足同状态关系 \(L\)。空 current 从实际支持时间晚于已关闭前任最后支持时间的观测中选择最新者开启继任片段，其余观测继续待决：

\[
 A(c,q,t)=[b_q\le t\le s_q]\land L(c,q),\quad A_{\rm ingest}(c,q)=\bigvee_{t\in\{s_c,v_c\}}A(c,q,t),\qquad
 \bar s_f=\max(s_f,v_f),\quad q^*=\arg\max_{q\in\mathcal N_f:\bar s_q>w_{\rm closed}}(\bar s_q,b_q),\qquad
 \mathcal N_f^+=\mathcal N_f\setminus\{q^*\}.
 \tag{5b}
\]

直接摄取与待决吸收共用同一个 $A$：摄取时分别检查已有直接支持 $s_c$ 与确认支持 $v_c$，其中任一落入候选区间即可授权；其余候选进入待决集合，之后由真实支持时间调用同一条件。两个时钟分别参与区间成员判断，保留较早但仍落在候选区间内的有效支持。

已关闭状态的最近实际支持形成单调继任水位 \(w_{\rm closed}=\max_{f\,\rm closed}\bar s_f\)。较早的未决观测保留其原时间，后续晋升遵守该水位。直接运动事件给此前静态状态提供离开边界；其待决几何保留在闭合历史中，新的静态观测单独开启继任状态。

D1 的输入是原生提取器已接受的原始运动输出。原生纯动态输出通过非空轨迹时间/位置对表达接受结果，静态片段的动态历史标志提供已有运动的先验。项目层读取该输出中实际保留的轨迹时间，并为每个实体保存已消费的运动时间水位 \(m_\ell\)。历史运动标志提供先验，实际事件只在新的已接受轨迹时间到达时消费；若它晚于 current 的最后实际支持，结束此前静态状态。

\[
 t_m=\max\{t\in\text{accepted native trajectory}\},\quad
 m_\ell^+=\max(m_\ell,t_m),\quad
 \operatorname{depart}(c)=[t_m>m_\ell]\land[t_m>\max(s_c,v_c)].
 \tag{5d}
\]

该步骤只传递已由 Khronos 接受的运动结果。没有新轨迹时间的历史标志保持摘要作用；运动之后的静态观测按实际时间进入继任状态。纯轨迹输出同样经过式 (5d)，空静态网格保持为空。

原生 presence 的物化接口保留 Khronos 的半开区间语义 $[a,d)$。同身份的原始片段先按直接观察起点、终点和节点 ID 排序；直接观察终点最新的片段 $r$ 提供原始几何和最终 presence 右界 $R$。旧片段的开放延续只覆盖到下一直接片段出现为止，再以最新右界统一截断：

\[
 \widetilde I_i=\{[a,\min(d,R,b_{i+1})):[a,d)\in I_i\},\quad i\ne r,
 \qquad I_{\rm native}=\operatorname{right}_R\!\left(\operatorname{mergeIntervals}(\{\widetilde I_i\}_i)\right),
 \quad \widetilde I_r=I_r.
 \tag{5f}
\]

缺少后继或后继起点为原格式未知零值时，省略 $b_{i+1}$ 项。区间首先作为带端点的记录处理：丢弃倒置记录 $a>d$，保留退化记录 $a=d$ 作为最新右界的锚点。`mergeIntervals` 合并相交或相邻记录并保留有限间隙，$\operatorname{right}_R$ 删除起点晚于 $R$ 的区间并把末段终点设为 $R$。完成右界归约后再把记录解释为半开存在区间，退化记录贡献空集合。这个输出适配算子让最新的原生存在性裁决拥有最终右界，同时保留较早片段已确认的有限间隙。项目 registry 按式 (4)-(5b) 提供 current 几何、状态标记和观测界；轨迹历史独立保留。原始最新片段为纯轨迹时，其静态几何保持空集合。

源码：`session_core/src/state/persistent_object_state.cpp`、`src/adapters/physical_object_merge.cpp`、`src/adapters/backend_session.cpp`。式 (5) 的关系定义、转移、物化及终结通过同一实现维护。

## 4. 同一测量的几何含义与物理证据

相机中心为 \(c_i\)，世界射线方向为 \(d_i(u)\)，径向测距为 \(r_i(u)\)。测量端点为 \(p_i(u)=c_i+r_i(u)d_i(u)\)。查询旧点 \(x\) 时

\[
q_i(x)=\|x-c_i\|,\qquad e_i(x)=r_i(\pi_i(x))-q_i(x).
\tag{6}
\]

光轴 Z 深度由相机模型转换到径向距离后比较。世界变换统一左乘；几何和测量使用同一坐标系。点证据使用式 (6)，体素积分使用其投影像素方向上的 projective 残差。

证据库与融合归档共用径向测距编码：量化单位 \(u=10^{-3}\mathrm m\)，有效测距同时满足传感器量程和 \(0<r\le65535u\)，编码为 \(\lfloor r/u+1/2\rfloor\)，零码表示无有效测距。范围之外的回波保持无效状态。量化误差界为 \(u/2\)，两个独立量化距离比较的误差界为 \(u\)。物理身份使用 16 位无符号值，0 为背景/未识别；超过该协议范围的正身份由入口报错。 同一会话的时间戳指定唯一不可变观测：相同时间、相机、位姿和编码测量的重复提交保持幂等；内容冲突在入口报错。证据库与帧归档遵守此来源合同，帧归档发布时按时间严格排序。针孔相机、有限有效参数及 32 位像素编号域由入口校验；所需帧格式错误立即使运行失败。

\[
 \widehat r=u\left\lfloor r/u+\tfrac12\right\rfloor,
 \qquad |\widehat r-r|\le u/2.
 \tag{6a}
\]

| 实测关系 | 物理实体证据 |
|---|---|
| 视野外或无有效深度 | unobserved |
| 测量在旧点前方 | occluded |
| 深度吻合且相同物理 ID | supported |
| 深度吻合且其他 ID / 背景 | replaced by other / background |
| 射线穿过旧点到达更远表面 | free space |

几何存在性允许任意身份的同位置测量提供支持；实体存在性同时检查物理身份。查询使用状态最后支持之后的实际测量；遮挡与视野外各自保留为观察条件。会话处理水位控制输入消费，实际支持时间控制证据窗口；跨会话保存的逐点证据仍按同一支持时间筛选。


点证据的对象是物理实体在查询位置的存在。匹配公差 $\tau$ 给出同身份的重建一致域；不同身份的前方回波只能说明当前射线被遮挡。编码回波对应区间 $I(\widehat r)=[\widehat r-u/2,\widehat r+u/2]$，查询 $q$ 是当前重建中的连续距离。因此“回波整个区间位于查询之前”的充分条件是 $\widehat r+u/2<q$，这里只使用回波的一份量化误差。对有效且已识别的端点，依次作如下共同分类：

\[
 V(e,L;\ell,\tau)=
 \begin{cases}
 O,&e<-\tau,\\
 F,&e>\tau,\\
 S,&|e|\le\tau\ \land\ L=\ell,\\
 O,&|e|\le\tau\ \land\ L\ne\ell\ \land\ e<-u/2,\\
 X_L,&|e|\le\tau\ \land\ L\ne\ell\ \land\ e\ge-u/2.
 \end{cases}
 \tag{6e}
\]

$S$ 为相同实体的支持，$F$ 为穿透，$O$ 为遮挡；$X_L$ 在已知其他身份或背景时提供身份替换证据，身份未识别时保持未知。视野外、无效测距独立保留其观测状态。相同 ID 的匹配回波直接支持该物理实体；其他 ID 的回波通过区间前后关系决定这条射线是否到达查询位置。状态点查询、表面采样和静态帧相容比较复用该分类及各自已有的匹配公差，量化单位统一来自式 (6a)。


每个决策事件在开始检测前固定输入时间界 $t$，其物理证据快照为

\[
 E_{\le t}=\{E_i:t_i\le t\},\qquad
 \operatorname{query}(a,b;t)=\{E_i:a\le t_i\le\min(b,t)\}.
 \tag{6d}
\]

快照同时固定存储版本和时间上界。枚举、单帧投影与稠密读取共用此上界，活动窗口提前处理的后续输入留给对应的未来决策。该时间界通过薄接口传给项目物理证据适配，原生 D1/D2 判定直接消费受限快照。

### 4.0 静态重建与状态裁决共用端点证据

一次静态提取只融合最新相容片段。比较旧帧 \(a\) 与最后帧 \(b\) 时，把同一 track 的表面点投影到对方图像，式 (6) 和 (6a) 同时给出两方向的端点类别。同身份相合为支持，穿透及同深度的其他身份/背景为反证；前方遮挡、无效值和缺少观测保持未知。每个目标实际像素是一个来源，键为 \(\mathrm{source}=(t_{\rm target},u,v)\)。两方向目标时间不同，来源集合合并后只作一次比较；给定共同失配率 theta，合并后的来源组使用式 (7c) 的条件独立模型。这里用联合预测取代旧实现的双向分别投票后取 OR。

同源多个查询取反证的逻辑与；任一支持或未知使该组保留。含有效分类的组计入 \(n\)，纯未知来源组及无目标来源的查询计入 \(U\)。式 (7) 给出最保留的完整计数，式 (7c) 使用固定均匀参考 \(H=\operatorname{Beta}(1,1)\)。静态相容域边界采用既定 \(\alpha=0.2\)，至少 20 个已分类来源构成一次有资格的比较，退出损失沿用式 (7d)。设按时间排序的候选帧为 \(F_0,\ldots,F_m\)，则

\[
 j^*=\max\{j<m:n(F_j,F_m)\ge20,\ \Lambda(F_j,F_m)>\log99\},\qquad
 \mathcal F_{\rm static}=\{F_{j^*+1},\ldots,F_m\}.
 \tag{6b}
\]

集合为空时使用全部候选。数据获取保留基线的源点预算：源 cluster 像素按步长 \(\max(1,\lceil N/512\rceil)\) 选候选，世界变换和目标投影沿用传感器模型；公差为既定 5 cm。观测单位就是投影像素的实际回波，图像边界有效像素遵守同一分类。20 为预先声明的观测资格预算，\(\alpha\) 为相容误差域；两者分别控制估计何时可用以及何种失配属于静态模型。旧配置 \(\alpha=1\) 表示关闭静态分段；\(\alpha=0\) 对应在位模型集中于零失配，完整反证数为正时退出赔率为正无穷。

原有提取适配沿用配置中的最小可见位移 $d_{\min}$。从已缓存动态 cluster 的有效质心序列 $c_0,\ldots,c_m$ 定义输入位移；缺少质心时位移为零：

\[
 d_{\rm input}=\max_j\|c_j-c_0\|,\qquad
 M_{\rm input}=(\operatorname{dynamic}\lor\operatorname{motionHistory})
                \land[d_{\rm input}\ge d_{\min}].
 \tag{6c}
\]

该量是既有 Khronos 提取链的运动资格测量。带物理身份的 track 低于该资格时，在提取副本中清空运动标志与运动时间，交给静态重建；原 track 保持原输入。静态重建继续由式 (6b) 检查 RGB-D 相容性。资格阈值沿用原配置，原生位移测量与 D1/D2 实现复用现有代码；项目状态消费的是提取链最终接受的运动输出。

### 4.1 状态缺失是有限表面上的预测比较

将物体表面按半个测距匹配公差的空间格取样，预算为 1500。候选为三角面心，纯点云使用输入点；每格取距格心最近的候选，等距依次按坐标、有效法向优先、法向字典序选择，再按格顺序等距取样。新表面取得三次同身份支持后建立可靠性，保存第三次支持时间；继承表面由 registry 的来源标志建立可靠性。斜视穿透按固定入射角协议过滤，角度配置的定义域为有限的 0–90 度。

每格保存最新支持与反证的时间及实际来源 \(a=(t,\mathrm{pixel})\)。在查询窗口内取最新有效事件，同源格组成一组。组事件为成员反证的逻辑与：一份实际像素测量至多贡献一份预测证据，组内支持使该组取保留值。设已观测来源组数为 \(n\)，其中全反证组数为 \(k\)，可靠但窗口内缺少有效测量的格数为 \(U\)。未知完成域允许每个未知格并入一个已有组或形成新组，已有来源组保持独立身份。最保留的完成先逐个消去反证组，再把剩余未知作为独立保留组：

\[
 z_a=\bigwedge_{x:\,a(x)=a}[x\text{ 的最新证据为反证}],\qquad
 k=\sum_a z_a,\quad n=|\{a\}|,\qquad
 K=\max(k-U,0),\quad N=n+\max(U-k,0).
 \tag{7}
\]

对式 (7c) 的有序参数域，添加保留观测使似然乘 \(1-\theta\)，因此降低赔率。将“未知独立补零”改为“未知并入已有反证组并将其翻零”，两种完整似然之比为 \(1/\theta\)，也降低赔率。因此先翻转 \(\min(U,k)\) 个反证组、再给剩余未知各加一个保留组，实现整个完成族的最低预测赔率。式 (7) 与 (7c) 合起来是完成族上的稳健评分；每一种完整数据的预测分别归一化。校准与评分共用同一来源归约；校准用已观测组的 \(k,n\)，评分对未知来源完成取下界 \(K,N\)。

校准帧由原始几何片段的直接支持区间 \([b_f,s_f]\) 授权；评分窗口取实际最新支持 \(\max(s_f,v_f)\) 之后。校准在帧开始前固定可靠集合：新格要求可靠性建立时间严格早于该帧。存在新增有效测量时，按该时间戳的最新来源 mosaic 提取已观测来源组，每个时间戳一次；每帧提前计算候选校准统计，迟到的直接支持水位消费待授权队列前缀。校准和评分各保存自己的 mosaic；已授权参考只吸收授权区间内的候选。

工作预测模型假设：给定查询几何、可靠集合、来源分组和窗口设计，给定参数 \(\theta\) 与该设计后，来源组事件独立同分布为 \(\operatorname{Bernoulli}(\theta)\)；校准与评分共享这种条件分布，缺失在给定观测设计下可忽略。校准将未知组按缺失边缘化，使用实际已观测组的似然；评分才对未知来源取完成族下界。\(\theta\) 是组事件为反证的概率。校准快照给出无截断参考分布 \(Q_j=\operatorname{Beta}(k_j+1,n_j-k_j+1)\)，以时间等权混合保留变化，再匹配一、二阶矩形成参考分布 \(H\)。在位预测是该参考分布条件于 \(\theta\le1/2\)，离位预测取 \(\theta>1/2\) 的均匀分布：

\[
 m_j=\frac{k_j+1}{n_j+2},\qquad
 q_j=\frac{(k_j+1)(k_j+2)}{(n_j+2)(n_j+3)},\quad
 m=\frac{\sum_jm_j}{J},\quad v=\frac{\sum_jq_j}{J}-m^2,
 \quad H=\operatorname{Beta}(m\kappa,(1-m)\kappa),\quad
 \kappa=\frac{m(1-m)}{v}-1.
 \tag{7b}
\]

混合二阶矩同时包含视角差异和有限样本的测量不确定性；每个 \(Q_j\) 的方差严格为正且低于伯努利方差，因此式 (7b) 的参数为正。设取样预算为 \(M=1500\)，每个成分的浓度最多为 \(M+2\)。由全方差公式，混合方差满足 \(v\ge m(1-m)/(M+3)\)。求形状参数前将数值矩投影到这个有限测量精度界，避免旧文件或浮点相消赋予超出观测预算的精度。局部模型汇总当前状态的授权校准快照；总体模型给每个已校准状态相同权重。取样几何或出生时间改变时，当前会话归档帧对新几何重新测量，重建该状态的统计并替换其原有总体贡献；其余轮次仅处理新增帧。同一轮所有物体裁决共用轮开始时冻结的总体模型，当前轮校准更新供下一轮使用。两者齐备时，以相同先验概率选择局部或总体模型并混合预测概率；只有总体数据时直接使用总体模型；初始模型为 \(\operatorname{Beta}(1,1)\)。这使首次观测和继承状态使用同一预测接口。

对任一参考模型 \(H\) 和已指定的相容域边界 \(0<\alpha<1\)，公共预测函数为

\[
 P_0(K\mid N,H,\alpha)=
 \frac{\int_0^{\alpha}\binom NK\theta^K(1-\theta)^{N-K}H(\theta)\,d\theta}
 {\int_0^{\alpha}H(\theta)\,d\theta},\qquad
 P_1(K\mid N,\alpha)=\frac1{1-\alpha}\int_{\alpha}^{1}\binom NK\theta^K(1-\theta)^{N-K}\,d\theta,
 \quad \Lambda=\log\frac{P_1}{P_0}.
 \tag{7c}
\]

registry 的状态退出域固定为 \(\alpha=1/2\)。条件分布归一化、来源完成和损失比较集中在同一个公共实现。正常协议的计数预算分别为 registry 的 1500 和静态比较的 1024，直接复用 Boost 的不完全 Beta 计算及标准 `lgamma`；可配置边界使尾概率超出浮点表示域时，用同一积分的连分式对数表达式求值。

两个假设的参数域按 \(\theta\) 排序，预测比随固定 \(N\) 下的 \(K\) 单调增加。式 (7) 定义未知表面来源完成域。原有阈值 99 写为决策损失比：错误退出损失为 99，错误保留损失为 1，假设先验相同，因此

\[
 g=[\Lambda>\log99],\qquad
 R(\mathrm{exit}\mid E_c)=99\Pr(H_0\mid E_c),\quad
 R(\mathrm{retain}\mid E_c)=\Pr(H_1\mid E_c).
 \tag{7d}
\]

其中 \(E_c\) 为一个完整来源完成；门控要求所有允许完成均偏好退出，因此用式 (7) 的最低赔率作比较。式 (7d) 将可信反证交给式 (5)。每次查询重新由空间证据计算 \(\Lambda\)，省去累积量、重复计数标记和阈值后的重置分支。校准在处理直接支持区间的真实帧时更新，判定域由实际支持时间限定，两者分别维护。传感器状态仅保存校准状态数与两个矩和；旧六字段统计先将已保存方差投影到给定均值的可行区间 \([0,m(1-m)]\)，再加入一个单位均匀先验；这给旧统计一个有效且有限的预测分布。旧文件以负方差表示尚无校准，此时使用初始模型。

证据缓存按 `(处理器, 物理身份, 片段不可变证据标识)` 定位。状态槽仅标识当前调用角色，片段在本会话槽与顶层 current 之间交接时沿用同一份样本与校准。几何补全可扩展首次观测时间，证据标识随原片段保持；开启继任片段时分配新标识。状态更换或输入时钟重启时整体初始化该状态的空间样本和局部校准；总体传感器校准独立持久化。帧处理区间为

\[
\mathcal I_f=[\max(t_{\rm processed}+1,b_f),\ t_{\rm latest}],
\qquad E_f^+=\operatorname{Accumulate}(E_f,Y|_{\mathcal I_f}).
\tag{7a}
\]

这使可靠样本、计数和累计量始终属于受测片段，也使出生之前的自由空间记录留在它所描述的历史时段。

每份测量携带片段证据键、几何修订号和实际测量水位。几何合并递增修订号；关系求解只消费归属吻合且测量水位覆盖该状态最新支持的证据：

\[
 \operatorname{Owns}(f,E)=[k_E=k_f]\land[r_E=r_f]
 \land[t_E\ge\max(b_f,w_f,s_f,v_f)],\qquad
 \operatorname{Fresh}(E,t)=[t_E=t].
 \tag{7e}
\]

局部转移要求本轮 Fresh；持久关系缓存保存完整归属，几何修订及状态更换自动使旧测量失效。几何修订号随 registry 保存恢复。


关闭物体对应的背景副本由 `closed_object_background.cpp` 查询实际后续深度：背景表面自身时间参与查询，同位置实测表面继续提供支持，最后几何支持之后的穿透才产生背景变化。静止物体同时参与背景和 private mesh 的既有表示由此协同更新。

## 5. 状态授权的会话末观测域

每个身份的当前静态状态给出授权域 \(\mathcal D_\ell\)：有 current 时为 \([b_\ell,\infty)\)，关闭且尚无静态继任状态、或没有已建立 current 时为空集。归档保留带稳定物理身份像素的原始有效测距，终态授权前保持测量可用；无物理身份的动态或无效语义像素沿用原输入屏蔽。归档中的本身份像素仅在 \(t\in\mathcal D_\ell\) 时进入当前表面。状态起点使用 registry 的 `birth_time`，由实际接纳的静态观察区间建立；`track_first_seen` 另行保存整条跟踪的时间。其他旧像素保留实测端点，自由空间受到当前物体首交 \(\lambda\) 限制。

\[
 A(t,\ell)=[\ell=0]\lor[t\in\mathcal D_\ell],\qquad
 \mathcal D_\ell=\begin{cases}[b_\ell,\infty),&\text{当前静态状态存在},\\
 \varnothing,&\text{当前静态状态关闭}.
 \end{cases}
 \tag{8a}
\]

表面延续由状态身份授权。载入前序 current 后固定其证据键 $k^-_\ell$；该键在同状态精化中保持不变，状态继任取得另一键。终结时读取当前键 $k^+_\ell$，定义

\[
 C_\ell=[c^+_\ell\ne\varnothing]\land[k^+_\ell=k^-_\ell],\qquad
 \operatorname{historyEligible}(e)=[\ell_e=0]\lor C_{\ell_e}.
 \tag{8b}
\]

式 (8a) 按最终 current 的实际出生时间授权本次测量；式 (8b) 按不可变状态键授权前序表面。出生于前序会话的 pending 在本次晋升时仍属于继任状态，其较早出生时间只决定测量域。加载旧格式时由同一次 registry 初始化建立键并立即固定前序 current 键，随后沿用同一比较。

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

只在八角均已积分的 cube 提取表面。这里的 `PresentTsdf` 是既有会话末模块；在线融合继续调用 Khronos。融合后的表面与物理身份共用已授权测量：令 \(H_i(f)=[|r_i(\pi_i(x_f))-q_i(x_f)|\le\tau(h,q_i)]\)，有效深度的面心支持观测投票给它自己的输入身份。当前注册表提供可物化身份集合 \(\mathcal C\)，\(\pi_{\mathcal C}(\ell)=\ell\) 当身份在集合中，其他端点归未实例化背景；原始身份保留在帧归档中。

\[
 N_f(\ell)=\sum_i H_i(f)[\pi_{\mathcal C}(L_i(\pi_i(x_f)))=\ell],\quad
 \ell_f=\arg\max_\ell(N_f(\ell),t^{\rm last}_f(\ell),-\ell),\quad
 [t_f^-,t_f^+]=[\min,\max]\{t_i:H_i(f),\pi_{\mathcal C}(L_i)=\ell_f\}.
 \tag{9a}
\]

这是对支持测量身份的零一损失最小解。并列票取最近被支持的身份，再按身份数值确定唯一结果。单一面心尚无直接支持时，身份为背景、观测时间为未知零值。相邻物体的距离只影响几何对应；式 (9a) 直接由像素身份决定输出归属。时间属性从归属测量进入面，再以邻接面的有效最早/最近支持时间进入顶点；旧表面携带自己的原属性。

表面公差使用 \(\tau(h,q)=\max(h,\sigma(q))\)，\(h\) 是层体素的一半；\(\sigma\) 由当前表面与测量的距离分箱残差估计。跨视图深度尺度 \(s\) 由残差的稳健中位数拟合，当前测量误差与旧面的来源误差共同定义式 (11) 的对应窗口。

### 5.1 测距校准的目标与数值协议

尺度校准以归档的原始径向测距建立固定跨视角对应；状态授权作用于当前表面残差统计和几何积分。两者的输入域分别对应传感器尺度与当前有效表面的误差。设一个对应为 $(o_a,d_a,r_a,o_b,r_b)$，其中 $d_a$ 是单位世界射线，零尺度投影端点和目标实测距离的差处于背景截断带。共享尺度参数的残差和估计目标为

\[
 e_j(s)=\left|(1+s)r_{b,j}-\|o_{a,j}-o_{b,j}+(1+s)r_{a,j}d_{a,j}\|\right|,
 \qquad J(s)=\operatorname{upperMedian}_j e_j(s),\qquad
 \widehat s=\arg\min_{s\in\mathcal G}J(s).
 \tag{9b}
\]

上中位数对应排序后索引 $\lfloor M/2\rfloor$。数值域沿用本地 `48a3033` 的固定协议：按步长 $\max(1,\lfloor n/64\rfloor)$ 从帧零开始最多取 64 帧；每张图从零起每 16 像素取候选，投影到其余选中视角建立有向对应。少于 1000 对应返回中性尺度零。有足够对应时，先评估零，再依序评估 $0.002k$（$k=-50,\ldots,50$），记最优值为 $s_c$；随后评估 $s_c+0.0002k$（$k=-10,\ldots,10$）。只接受严格下降，等值保留先到的候选。式 (9b) 的集合 $\mathcal G$ 是这两轮实际评估集合，第二轮由第一轮确定。对应在零尺度时固定，搜索过程中保持相同观测集合。

表面残差使用状态授权后的所有帧，投影当前重建顶点，仅计入朝向相机、具有有效实测端点且残差 $e=|r-q|\le T$ 的样本。按测距分箱并统计绝对残差直方图。设箱 $b$ 中总数为 $C_b$，残差格宽为 $\rho$，首个累计数达到 $C_b/2$ 的格为 $j$，格内作均匀插值：

\[
 m_b=\rho\left(j+\frac{C_b/2-\sum_{i<j}H_{b,i}}{H_{b,j}}\right),\quad
 \sigma_b=1.4826m_b,\qquad \tau(h,q)=\max(h,\sigma_{b(q)}).
 \tag{9c}
\]

$1.4826$ 为零中心高斯绝对残差中位数的尺度换算。这里估计的是当前表面匹配残差尺度。既定协议为 16 个测距箱、箱宽 0.5 m、残差格宽 0.0005 m，每箱至少 1000 样本；高测距归入末箱。样本不足的箱从最近的原始合格箱继承，距离并列取较低箱；全体箱都缺少合格样本时，使用零经验残差，公差由半体素 $h$ 决定。直方图计数、测距与坐标须在各自表示域内。

`RangeCalibration` 实现式 (9b)-(9c)；归档遍历负责固定对应与直方图采集，表面判定只调用校准结果。

源码：`session_core/src/surface/session_refusion.cpp`、`present_tsdf.cpp`、`triangle_grid.cpp`、`frame_archive.cpp`、`range_calibration.cpp`。

## 6. 一个表面解释、一个保留判定

会话末重建先给出当前表面 \(P\)。剩余问题是：历史表面提供了哪些当前重建尚未解释的信息。候选元素统一写成

\[
e=(x,\ell,h,T,p,\varepsilon),\qquad
p=\begin{cases}1,&\text{前序会话表面},\\0,&\text{本会话补全面}.\end{cases}
\tag{10}
\]

\(x\) 为面心，\(\ell\) 为物理身份，\(2h,T\) 为它所在表示层的体素和截断距离。历史物体面由当前状态授权：该身份仍为 current，且式 (8b) 的状态身份从前序会话延续。在线补全面来自当前 TSDF 的面心插值格存在未积分角的位置，统一使用本会话重建分辨率。这样同一组测量在已有重建域内产生一个估计，在域外由在线几何提供候选。 历史集合直接来自前序会话的显式表面输入；初始会话使用空集合。候选资格只由当前状态和积分域决定，坐标接近本身保持为几何关系。相同的有向三角面由式 (14a) 的输出商集合存储一次。

### 6.1 测量与重测的对应关系

测距模型的随机误差使用式 (6) 的 \(\tau(h,r)=\max(h,\sigma(r))\)。正深度尺度 \(s_+=\max(s,0)\) 描述径向位置偏差；一个已测表面的来源误差为 \(\varepsilon=s_+q_{\rm support}\)，\(q_{\rm support}=\min_{i:H_i=1}q_i\) 是有效支持视角中的最小测距，选取已有支持提供的最紧径向误差界。当前重建面以面心查询支持；插值面心尚未形成直接支持记录时，来源误差初值为零。历史表面延续时携带自己的 \(\varepsilon\)。旧格式地图以其已保存坐标建立初值 \(\varepsilon=0\)。

在一帧中，令面心投影处的实际测距为 \(r\)，面心距离为 \(q\)。测量落在面心前方时，判断它是否在重新估计同一表面，而真实遮挡另行记录。两份几何的径向误差相加得到对应窗口

\[
\Delta=\tau(h,r)+s_+r+\varepsilon.
\tag{11}
\]

局部表面采用单层分支模型：同身份、同向且处于联合定位误差窗内的可见首交，作为该局部分支的位置重估。每帧从实际中心像素产生单位世界射线 $d_i$，当前全体表面预测的回波为

\[
 t_i^*=\min\{t>0:o_i+t d_i\in P\},\qquad p_i=o_i+t_i^*d_i.
 \tag{11a}
\]

首交之后核对完整观测合同：原始像素身份和式 (9a) 的首交面身份均等于候选身份；旧面与首交面同向、两面朝向相机，$p_i$ 位于旧面朝相机的一侧，$\|p_i-x\|\le\Delta$，且实际回波满足 $|r_i-t_i^*|\le\tau(h_{\rm obj},t_i^*)$。这些条件给出局部单层模型下的重测解释；前景身份不同、方向不同、误差窗外或缺少预测支持时保留遮挡解释。查询只消费全局首交，核验失败即结束对应查询。共享边上等距命中采用面索引顺序确定代表。截断距离仍仅用于式 (8) 的积分域。

候选集合在投票前先按式 (14a) 的精确有向几何键归约。历史与在线补全含有同键面时，已授权历史代表提供唯一的 $p,h,T,\varepsilon$；在线副本表达同一几何候选。每个键只经历一次测量分类和一次解释损失。这样删除、添加或重复存储在线副本都保持该几何假说的裁决一致，历史来源误差随该假说保留。当前重建 $P$ 仍按输出优先级提供本次估计。

每帧按以下顺序给每个候选至多一票：

\[
v_i(e)=
\begin{cases}
R,&r<q-\tau\ \land\operatorname{sameSurface}_i(e),\\
S,&\text{投影足迹有测量端点落入 }B(x,\tau),\\
F,&\text{整个足迹深度有效、均穿过 }x\text{，且满足式 (8) 的自由空间域},\\
O,&r<q-\tau,\\
\varnothing,&\text{其余}.
\end{cases}
\tag{12}
\]

式 (12) 的足迹由实际像素射线与球 $B(x,\tau)$ 求交确定。令相机坐标 $a=(X,Y,Z)$，像素中心 $w=((u-c_x)/f_x,(v-c_y)/f_y,1)$，$d=w/\|w\|$。射线到球心的最近投影及前向交区间为

\[
 t=a\cdot d,\qquad b_\perp^2=\|a-td\|^2,\qquad
 t_\pm=t\pm\sqrt{\tau^2-b_\perp^2},\qquad
 \mathcal P=\{(u,v)\in\mathbb Z^2:b_\perp^2\le\tau^2,\ t_+\ge0\}.
 \tag{12a}
\]

$S$ 要求至少一个有效回波满足 $\|rd-a\|\le\tau$。$F$ 要求 $\mathcal P$ 非空、所有射线均有有效观测、回波越过各自远交点 $r>t_+$，并逐射线满足式 (8)；其中与 `PresentTsdf` 相同的投影距离为 $\widetilde q=Z\|w\|$，$\delta=r-\widetilde q$。局部截断带 $\delta\le T$ 和首交之前的自由空间 $\widetilde q<\lambda-T$ 共同形成授权域。视野外射线提供未知，保持其覆盖含义。

足迹枚举使用球投影的精确包围盒。在 $Z>\tau$ 时，横轴归一化边界为 $(XZ\pm\tau\sqrt{X^2+Z^2-\tau^2})/(Z^2-\tau^2)$，纵轴把 $X$ 换成 $Y$；分别应用 $f_x,c_x$ 和 $f_y,c_y$ 后枚举整数像素中心，最后按式 (12a) 筛选。球跨越相机平面时在图像内仍可取得支持，外部无界足迹使完整穿透覆盖条件为假。

中心射线的重测归属优先于邻域中的支持，避免邻近噪声像素把已重新测量的旧位置再算作支持。`sameSurface` 的身份输入来自对象状态，几何输入来自当前重建，尺度输入来自式 (11)。背景颜色、光照变化分别进入外观属性，几何判定使用深度和状态。

### 6.2 从解释损失导出保留式

同一测量只产生一类证据。支持票 \(S\) 的最佳解释为保留该位置，穿透票 \(F\) 的最佳解释为清空该位置；满足有向对应的重测票 \(R\) 已由当前重建解释，其最佳位置为对应的新表面。遮挡票 \(O\) 对被遮挡位置的两种存在性解释有相同损失，因此在损失差中相消。定义每类测量的不一致损失：

\[
 (c_i(1),c_i(0))=
 \begin{cases}(0,1),&v_i=S,\\(1,0),&v_i\in\{F,R\},\\(0,0),&v_i\in\{O,\varnothing\}.\end{cases}
 \tag{13}
\]

记保留位为 \(z\in\{0,1\}\)，历史先验 \(p\) 与测量损失相加：

\[
 \mathcal L_e(z)=z(F+R)+(1-z)(S+p),\qquad
 \boxed{z_e=[p+S>F+R]}.
 \tag{14}
\]

式 (14) 是式 (13) 的逐测量不一致数最小解。历史先验 \(p=1\) 给已保存存在性一票；新补全面以 \(p=0\) 由本次支持建立。无观测旧面保留、无观测补全面退出、几何支持与穿透平票时旧面保留。近端回波只有满足上述有向对应时才承担重测反证。

所有历史面和补全面使用式 (12)—(14)。装配只消费已计算的 \(z_e\)，依物理身份写入对应槽；当前重建直接进入它的归属槽。每个输出槽先完整构建后提交，空输出也作为该次估计的结果。背景槽从空地图开始同样存在。属性按来源搬运；当前重建的时间来自式 (9a)，历史顶点按式 (14b) 合并同几何观测的属性和时间。输出网格统一携带颜色、语义标签、首次和最近观测时间数组，属性标志与数组一起提交。来源未记录的时间以零值表示未知，颜色和语义标签使用同槽已有属性。

候选推理与最终装配共用“同一物理槽、三个世界坐标顶点的相同有向循环”的规范键。候选先按该键形成一个假说，装配再按该键每类保存一个面；当前重建作为本次估计优先，之后为保留历史面、补全面。此操作只移除坐标完全相同的存储副本，保留不同三角剖分和不同位置的表面。颜色与光照属于等价类代表的属性，独立于几何键：

\[
 G=\{f\in P\cup\{e:z_e=1\}\}/\sim,\quad
 f\sim g\iff\ell_f=\ell_g\land\operatorname{cyclicMin}(V_f)=\operatorname{cyclicMin}(V_g).
 \tag{14a}
\]


同槽、同有向三角面在当前在线网格与历史中重复时，几何假说仍由历史代表承担，属性使用该等价类的实际观测。对匹配顶点，设已记录的最近时间为 $t_j$、首次时间为 $b_j$，未知时间取零：

\[
 t^+=\max_j t_j,\qquad b^+=\min\{b_j:b_j>0\},\qquad
 a^+=a_{\arg\max_j(t_j,\operatorname{current}_j,-\operatorname{index}_j)}.
 \tag{14b}
\]

首次时间集合为空时仍为零。颜色与语义属性 $a$ 采用最近有记录的来源；同时间优先本次在线估计，同次重复顶点以索引确定代表。缺少对应属性的来源保留已有属性。该合并仅针对同槽同面上的精确坐标匹配，历史来源误差、几何先验和表面保留位沿用原假说。因而重复存储归约与外观更新分别保留几何证据和观测新鲜度。

源码：`session_core/src/surface/session_refusion.cpp`。归档帧解码、必需的表面更新及状态保存失败会中止本次最终发布，执行器获得失败结果。输出顺序是背景、DSG 对象顺序；每槽为保留输入面、当前重建面、保留历史面。每个输出面的来源误差按同一顺序写入 `surface_error.bin`，并绑定完整表面几何及身份摘要。

## 7. 终结、序列化与递归接续

在线变化检测在每个独立片段上完成；其后按物理 ID 归并逻辑实体。终结最后一次归并后重新处理新摄取片段并刷新射线索引，然后执行会话末表面更新，最终时间与最后 ACK 输入一致。

恢复版持久包包含：

| 文件 | 保存的内容 | 下一会话用途 |
|---|---|---|
| `final.4dmap.zpk` | 本 session 的时间线及更新后最终图 | 评估、查看、旧格式恢复入口 |
| `chain_state.4dmap.zpk` | 表面更新前的最终推理 DSG | 下一会话在线变化推理的 seed |
| `shown_state.4dmap.zpk` | 更新后最终表面的一份快照 | 下一会话表面更新中的旧记忆 |
| `registry_state.cbor` | current 元数据、各个未决片段及输入处理水位，绑定 chain 文件 SHA256 | 恢复完整待决集合与状态所有权 |
| `surface_error.bin` | 每个最终表面的来源误差及几何身份摘要 | 式 (11) 的历史误差窗 |
| `depth_scales.txt` | 历次深度尺度 | 校准审计记录 |
| `sensor_statistics.txt` | 缺失证据总体统计的可读导出 | 校准审计及旧格式接入 |
| `evidence_state.cbor` | 实例独立的总体校准、活跃查询几何及逐格最新证据 | 式 (7) 的增量接续 |
| `session_bundle.json` | 本次发布成员的长度、摘要和终结时间 | 完整提交校验 |
| `transition_manifest.json`、`state_summary.json`、`control/` | 数据、版本、配置、帧账、退出状态、地图摘要 | 追溯和验收 |

加载时优先读取 chain，旧格式直接读取 final；把最新推理 DSG 作为一个初始快照，随后追加本会话状态。已有各 session 完整历史留在各自地图文件中。current 的几何与状态元数据共同写入 OBJECTS：当前静态状态是否存在、出生时间、最新实际支持时间、所选状态的跟踪起点。恢复这些元数据时使用相同的 current 几何；观察段的原始时间以 `session_input_first/last` 独立用于摄取去重；提供给 Khronos 的观测边界随物化 current 的几何一起更新。旧格式读取有限的显式观测时间或网格时间；无限 presence 仅表示状态延续，已知首次观测提供最近已知支持的下界。输入地图的时间单独作为处理水位。证据模型保存各查询已处理时间和最新逐格表决，下一会话在同一查询上只处理新增帧；实际支持时间决定退出证据的有效窗口。

接续状态以 terminal drain 后的 current 和未决集合为活跃状态。current 几何复用 chain DSG；每个未决片段保留自身几何、包围盒、直接支持与确认支持时间、状态键。证据键在同一状态的归并、晋升和保存恢复中保持不变。序列化绑定具体 chain 文件摘要，全部记录验证后一次性替换内存 registry：

\[
 P_{\rm registry}=(w,\{\ell,c_\ell,\mathcal N_\ell,\eta_\ell\}),\qquad
 \operatorname{Load}(\operatorname{Save}(P_{\rm registry}))=P_{\rm registry}.
 \tag{15}
\]

这里 \(w\) 是已排空输入的处理水位，\(\eta\) 为已经发生状态变化的摘要。下一会话完整保留直接支持与已确认支持，新的测量继续推进确认时间；快照水位只限定接续输入。完整闭合几何由所属会话产物提供，未来背景维护所需的清理义务单独携带。 背景义务绑定 chain 中表面的世界坐标及该表面的原重建时间，只携带该点最近实际支持；状态关闭后新生成或重新支持的背景不继承旧义务。候选关联使用既定地图单元对角线，最终是否清空仍由该点之后的真实射线及原生变化检测决定。同一个背景点的多项关联取最近支持时间最大值，形成一个查询：

\[
 D=\{(x,t_{\rm mesh},s):x\text{ 与已关闭状态有关}\},\quad
 s(x)=\max(s_{\rm state},v_{\rm state},t_{\rm mesh}),\quad
 D^+=\operatorname{bind}_{G_{\rm chain}}(D\cup D_{\rm newly\ closed}).
 \tag{15a}
\]

载入后的 bind 要求世界坐标和重建时间都精确相同；清理过或重建过的表面自然退出该集合。此状态大小由仍在 chain 中的候选表面决定，闭合物体几何留在所属 session 的历史产物。

当前恢复版直接使用原有 map 与 sidecar，`session_surface.bin` 未加入此状态协议。D3 通过加载与接入现有变化检测实现，下一会话按同一入口继续。

### 7.1 活跃查询的充分状态

一个建图实例独立持有校准总体 \(H\) 和活跃查询集合。查询由物理身份、不可变证据键及实际取样几何定义，保存采样合同、出生时间、已处理帧水位、局部校准和逐格最新表决：

\[
 E_f=(\ell,k_f,G_f,b_f,w_f,w_f^{\rm cal},H_f,
        M_f,M_f^{\rm cal},\mathcal Q_f),\qquad
 M_f=\{(u,h_u,t_u^{\rm reliable},t_u^{\rm surface},p_u^{\rm surface},
                t_u^{\rm free},p_u^{\rm free})\},\qquad
 \mathcal Q_f=\{(t_j,k_j,n_j):t_j>w_f^{\rm cal}\},\quad
 P_E=(H,\{E_f:f\in\mathcal F_{\rm live}\}).
 \tag{15b}
\]

每个新帧同时推进评分 mosaic 与候选校准 mosaic；候选只由帧开始前的可靠集合产生。直接支持水位授权前，候选的时间与两个计数保存在队列中，水位推进后按序归入局部参考分布。队列和候选 mosaic 跨会话恢复，从而支持迟到授权而无需保存旧 RGB-D。registry 独立保存直接几何支持与传感器确认时间。固定查询的未来更新只读取这些量及新增测量，因此保存、载入后对同一新增输入得到相同预测。载入以完整状态替换实例状态；重复载入保持同一统计。活跃集合来自 registry 的 current、会话 current 与未决片段。退出活跃集合后释放逐点记录，其已校准的总体成分保留在固定大小的矩汇总中。取样几何或合同变化时重新处理当前会话归档帧，并按式 (7b) 替换该查询的总体贡献。新几何上可用的测量域由实际帧归档决定。

当前总体在一次完整裁决开始时冻结；同轮各物体和嵌套查询使用同一冻结值，避免遍历顺序改变本轮先验。模型跟随后端对象释放，独立后端各自保存和恢复自身状态。

### 7.2 同一历史查询与完整发布

项目查询采用因果快照：时刻 \(t\) 读取不晚于它的最后一个已经发布的快照，返回独立副本。查询早于第一份快照时结果为空。快照中的 current 几何描述当时已完成的推理；各会话的历史通过其自身产物查询。

\[
 j(t)=\max\{j:t_j\le t\},\qquad Q(t)=\operatorname{clone}(G_{j(t)}).
 \tag{16}
\]

查看器、导出器、状态检查和会话 seed 使用同一项目查询接口。压缩快照按需解码，查询顺序保持结果一致。状态摘要协议 `session_update_current_scene/v2` 按式 (14a) 的有向循环面归一：循环重排保持相同摘要，反向绕序保留为不同几何状态。

最终发布先验证终结已完整成功，再独占建立新协议标志。每次发布使用新的输出目录，完整产物保持不可变；中途失败保留协议标志，读取时报告未提交。`session_bundle.json` 列出 final、chain、shown、registry、证据、校准及表面误差文件的长度和 SHA256。所有文件成功写入并同步后原子提交该记录；各项目读取入口先验证整个记录，再读取各状态组件。必需表面更新成功后才设置终结完成状态，原生保存函数的成功值传给项目发布入口。

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

构建记录 `ports/mapping_core`、`session_core` 和运行层源码指纹；每次运行前按同一文件集合重新计算并与构建记录比较，同时核对安装位置及动态库来源。运行 manifest 绑定检查通过的两份源码摘要和启动时提交号。构建使用锁避免并发构建。当前 Hydra/PGMO 等研究依赖由 `/home/jixian/ros2_ws/install/setup.bash` 提供；Khronos 链接到本项目 canonical install。已有构建按脚本支持的 `--incremental` 方式更新。

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

当前任务是在上述源码边界内完成原理推导、统一实现、独立审查及真实指标验收。前段的恢复与短程检查记录属于 rewrite 开始点。地图质量验收继续使用真实在线 A→B→C 的物体/几何、D1/D2/D3、残影、中间状态与最终图指标，并与纯 B、有效历史版本逐项比较。离线快速评估用于定位原因，报告单独标注输入来源。算法改变先落实数学含义，再实现和运行；场景名、特定物体编号只用于数据或评估记录。

旧 V6 的全新在线执行器、重写 Khronos 融合和 D1/D2、同时维护两套参考档案等任务已移出活动设计。fin4 原文及其表面模型的比较放入归档审查记录。当前唯一活动说明按上述 Khronos 复用边界演进数学模型及其实现。

历史设计归档于 `/home/jixian/Desktop/FT/archive/design_models_20260930/`；旧 README 和恢复前 TODO 保存在本仓库 `archive/design_consolidation_20260930/`。Windows 的同批设计稿归档于 `D:/3Study/ETH/FT/archive/design_models_20260930/`。V3 等原文均保留。

合成地图清理沿用既定规则：新一轮完整运行、评估并更新 CURRENT_VIEW 后，核对旧产物已被替代且无进程使用，再记录路径/尺寸并清理。保留原始输入、USD/GT、真实 ABC 地图、当前可查看结果、日志和指标。

## 10. 来源与许可

Khronos：MIT-SPARK，BSD-3-Clause，`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`。
Panoptic Mapping：ETHZ ASL，BSD-3-Clause，`3926396d92f6e3255748ced61f5519c9b102570f`。
移植文件保留原许可证和文件头；原 namespace 用于来源追溯。仓库：`https://github.com/giltchcity/session_update_baseline_project`；论文：`https://github.com/giltchcity/myncv---rpsl---daicma`。
