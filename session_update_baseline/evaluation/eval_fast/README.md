# eval_fast — 独立的低磁盘评测副本

这个目录在 `/home/jixian/Desktop/FT/eval_fast`。旧 `eval/`、`tools/` 和 `results/current_version_20260921/tools/` 未被修改。`vendor/` 保存从这些目录复制的评分代码、源码和可执行文件；`official/install/` 是官方 Khronos evaluator 运行时的副本。`run.py` 只负责按时间片取图、调度原版程序、恢复原始编号与汇总输出，没有重新实现 object、geometry、change 或 ghost 的评分公式。复制来源与哈希见 `COPIED_SOURCES.json`。

## 解压与磁盘

`.zpk` 仍需做 zstd **解码计算**，但不会生成完整 `.4dmap` 磁盘文件。每次最多取一个时间片到 `/dev/shm/eval-fast-*`，原版 C++ 的临时 spill 也被 `TMPDIR` 放入同一临时目录，评完自动删除。`/dev/shm` 在当前 WSL 是 `tmpfs`（RAM）；入口在每片前检查可用 RAM、tmpfs 空间和 swap 是否新增。若系统在评测进程之间突然出现严重内存压力，Linux 仍可能换页，因此这不是对硬件层面“绝无写盘”的保证。持久输出是指标和逐片导出，规模是 MB 级而非几十 GB 地图。

`eval/run.sh` 的旧入口仍会调用 `map4d_open.py` 完整解包；压缩图评测请使用本目录入口。

## 快速定位（约一片的时间）

```bash
python3 /home/jixian/Desktop/FT/eval_fast/run.py \
  --map /home/jixian/Desktop/FT/runs/t2_bc_20260923/session_b/final.4dmap.zpk \
  --session b --name candidate_b --mode quick --indices final \
  --out /home/jixian/Desktop/FT/eval_fast/outputs/candidate_b_quick
```

`--indices 0,31,final` 可检查首、中、末三片；省略时默认取首、中、末。`quick` 只产生所选时间片的官方 object 行；若包含终态，则另输出与旧方法相同的 viewer PLY、G1/G2 几何指标。它是诊断结果，不能代替完整在线指标。

G1/G2 测的是最终地图（2026-10-23 起）：viewer PLY 是终态时刻的查询视图里的背景，加上在终态时刻在场（`isPresent`）的物体（`vendor/geometry/geometry_export.cpp`）。带网格但不在场的节点（4D 历史、已消失的物体）不算终态几何。此前的口径是"有网格就算、不看在场"，原版在 `/home/jixian/Desktop/FT/backups/eval_fast_geometry_20261023/`。

## 完整指标（A/B/C 时间线）

```bash
python3 /home/jixian/Desktop/FT/eval_fast/run.py \
  --map /home/jixian/Desktop/FT/runs/t2_bc_20260923/session_b/final.4dmap.zpk \
  --session b --name candidate_b --mode full \
  --prior /home/jixian/Desktop/FT/runs/t2_a_20260923/session_a/final.4dmap.zpk \
  --out /home/jixian/Desktop/FT/eval_fast/outputs/candidate_b_full
```

完整模式按原始时间顺序调用复制的官方 `exp_pipeline`、`export_timeline`、`export_cleanup_binary`、`cleanup_mesh` 和现有派生指标脚本。B/C 必须给出 `--prior` 标记 D3 来源；真正被评测的继承状态在当前地图首片中。输出在指定目录的 `real_abc/` 中，保留原 harness 的 `objects/exports/changes/geometry/ghost/summary` 结构。一次评测只启动一个 evaluator 链，避免同时占用大量内存。

与旧完整评测逐字节核对时，加：

```bash
--compare-archive /home/jixian/Desktop/FT/results/current_version_20260921/real_abc
```

此参数要求 `--name` 与旧结果标签一致，例如旧目录 `objects/eval_cand_t2_b` 对应 `--name eval_cand_t2`。官方 object CSV、时间线 JSONL、cleanup 二进制、ghost mesh JSONL 必须分别完全一致，否则评测退出失败。

单独核对终态 PLY 与所有汇总数值，可在压缩导出前运行：

```bash
python3 /home/jixian/Desktop/FT/eval_fast/verify.py \
  --run /home/jixian/Desktop/FT/eval_fast/outputs/candidate_b_full \
  --archive /home/jixian/Desktop/FT/results/current_version_20260921/real_abc \
  --geometry-ply /home/jixian/Desktop/FT/eval/results/real/ours/b/geometry/mesh_viewer.ply
```

`PARITY_REPORT.json` 会分别记录原始文件哈希一致性和每个指标区段的数值一致性。路径与结果标签属于元数据，单独列出，不混入数值判断。

评测完成且已核对指标后，可压缩由本工具生成的大导出：

```bash
python3 /home/jixian/Desktop/FT/eval_fast/compact.py \
  /home/jixian/Desktop/FT/eval_fast/outputs/candidate_b_full
```

`compact.py` 逐个验证解压 SHA-256 后，才把本次生成的 PLY、JSONL、探针二进制换成 `.zst`；指标 JSON/CSV 保留原样，`COMPACTED_EXPORTS.json` 记录原始哈希。压缩后的导出仍可恢复，但若要重新运行派生指标，需先恢复对应文件。B 终态快评的 38.7 MB PLY 实测压到 14.8 MB。

## 已验证的 B 实验

`outputs/t2b_quick_final`：终态模式耗时约 41 秒，官方 object 行、几何 PLY 字节、G1/G2 的 P/R/F1 数值与旧整图评测一致。`outputs/t2b_full_parity`：63 片完整模式耗时约 716 秒，官方 object CSV、timeline JSONL、cleanup 二进制、ghost mesh JSONL、终态 PLY 都逐字节一致；object、change、geometry、ghost 的 61 个汇总数值字段全部一致。全量结果占用从 289 MB 压到 45 MB。最小观测可用 RAM 为 47.55 GiB，最小 `/dev/shm` 可用空间为 27.46 GiB，swap 使用为 0，整图解包缓存为空。

## 外部依赖与边界

- `configs/*.yaml` 是现有官方 GT 配置副本，仍指向 FT 的原始 GT/数据集；数据没有重复复制。
- ROS Jazzy、`ros2_ws` 和 canonical mapping 的共享库仍来自已安装环境；代码和官方 evaluator 的 `install/` 已复制到本目录。
- 几何参考网格、裁剪参数、RGB-D 证据仍从 FT 的原始只读数据目录读取。
- 建图程序当前仍先产生完整 `.4dmap` 再压缩成 `.zpk`。本目录解决**评测时再次完整解包**的问题；避免建图阶段的原始地图峰值，需要后续在 mapper 直接写 `.zpk`。
