# 旧 U 基线：本地证据审计

日期：2026-09-10。对应新实验协议：v0.9。

**结论：本地已经有完整的 800 条 U 结果摘要及相应配置记录，但还不能据此确认旧 U 可以直接作为新实验的严格匹配基线。基线状态保持 `unresolved`。**

本次只检查本地文件，没有 SSH、仿真初始化、模型推理、GPU作业或新数据下载；没有修改任何旧结果或协议，没有授权或执行 U 重跑。检查范围是本仓库的旧结果目录及同级 `lighthouse_migration` 迁移/诊断目录，不代表已搜索服务器或所有独立备份。

## 1. 能确认什么，暂时不能确认什么

| 项目 | 本地发现与证据层级 |
|---|---|
| 800 条 U 完整任务结果 | 从本地逐轨迹摘要重新计数：16 tasks × test episodes 0–49，800 个唯一科学键，没有缺失或重复 |
| U 成败结果 | 369 success、404 fail、27 timeout；与已发布 summary 一致；14,599 次策略调用摘要 |
| 五份已发布 aggregate 文件 | 本次重新读取原始字节，大小和 SHA-256 均与 `RESULTS_MANIFEST.json` 相符 |
| 每条 U 的运行配置 | 800 份嵌入式 episode manifest 均有 split、resolved environment seed、难度、policy seed、checkpoint ID、执行长度等字段 |
| U 原始伴随文件的完整性 | 本地导出保存了当时核验 manifest、完整 trace、initial-condition 文件均为 true 的结果；本次没有重新读取服务器原文件，不能称为重新核验了这些原文件 |
| U 五项初始 hash 的实际值 | 当前检查的本地 U 摘要没有保存实际值；只有 `initial_condition_hashes.json` 当时校验通过的布尔记录 |
| 现有完整初始 hash inventory | 包含 800 条旧 **OC**，不是 U。不能改名为 U，也不能当成直接重读 U 得到的证据 |
| 原始无损初始图像/状态数组 | 在本次检查的本地材料中没有找到可恢复旧 U 全部初始输入的数组；此前服务器 inventory 的限定搜索也没有找到所列无损图像格式，不能推断所有其他备份都不存在 |
| 新 UK48/UN48 对旧 U 的匹配 | 尚无新运行的初始输入与环境证据，未核验；旧 OC3/OC5 对比不能代替新组的匹配核验 |
| 新实验完整执行环境 | 尚未将实际选定的新运行环境与历史 U 的完整软硬件记录绑定，不能宣布可比性成立 |

“结果有了”和“结果能在新实验中作严格匹配对照”是两个不同的判断。本次没有把任何准备状态置为 `verified`，也没有生成正式 baseline attestation。

## 2. 已有结果来源及本次完整性核对

候选 U 来源 run ID：`20260829T231425Z_7b594786_formal_v1`。正式源目录标识仍为仓库相对路径 `runs/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1`，不是新建或重跑结果。

本地源文件：

- [发布清单](</Users/panjunyu/Downloads/world model/dual_memory_work/robomme_policy_learning/results/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1/RESULTS_MANIFEST.json>)。
- [原正式完整性报告](</Users/panjunyu/Downloads/world model/dual_memory_work/robomme_policy_learning/results/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1/aggregate/completeness_report.json>)。
- [原正式统计摘要](</Users/panjunyu/Downloads/world model/dual_memory_work/robomme_policy_learning/results/keyframe_oracle_sampling/20260829T231425Z_7b594786_formal_v1/aggregate/summary.json>)。
- [逐轨迹本地导出](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/failure_audit_20260908/athena_records.json>)。
- [该导出的采集脚本](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/failure_audit_20260908/collect_existing.py>)，本次只读其源码，没有执行其远端采集功能。

五个已发布文件的字节核验均与清单相符：`analysis.md`、`completeness_report.json`、`per_episode.csv`、`per_task.csv`、`summary.json`。CSV仅作为文件进行大小/字节 hash 核验；本次结果计数从 JSON 逐轨迹摘要独立计算，没有改写统计表。

`athena_records.json` 共包含 800 U 和 800 OC。U覆盖每个规范任务恰好50条，episode恰好0–49；其中786条使用 attempt0、14条使用 attempt1。该分布不代表本次发起了重跑。各 U 的 success 布尔值和 terminal_reason 一致，未出现将非成功终止写成成功的矛盾。

导出脚本只保留 `success`、`fail`、`timeout` 终止的记录，因此不能仅凭导出空错误列表就判断任何实验都没有 error。本次额外核对完整800键及原 summary；原 U 摘要的终止统计也只有上述三类，二者一致。

原完整性报告记载原3200格矩阵完整、800块初始条件和配置配对一致，原始结果/trace审计完成。这是**已有历史审计报告的声明**，不是本次在本地重做了所有原始轨迹文件审计。

## 3. Seeds、难度与初始化历史

本次逐条检查 U manifest 得到：

- dataset均为 `test`；policy seed均为7；max_steps均为1300；executed_action_horizon均为16；checkpoint ID均为79999。
- 难度分布为 easy416、medium192、hard192。这是原数据的已解析配置，不允许新实验把全部episode统一设为某种难度。
- 与相同 task/episode 的旧 OC 比较，dataset、max_steps、executed_action_horizon、evaluation_policy_seed、checkpoint_id、seed_table_sha256、resolved_environment_seed、resolved_difficulty_hint、difficulty 这9项字段，800块均无差异。
- **72条 U 的实际 `resolved_environment_seed` 末两位不是0。** 例如 BinFill ep1为540101，VideoPlaceOrder ep0为610002。因此不能自行用“任务基数＋episode×100”的简化公式替代原来的解析流程。新运行应使用原 resolver，并将实际解析值与原 manifest 逐条比较。
- 首次策略调用的历史长度范围为1–1140帧，其中450条 U 的初始历史不止1帧。对含演示的任务，只比较第一张图不能替代整个初始批次的核验。
- 800条 U 导出均含 `task_goal:` 的日志文本，但日志中的可读字符串不等于原始 task instruction 字节 hash 的直接证据。

脚本按规范任务顺序、episode顺序，将 task、episode_id、resolved_environment_seed、difficulty 序列化后的 SHA-256 为：

`ed5fef94211e17c6ef4b2efadd91e5446df780e931f6cdce2222a35ea4f8d483`

这是本次本地已解析映射的派生校验值，不是新 RNG seed，也不是原 seed-table 文件的 SHA。不得把环境 seed、policy seed7与UN48选帧 RNG 混为一谈。

## 4. 初始状态和图像证据为何还不够

采集脚本确实在当时读取了 U 的 `initial_condition_hashes.json` 并和结果记录的对应 SHA 比较，但输出仅保留 `hash_checks` 布尔值，没有将文件中的五项实际 hash 加入导出 JSON。本次看到的800个 true，是**当时文件完整性检查的记录**，不能用于现在计算 U 与新运行逐字段相等与否。

另外一份 [Athena初始 inventory](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/initial_image_audit_20260908_r1/athena_inventory.json>) 确有五项完整值——front、wrist、robot state、task state、task instruction——但其800条记录的arm全部是OC。原四组配对通过的历史报告提供了间接关系，仍不能将它描述为“已直接取得U的原始初始hash”。更不能用重新采集的新输入冒充旧输入。

已有 [跨服务器逐块检查](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/initial_image_audit_20260908_r1/paired_initial_records.json>)，本次重新计数如下：

| 已有比较，不是新 UK48/UN48 | 配对数 | front不同 | wrist不同 | 至少一路图像不同 | robot/task state及task text不同 |
|---|---:|---:|---:|---:|---:|
| 旧OC → OC3 | 800 | 254 | 350 | 350 | 0 |
| 旧OC → OC5 | 800 | 254 | 350 | 350 | 0 |
| OC3 → OC5 | 800 | 0 | 0 | 0 | 0 |

涉及 InsertPeg、MoveCube、PatternLock、RouteStick、VideoPlaceButton、VideoPlaceOrder、VideoRepick。其余450块两路图像相符，但不能据此只选这450块作新正式主分析。

这说明过去已经发生过“seed和状态配置匹配，但输入图像批次hash不同”的情况。它不证明将来的UK48/UN48一定不同，也不证明差异必然降低成功率；它说明不能跳过新比较的输入核验。详见 [原图像审计报告](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/initial_image_audit_20260908_r1/REPORT.md>)。

MP4为有损录像。上述报告已发现即使原始图像hash相同，解码MP4也有像素差异，因此不能凭录像肉眼相似、MP4像素误差小，或单帧截图来认证原RGB批次相等。暂缺本地原数组也不等于服务器上的hash文件丢失；本次没有访问服务器确认现存状态。

## 5. Checkpoint、代码及执行环境

原完整性报告提供以下可追溯标识：

- 正式评估代码commit：`7b59478619db222eefcd30575e268e1384f75953`。
- 原聚合代码commit：`ac77b3ca3983875cbbc588ab6156107f9c284443`。
- checkpoint内容树：`d25ea735413db4a2a52febe95128ed158c550501fe4edee4ec6b78049b6d3fbf`。
- checkpoint元数据：`313e483ae32881e606402365a8b01a599eda22a367e436f9dfbd5456120dca26`。
- 原launch manifest的已记录SHA：`60ddbf2c114c3581d0cbeea6a973cb5c5600a94c1628b77804276b7e2e50cd05`。

[迁移时checkpoint核验记录](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/checkpoint_verified_20260907.json>) 的内容树和元数据与原报告相同，archive SHA也与新协议要求一致；记录为18个文件、11,877,152,238字节。这是本地缓存的迁移核验结果，本次没有读取118亿字节权重或重新核验服务器权重，也没有执行768-token加载测试。

[Athena源码检查](</Users/panjunyu/Downloads/world model/dual_memory_work/lighthouse_migration/initial_image_audit_20260908_r1/athena_source_check.json>) 记录2026-09-08当时的policy HEAD为d6d1b72、benchmark HEAD为856bc3a；它提供相关图像读取/哈希方法的指纹。不能把该较晚检出状态当成2026-08-29正式执行时的完整环境。当前源码相同，也不等于GPU、驱动、渲染库、Python及底层依赖全部与历史运行相同。

本次指定结果目录主要发布aggregate及清单，没有旧U的逐attempt原文件和旧launch manifest正文。迁移目录存在较完整的Lighthouse环境检查及OC3/OC5启动记录，但不能替代历史U环境正文。要认证新主比较，仍需绑定旧运行的实际环境证据和新选择环境的实际证据；不能只填一组相符的字符串或hash格式就置为verified。

## 6. 下一步应补的证据，不代表启动授权

1. 在获准的只读同步中，取得旧800条U的原始 `episode_result.json`、`episode_manifest.json`、`initial_condition_hashes.json` 和相应SHA链；完整trace及原launch/environment记录按新验收需要同步。保持原始字节，不重写或覆盖旧文件。
2. 先对旧文件本身复核完整性，明确哪些初始数组确实有归档、哪些只有hash；为含初始演示的450条保留批次长度与标签对齐证据。
3. 固定新环境后，核验其初始图像/状态/文本及实际seed映射。若另做reset-only捕获，只有匹配旧hash的捕获才能作为该旧输入的可验证重建；不匹配时必须如实保留。
4. 将旧结果来源、对应证据和新输入比较绑定成单独的审计清单，经复核后才决定旧U是否可复用。不能用这个只读盘点报告自动生成正式比较认证。
5. 若旧U最终仍不可复用，再提出单独的U基线方案并取得用户授权。**本报告不宣布必须重跑U，也不自动新增800条轨迹。**

## 7. 本次可复查文件指纹与命令

| 文件 | 本次读取的SHA-256 |
|---|---|
| RESULTS_MANIFEST.json | `88928c467dc37d139d9d8428ce754af1b7a18aafd4e512f7eed520817b8a46ac` |
| completeness_report.json | `9f58713cb8da6762c3a81fcfa1776c25bb52a1be75ed26326325ce0cec655ccc` |
| summary.json | `e3f40b7192fc1be9a3b280c30234170b63dfae26d389c9b99ba321e82e3efbe9` |
| athena_records.json | `1dda96f1673fd971d1f6fcd7f3a389ad5562789de95d54aff7660dcacfd5a11b` |
| collect_existing.py（只读，未执行） | `9200caa6487ec16b72588acd4c94c5160ae2589acd1fc1cfe4ff891872013cea` |
| athena_inventory.json | `30ab275cc281493a926df83ee50b26627512c955f67b38043d88b5c0a8f0cdd9` |
| paired_initial_records.json | `70838361dcd13c632b3db9f741262d8557992baa2fbac97ba929a7d1a750d631` |
| athena_source_check.json | `8799fc9b6d61e0150a2644c2e36c0eb5603a087a0b1a8671f1fd745f3bc65791` |
| checkpoint_verified_20260907.json | `5725e1ac69970edd51ba9602fbff9eab5c63f004d4b225da9bc59ebdb353fcf6` |

新增 [本地只读盘点脚本](</Users/panjunyu/Downloads/world model/dual_memory_work/robomme_policy_learning/experiments/uniform_keyframe_expansion/audit_local_baseline.py>) 仅使用Python标准库读取上述缓存，输出JSON；不接网络、不导入模型、不写文件、不修改readiness，也不核验不存在本地的原始字节。已在仓库根目录执行：

```sh
.venv/bin/python experiments/uniform_keyframe_expansion/audit_local_baseline.py
```

输出明确保留 `baseline_status: unresolved`、`launch_authorized: false`。本次检查未发现已核对的本地摘要之间自相矛盾，但这不是对新主比较的可比性背书。
