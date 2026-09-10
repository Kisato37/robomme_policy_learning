# UK48 / UN48 运行适配验收

日期：2026-09-10。对应协议：`EXPERIMENT_PROTOCOL.md` v0.9。

**本次完成的是本地运行接口与证据链，不是服务器/GPU实验。** 没有 SSH、提交或推送 Git、下载数据、加载真实 checkpoint、运行仿真/GPU轨迹或训练模型。所有模拟环境、策略输出及统计样本均为测试夹具；不能据此判断 UK48 是否优于 U。

## 1. 本轮接通的流程

```text
预约不可覆盖的 attempt
  → 原 EnvRunner 初始化一次，取得完整初始观测/演示前缀
  → 核验常驻模型服务身份，清空上一集状态并固定 policy seed
  → 保存无损初始图像、机器人状态、任务状态、任务文字及其校验信息
  → 每次送入新增历史，核验因果边界与选帧记录
  → 预测20步、执行前16步；正式终止时停止
  → 发布录像、结果和完整 trace，关闭该集连接/环境
  → 验收产物后才接受该科学结果
```

服务器可以为不同 episode 复用同一个模型进程，但每一集必须重新核验和重置。关闭 episode 连接不等于卸载模型。这里没有实现未经审批的自动服务器启动、GPU分配或正式任务提交入口。

| 独立文件 | 作用与约束 |
|---|---|
| `serving.py` | 新实验专用 client/server；配置方法、消息格式及身份校验不复用旧组接口。一个常驻服务同一时刻只接受一个有效 episode；协议/模型错误会停止继续接活。 |
| `evaluator.py` | 调用原 `EnvRunner`、`EpisodeState`、`pack_buffer` 等组件执行一集，不更改旧评估路径；保存完整 reset-prefix，校验每次调用的真实历史与因果边界。 |
| `artifacts.py` | 建立独立 run/attempt 目录，保存初始无损数组、任务状态/文字、选帧 trace、录像与结果；结果写入后不得覆盖，文件与清单相互绑定校验。 |
| `outcome_ingestion.py` | 只读验收完整1,600条正式新组结果，生成统计内核所需记录及逐文件来源链；核对800对 UK48/UN48 初始条件。不能接纳 smoke、部分结果或损坏记录，也不生成 U 基线认证。 |
| `audit_local_baseline.py` | 只读核查本地旧 U 结果和迁移证据；不连接远端，也不把历史 PASS 标记当作重新核验原始文件。 |

新组仍是完整保留原 U32 选帧后追加边界或等量随机非边界；固定48帧槽，但不足48的部分不会补入普通帧。新增接口没有将旧 Uniform 默认配置改为48帧。

## 2. 本轮重点验证的保护

- **不丢初始演示**：含演示的任务可在第一次推理前已有大量观测；存储与选帧从整个前缀开始，不仅取最后一张。
- **不混用 episode**：模型身份、配置摘要、逐集 seed、历史、边界、调用计数和选择器状态都要检查；第二集不能沿用第一集记忆。
- **不混用动作长度**：收到20×8动作后按原设定执行前16步，1300步上限处不越界；环境提前 success/fail/timeout/error 时按正式终止处理。
- **不择优重跑**：科学失败同样是已完成结果，不能重试。仅限定的连接/文件系统故障可进入基础设施失败路径；其他异常停止等待检查。
- **不利用后处理故障重跑**：一旦已知正式终止或到达该行上限，之后录制最后一帧、保存录像、发布产物出错都要求停止并保留终止证据，不能变为可重跑故障。若结果已发布、账本却写入失败，也只能审计恢复。一个硬停止会阻止同 run 新开其他 attempt。
- **可重新检查输入**：保存无损 `initial_observations.npz`、保留类型的 `initial_task_state.json` 和 `initial_task_instruction.json`，而不只是录像截图或 hash PASS 标记。NPZ 不使用 pickle。
- **不拖慢每次推理**：活跃 writer 复用已校验的初始证据，避免每次记忆更新都重新解压/哈希长演示；最终完成与离线审计仍重验磁盘证据。

这些机制验证的是实现与记录一致性，不单凭内部摘要证明真实模型、GPU或模拟器来源。实际执行来源仍需由后续服务器启动前检查绑定。

## 3. 分阶段运行测试

本轮四套运行接口测试一起执行：

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/uniform_keyframe_expansion/test_artifacts.py \
  tests/uniform_keyframe_expansion/test_evaluator.py \
  tests/uniform_keyframe_expansion/test_episode_integration.py \
  tests/uniform_keyframe_expansion/test_serving.py -q
```

结果：**91 passed，0 failed，7.71秒**。其中包括真实本机 WebSocket、生产 `MemoryBuffer` 和新选择器的联测；只替换环境、特征/动作生成和录像内容。没有读取真实权重或产生可用于实验分析的机器人结果。

随后只读复核发现并修正了一个边界问题：**已收到科学终止，但结果文件尚未发布时，录像/归档的磁盘错误原本可能被归为可重跑故障。** 现在在取得终止时先保存其证据，后处理故障一律不可重跑。增加21项测试，覆盖四种正式终止、达到上限、各录制/发布阶段故障，以及真实存储层拒绝同组重试和继续启动其他组。修改后 `test_evaluator.py` 与 `test_episode_integration.py` 合计 **60 passed，6.62秒**。

结果接入层另有36项测试；包括1,600格映射夹具，以及真实存储的 smoke/缺失/重复/损坏拒绝测试。跨组只要求初始条件和重置证据一致，不要求闭环运行后的边界、有效帧数、调用次数或终止结果相同。即使新两组配对通过，UK48−U 主统计仍需另外完成 U 的来源与匹配认证。

部分测试需要本机127.0.0.1监听权限，已仅为本机通信测试使用沙箱外运行；不涉及 SSH、外部服务器或 GPU。

旧核心实现验收见 `LOCAL_VALIDATION.md`；该文件的768项通过是前一阶段的历史记录，不应与本轮测试直接相加充当一次完整回归。

### 最终新旧合并回归

全部修复和接入代码稳定后，在现有 `.venv` 中执行：

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/uniform_keyframe_expansion \
  tests/keyframe_oracle_sampling \
  tests/keyframe_neighborhood_sampling \
  tests/dual_memory -q -rs \
  --junitxml=/private/tmp/uk48-runtime-tests.ME4qWy/full-regression-final.xml
```

**916 passed，24 skipped，202 subtests passed，0 failed；46.45秒。** 新实验288项、旧实验628项通过。24项跳过均为本机 macOS 不支持的 Linux `/proc`、进程组或继承监听端口检查，仍需在服务器补验。一个第三方 `beartype` 旧类型注解弃用警告，不影响测试结果。

JUnit 原始记录 SHA-256：

```text
7d1f902adf90bd72fcd07c7577e5d76d82890e7a91c80291e12c2653457c07fb
```

XML 位于上述本机临时目录，不是已推送或已归档的服务器产物。`git diff --check` 通过。旧 `policy.py`、`mem_buffer.py` 与两套旧实验协议目录未出现 Git 差异；旧科学配置没有为新实验放宽。

## 4. U 基线审计结果

详见 `BASELINE_EVIDENCE_AUDIT.md`。本地800条 U 结果完整，五份发布产物的大小/校验值也通过重新检查。但是，本地 U 导出只有“当时初始 hash 文件核验通过”的标记，**没有五项初始 hash 的实际值**。现有另一份完整 inventory 对应 OC，不能改名当作 U 证据。

本地还发现72条 U 的实际环境 seed 不能用简化公式复原，450条有多帧初始演示。因此后续必须沿用原数据解析器，并检查完整初始批次，不能只对照 task/episode 编号或第一张图片。

目前保持 `u_baseline_status: unresolved`。这是本地证据不足，不代表服务器文件丢失，更不代表已经决定重跑 U。没有生成基线匹配认证或改变 UK48−U 主比较。

## 5. 尚未完成的服务器工作

1. 将独立接口接入带来源校验和授权检查的服务器启动器，绑定真实源码版本、checkpoint、执行环境、物理渲染GPU及 run/row/attempt 身份；原服务器入口不能直接换参数使用。
2. 在同步获准后补验 Linux 专用测试，并只读取得旧 U 的原始初始 hash、逐集 manifest、结果及 launch/environment 证据。
3. 获准 GPU smoke 后检查真实权重严格加载、768槽推理、确定性及 padded-U512 对 U768 的位置编码诊断，再执行协议规定的32短＋16完整 val轨迹。
4. 完成真实产物接入及 U 可比性审核后，才可冻结正式版本并申请1,600条新轨迹。不能以当前CPU测试代替正式就绪审核。

本次基准 HEAD 为 `297cbb2abaf50e7005d40a3ffb7076cdc26ee986`，改动仍在工作区；该 HEAD 不是包含本轮实现的已提交版本。旧实验结果及其他未跟踪文件均未清理或覆盖。下一步交接见 `SERVER_HANDOFF.md`。
