# Uniform 保留＋关键帧扩容：设计依据

日期：2026-09-11。状态：设计说明；正式执行规范以同目录
`EXPERIMENT_PROTOCOL.md` v1.0 为准。

## 核心动机

旧 OC 在固定 32 帧容量里优先加入 boundary，再用时间覆盖补全。它改变了
原 U 的帧集合，因此 OC 没提升并不能区分两种解释：boundary 本身没价值，
还是加入 boundary 时丢掉了有用的 U 历史。

新实验完整保留原 U32 集合，并只在扩展槽位中加入额外信息：

- U：原生 32 帧 / 512 tokens，作为同批重新运行的基线；
- UK48：U 加当前因果历史里所有不在 U 中的 boundary；
- UN48：U 加等量的随机非 boundary、非 U 帧。

主要比较是 UK48−U；UN48−U 与 UK48−UN48 只用于辅助判断一般扩容历史和
boundary 内容各自可能带来的作用。

## 为什么容量为 48 帧

对既有 62,492 次策略调用的只读核算中，U32 与全部可见 boundary 的并集
最大为 46 帧，99% 分位约为 44 帧。48 帧覆盖已有最大值并留 2 帧余量，
同时比直接扩到 64 更节制。每帧 16 tokens，因此扩展组固定为 768 个
memory-token 槽位；不足部分只做 zero padding 和 false mask，不人为填满。

这不是新轨迹永不溢出的证明。超过 48 或 UN48 候选不足均按协议硬停，
不得临时改规则。

## 为什么 U 必须重新运行

早期方案想复用 Athena 上的 800 条 U。跨服务器审计发现，旧 U 与新
Lighthouse 环境的初始 front-view 输入并非字节一致。即使肉眼相近，也无法
证明差异只来自记忆选择。v1.0 因此将三个组都放在同一 Lighthouse 实验族中
重新运行，并逐 task/episode 核验初始 front、wrist、state 和 task evidence。

正式规模由此变为：

```text
16 tasks × 50 test episodes × 3 arms = 2,400 trajectories
```

旧 Athena U 只保留为历史证据，不进入新实验的正式配对统计。

## 模型形状与执行方式

U 使用发布模型原生的 512-token memory shape；UK48/UN48 使用同一 checkpoint
权重和推理期 768-token shape。Memory Modulator 的相对位置计算会随 sequence
length 变化，因此不能把 U 填充成 768 再声称它仍是原版 U。

模型加载后的 shape 是静态的，所以单个常驻进程不能交替跑 U 与扩展组。
执行时需要 U-only 进程和 UK48/UN48-only 进程，但它们仍属于同一个正式矩阵、
同一 provenance 家族，并按同一 task/episode 做三组配对。

## 能与不能得出的结论

若 UK48 优于 U，说明“保留 U 并扩容加入 boundary”这一整体方案有用；若还
优于 UN48，则进一步支持 boundary 相对等量随机历史具有特殊价值。实验仍不能
单独隔离所有容量、位置编码或额外真实信息效应，也不能证明 boundary 是全局
最优记忆，或真实机器人天然拥有 oracle boundary。

这是固定 checkpoint、无训练的 test-time 选择实验。不得加入 symbolic
subgoal、匹配训练、邻域帧、动作 oracle、未来 boundary 或按结果挑 seed。
