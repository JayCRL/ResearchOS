# README（中文版）

[English](README.md) | **简体中文**

**ResearchOS 是面向 AI/ML 科研的 Research Operating System（科研操作系统）。**它不是聊天助手，也不是"AI 写论文工具"。它把**研究方向、证据、claim、文献语境与技能**变成人和机器都能审计的结构化状态。

> **聊天记录不是研究状态。**
> **LLM 不是科学真理的来源。**
> **论文不是真值来源，而是研究状态的编译产物。**

```
全局研究状态 → 当前任务 → Agent 执行 → 实验 / 分析 / 文献
      → 证据 → 审计 → claim 更新 → 研究状态更新 → 论文编译器
```

---

## 快速开始

```bash
pip install -e ".[dev]"

# 接管一个已经在进行的研究项目（这是最重要的第一体验）
researchos import ./my-existing-research --into ./my-project

# 看到的是研究状态，而不是一份摘要
researchos status
researchos dashboard

# 用有边界的任务工作，让局部问题保持局部
researchos task create "解释 seed-3 上的指标异常" --purpose DEBUG --priority EXPLORATORY \
  --stop "指标解释清楚，或上报给人决定"

# claim、证据与审计
researchos claim list
researchos claim audit          # 指出哪些措辞超出了证据，并给出校准后的改写
researchos evidence verify <evidence-id>
researchos audit stats
researchos mechanism --claim <claim-id>

# 文献与新颖性
researchos literature search "write placement in fast weights"
researchos novelty audit "nobody has compared placement under a matched-energy control"

# 项目能解释自己的历史，也能攻击自己的 claim
researchos timeline                       # 叙事线：观察 → 假设 → 实验 → 修正 → claim
researchos timeline ask --question "why was this control added?"
researchos redteam                        # 红队报告：什么能推翻它，最弱的一环在哪

# 论文是"编译"出来的，且拒绝无依据的数字
researchos paper compile --out paper.md
researchos paper readiness

# 技能必须挣得自己的状态，升级要过回归门禁
researchos skill benchmark --suite LITERATURE --seed-tasks
researchos skill evolve --from <skill-a>,<skill-b> --rule "每条 claim 必须带定位符" --findings findings.json
```

`researchos --help` 会列出全部命令。**详细使用指南（含实测输出、每种"拒绝"的含义、命令速查表、Python API）：[`docs/usage.zh-CN.md`](docs/usage.zh-CN.md)。**

### 看板（Dashboard）

```bash
pip install -e ".[api]"
researchos api                    # → http://127.0.0.1:8765/
researchos api --enable-actions   # 额外启用"人在回路"按钮（仅本机客户端）
```

单个 HTML 文件，**零构建、零 npm、零 CDN**——五年后离线也能打开。它分成**两个视图**，这个划分是设计规则而不是配色选择：

| | **Research Cockpit**（`#/`，默认） | **Audit Console**（`#/audit`） |
|---|---|---|
| 回答 | *我的研究走到哪了？下一步做什么？* | *这套系统可信吗？* |
| 内容 | 研究阶段、核心问题、当前可知的结论、**需要你处理的事**（按严重度排序、每条给出确切命令与"它阻断了什么"）、证据→claim 链条（为什么还不能升级）、论文就绪度（Not ready + 原因，7 维折叠展开）、**研究地图**（只画真实存在的关联）、项目在自身流程中的位置（reconstruct→review→resolve→verify→literature→analyse→promote→compile） | revision、integrity、event head、哈希、provenance、冲突明细、证据成熟度、技能健康、时间线、导入诊断、审计历史、复核队列 |
| 禁止出现 | revision / 哈希 / event head / integrity 计数器 / 技能计数 | — |

首页只回答五个问题：**我在研究什么 · 我已经知道什么 · 哪些还不能信 · 为什么现在不能继续 · 我下一步做什么。**

阶段不是心情，而是**最早未完成的 gate**：`QUESTION_UNCONFIRMED → IMPORT_REVIEW → CONFLICT_RESOLUTION → EVIDENCE_VERIFICATION → LITERATURE_AUDIT → EXPERIMENT_DESIGN → ANALYSIS_PENDING → CLAIM_VALIDATION → PAPER_COMPILATION → PAPER_READY`。地图只画记录里真实存在的边（claim→experiment 的声明关联、结构性 belongs-to、以及明确标注"同源而非支持"的 `SAME_SOURCE`）；一个没有实验指向的 claim 会显示为未连接，并**本身成为一条阻断项**——因为没有关联实验的 claim 永远无法离开 `HYPOTHESIS`。

看板只是**视图**：真值源仍是 `.researchos/`。除非加 `--enable-actions`，否则没有任何写操作；即使启用，
它也是**以 human principal 身份、走与 CLI 完全相同的 kernel 门禁**——证据不足的 claim 照样被拒、过期的
状态转换照样被拒，而且每次点击都会记录成 `ui.action` 事件，与 shell 命令可区分。

---

## 它如何解决"它之所以存在"的六个问题

| 问题 | 代码里的机制 |
|---|---|
| **上下文污染 / 研究主线漂移** | `research_state.yaml` 是真值来源；`Task` 必须带优先级、停止条件与允许动作集；受保护状态（核心问题、核心 claim、优先级、非目标、范围）**只能**通过被批准的 `StateTransitionRequest` 改变，而且 `DEBUG` / `EXPLORATORY` 任务连提交请求的资格都没有（`kernel/state.py`、`kernel/transitions.py`、`kernel/tasks.py`） |
| **编造事实、数字与实验结果** | 数字只存在于 `Analysis` artifact 中；论文编译器从 artifact 构建 `NumberRef` 数字池，并对渲染后的正文重新抽取每一个数字；匹配不上的数字是**阻断式** `UNGROUNDED_NUMBER`（`paper/grounding.py`） |
| **对已有研究理解不足** | `researchos import` 执行研究考古：扫描 → 分类 → 抽取 → 冲突检测 → 研究状态重建 → **人工复核队列**。导入的 claim 永远不会高于 `HYPOTHESIS`（`importer/`） |
| **文献检索面太窄** | `QueryFamily` 强制覆盖精确术语、同义词、历史术语、机制等价、功能等价、邻近社区与近期术语；覆盖率按查询族与 provider 实测，覆盖率不达标时新颖性结论在模型层就无法构造（`literature/`） |
| **论文"一眼 AI 写的"** | claim 是编译出来的，不是生成的：已批准 claim + 已验证证据 + 分析数字 + 带定位符的文献 claim。确定性文风审计器会标记空洞背景、模板化语言、重复 n-gram、流行词密度、无依据的因果/新颖性表述，以及**与研究史不符**（`paper/style_audit.py`、`claims/language.py`） |
| **科研能力本身不会进化** | Skill Meta-System：发现 → 注册 → 沙箱 → 基准 → 回归 → ACTIVE；并且规定**生成的技能不是可信技能**，破坏既有基准的升级会被拒绝。`researchos skill evolve` 会跑完整闭环（`skills/evolution.py`） |

**ResearchOS 明确不承诺：**不保证论文正确、不保证 claim 新颖、不证明机制、不保证实验可复现。它做的是让这些问题**可回答、可审计**，并把不确定性显示在界面上，而不是写进正文里。

---

## 一屏看架构

```
┌─────────────────────── RESEARCHOS KERNEL ───────────────────────┐
│ 全局研究状态 · 权限门 · 状态转换 · provenance · 任务边界 ·        │
│ claim/证据/文献/技能状态 · 只追加的哈希链事件日志 · 冲突账本       │
└───────┬───────────────┬───────────────┬──────────────┬───────────┘
        ▼               ▼               ▼              ▼
   Literature OS    Evidence OS    Experiment OS    Skill OS
        └───────────────┴───────────────┴──────────────┘
                        ▼
              AGENT OS（15 个有边界的 agent）
                        ▼
   AUDITORS（统计 · 机制 · 新颖性 · 文风 · 红队）
                        ▼
                 CLAIM REGISTRY
                        ▼
                 PAPER COMPILER
```

完整设计与取舍理由：`ARCHITECTURE.md` ·
不变量映射：`docs/invariants.md` ·
先行工作评审（借鉴 / 改造 / 拒绝）：`docs/prior_art_review.md`

---

## 磁盘布局：一切都是可评审的纯文本

```
.researchos/
├── project.yaml
├── state/
│   ├── research_state.yaml      ★ 全局研究状态（可 git diff 的真值来源）
│   ├── events.jsonl             ★ 只追加、SHA-256 哈希链历史
│   ├── tasks/                   带优先级与停止条件的边界化任务
│   ├── transitions/             每一次被提出与被批准的方向变更
│   └── decisions/               为什么改（永不重写）
├── claims/  experiments/  evidence/{,raw}  analysis/
├── literature/{papers,graph,claims,search,prior_art,novelty,gaps}
├── skills/{registry,versions,benchmarks,sandbox,gaps,deprecated}
├── audits/  conflicts/  review/  notes/  timeline/  paper/{artifacts,audits}
└── cache/                       可随时删除、可重建 —— 永远不是真值来源
```

---

## 十五个 Agent

Agent **不是**带系统提示词的自主循环。每个 agent 都是一个具名 principal，拥有固定的能力集；它永远可以**提议**，由内核决定是否提交。两个具体后果：

* `paper_writer` 不能提交 claim，也无法产生任何"没有 analysis artifact 支撑"的数字；
* `researchos agent list` 会打印每个 agent **不**拥有的能力 —— 包括 `claim.approve`、`state.transition.approve`、`state.core.write`、`skill.activate`，这些能力**没有任何 agent 拥有**。

```bash
researchos agent list
```

---

## 测试

```bash
python -m pytest -q                 # 180 个测试
python -m pytest -m invariant -q    # 只跑十条不变量
```

只读 HTTP 接口（可选依赖）与研究者声音视图：

```bash
pip install -e ".[api]"
researchos api --port 8765          # 只读：写操作需要显式 principal，因此不暴露在 HTTP 上
researchos paper voice              # 真实研究叙事线 + 研究者本人的声音指纹
```

不变量测试就是产品本身，而不是产品的证明。它断言（部分）：agent 不能修改原始证据；调试任务不能重新定义核心问题；写作者不能编造数字；`HYPOTHESIS → SUPPORTED` 被拒绝；`UNKNOWN` 永远不会变成 `FALSE`；外部技能不能修改研究状态；技能升级不能破坏基准；被改动的 artifact 会被重新哈希发现；被否决的 claim 永不消失。

---

## 状态与诚实的限制

* **已实现并有测试：**内核（状态、权限、状态转换、provenance、事件日志、冲突）、数据模型、Research Import、Evidence OS、Claim OS（生命周期 + 语言校准）、Literature OS（provider、查询规划、覆盖率、文献图、先验矩阵、新颖性审计）、分析（统计、统计审计、机制审计）、论文编译器与四道 grounding gate + 文风审计 + 就绪度、agent、完整 CLI、可选 LLM 适配器、只读 HTTP 接口、研究时间线与红队闭环、可执行的技能进化闭环、研究者声音。
* **未实现：**Web UI（按设计要求后置）、超出文本抽取的深度 PDF 解析、容器级可复现、SQLite 读取索引、技能合成的质量基准（门控与闭环已有，尚无公开合成基准）、超出 commit subject 的自动 git 历史挖掘。
* **已知限制：**导入启发式是确定性的，因此偏保守（宁可漏抽也不编造）；文风审计基于规则，会漏掉新的 AI 措辞；语言校准可能过度弱化，而不是选取"该证据等级下仍合法的最强措辞"；哈希链在没有外部锚点的情况下无法检测**尾部截断**（请把 head digest 提交进 git）。

*ResearchOS 不会让研究变正确。它让研究方向、证据与 claim 变得可审查 —— 并拒绝让语言模型决定什么是真的。*

---

许可：[Apache-2.0](LICENSE)
