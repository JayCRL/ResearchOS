# ResearchOS 使用指南（中文）

> 面向第一次使用的人。所有命令都在 Windows PowerShell 下实测过，输出为真实结果。

- [一、安装与 5 分钟试跑](#一安装与-5-分钟试跑)
- [二、用在你自己的研究上](#二用在你自己的研究上)
- [三、日常循环](#三日常循环)
- [四、你会撞到的"拒绝"](#四你会撞到的拒绝)
- [五、命令速查表](#五命令速查表)
- [六、Python API：接进你自己的脚本](#六python-api接进你自己的脚本)
- [七、常见问题](#七常见问题)
- [八、别期待什么](#八别期待什么)

---

## 一、安装与 5 分钟试跑

```powershell
# 1. 拿到代码并安装（可编辑安装，改代码即时生效）
git clone https://github.com/JayCRL/ResearchOS.git
cd ResearchOS
pip install -e ".[dev]"          # 只要核心：pip install -e .
researchos --help                # 20 个顶层命令
```

用仓库自带的示例项目（`examples/DLA`：一个包含过期审计、重复 run、日记与 README 相互矛盾的真实"脏"项目）跑一遍：

```powershell
mkdir D:\demo
researchos init D:\demo --name DLA --domain "AI/ML"
researchos import D:\ResearchOS\examples\DLA --into D:\demo
```

你会先看到 **RESEARCH IMPORT REPORT**（不是摘要，而是重建出来的研究状态）：

```
Core Research Question:
  Does where a fast weight is written back to (its originating position vs a
  shuffled position) determine how much of it survives into the next context window?
  [CONFIRMED, confidence 0.89]

Current Core Claims:   （9 条，全部为 HYPOTHESIS）
Rejected Claims:       （2 条，带否决理由的"墓碑"）
Major Experiments:     （3 个，含 matched-energy 对照与 sleep0 消融）
Known Conflicts:       （审计报告里的过期数字 vs 数据文件；重复 run）
Import Confidence: HIGH   （score 14/16，逐条列出加扣分原因）
Created objects: analysis=1, author_note=25, claim=11, conflict=5, decision=1,
                 evidence=24, experiment=3, literature_paper=6, open_question=4, ...
Human review queue: 28 item(s) awaiting a decision
```

然后：

```powershell
researchos status      --root D:\demo   # 核心问题 / claims / 前沿 / 完整性检查
researchos timeline    --root D:\demo   # 叙事线：观察→假设→实验→修正→claim
researchos claim audit --root D:\demo   # 哪些措辞超出证据 + 校准后的改写
researchos redteam     --root D:\demo   # 红队：什么能推翻它
researchos paper readiness --root D:\demo
```

**注意**：`import` 之后系统里的 claim **全部是 HYPOTHESIS**。这不是没做完，这是设计——导入永远不会"认证"任何结论。

---

## 二、用在你自己的研究上

### 第 0 步：整理材料

把要导入的东西放进一个目录（不用整理得很漂亮，乱一点更能体现价值）：

| 可以有 | 会被怎么用 |
|---|---|
| Git 仓库（推荐） | 提交历史 → 研究时间线；代码 → argparse/配置面 → 实验变量 |
| `README.md` / 笔记 / 日记 | 研究问题、claim 候选、异常、被否决的 claim、决策 |
| `configs/*.yaml`、`*.json` | 实验条件；**跨 arm 比对可自动推断"哪些条件被匹配、哪个变量被操纵"** |
| `runs/*.csv`、`*.jsonl` | 实测数字 → 自动生成描述统计；重复 run / 缺失 run / 失败 run 会被发现 |
| `logs/*.log` | 训练中途指标、失败标记 |
| `refs.bib` | 文献条目（只到摘要级，绝不假装读过全文） |
| `paper/draft.tex`、`audit/*.md` | prose 中的数字 → 与数据比对，**矛盾会被建成 Conflict** |
| PDF | 有 `pypdf` 就抽文本；抽不出来会诚实地记为 UNPARSED 待人工处理 |

### 第 1 步：初始化并导入

```powershell
cd D:\my-research            # 也可以就在项目目录里操作
researchos init . --name "MyStudy" --domain "AI/ML"     # 创建 .researchos/
researchos import .                                     # 就地导入当前目录
# 或者项目状态放别处：
researchos import D:\my-research --into D:\my-project
```

两个开关值得知道：

```powershell
researchos import . --no-approve    # 核心问题只生成"待批准的状态转换"，不自动落地（更保守）
researchos import . --max-claims 30 # 限制 claim 候选数量
```

### 第 2 步：处理复核队列（**最重要的一步**）

导入的每一条不确定内容都在队列里等你决定：

```powershell
researchos review list                    # 28 条：CLAIM_CANDIDATE / CONFLICT / DECISION_CANDIDATE / ...
researchos review show <review_id>        # 看它带来的证据（文件、行号、原文引用）
researchos review decide <review_id> ACCEPTED --note "确认"
# 可选：ACCEPTED / EDITED / REJECTED / DEFERRED
```

队列里最常见四类：

| 类型 | 含义 | 你要做什么 |
|---|---|---|
| `CLAIM_CANDIDATE` | 从材料里抽出的断言 | 确认措辞与 scope |
| `CONFLICT` | 两处来源数字不一致 | 判断哪个是当前的（系统已按信任序给出默认裁决，但**两边都保留**） |
| `DECISION_CANDIDATE` | "我们决定……" 这类句子 | 补上理由 |
| `UNPARSED_MATERIAL` | 解析不了的文件 | 导出成文本/CSV 后重跑导入 |

**核心问题**如果置信度不足，会以"占位符 + 待批准 STR"的形式出现：

```powershell
researchos task transition list                       # 看 PROPOSED 的请求
researchos task transition preview --str <str_id>     # 看它将把什么改成什么
researchos task transition approve --str <str_id> --reason "确认这就是我们的问题"
```

### 第 3 步：把研究"接着做"下去

导入只是起点。从第二天开始你用的就是下面这套。

---

## 三、日常循环

### 干活前：开一个有边界的任务

这是整套系统的防漂移核心。**调试一个问题不等于新的研究主线**：

```powershell
researchos task create "解释 seed-3 上的指标异常" `
  --purpose DEBUG --priority EXPLORATORY --stop "解释清楚，或上报给人决定"
```

再看一眼这个任务被允许做什么：

```powershell
researchos task run <task_id>
```

它会打印该任务的**上下文边界**（objective / allowed_actions / forbidden_actions / may_request_core_change / stop_condition）。这就是"该给 agent 看什么"的答案——不是整个聊天历史。

干完记录发现、然后关掉：

```powershell
researchos task run <task_id> --finding "异常来自数据顺序，两个 arm 看到的语料不同" `
  --kind ANOMALY --suggests-core-change
researchos task run <task_id> --close "已定位为数据顺序问题"
```

`--suggests-core-change` 只会**记录一个建议**。想真正改核心问题，必须走状态转换 + 人工批准（而且 DEBUG / EXPLORATORY 任务连提交资格都没有）。

### 实验：注册 → 跑 → 登记产物

```powershell
researchos experiment register "direct vs shufwrite" `
  --model gpt2-small --dataset wikitext-103 --scale 124M `
  --seeds 0,1,2 --n 3 --control shufwrite --treatment direct `
  --matched model,dataset,optimizer,learning_rate `
  --intervention --config runs\direct.yaml --claim <claim_id>

researchos experiment run <exercise_id> --artifact runs\metrics.csv
```

注册时它会直接告诉你**这个设计最多能支撑多强的结论**，以及缺什么：

```
ok experiment exp_... registered; design ceiling L4_INTERVENTION
warn design gaps (these cap the claim you can make):
  - no seeds recorded: single-run results are not reproducible evidence
  - metrics without definitions: ['retention_at_1']
```

### 证据：验证靠"重新哈希"

```powershell
researchos evidence list
researchos evidence verify <evidence_id>        # 重新哈希文件并核对 provenance
researchos evidence trace <claim_id>            # claim → evidence → analysis → experiment → artifact → code
```

文件被改动过？它会变成 `HASH_MISMATCH`，并**生成一条阻断性 Conflict**——不是一条被忽略的警告。

### claim：一级一级往上走

```powershell
researchos claim propose "write position 决定 retention" --scope "gpt2-small, 124M, wikitext-103" --evidence <evd_id>
researchos claim transition <claim_id> HYPOTHESIS --reason "已声明 scope"
researchos claim transition <claim_id> TESTED     --reason "实验已运行"
researchos claim transition <claim_id> SUPPORTED  --reason "L2 证据 + 分析产物"   # 需要人类 principal
researchos claim list
researchos claim audit          # 措辞是否超出证据 + 给出校准后的写法
```

`claim audit` 的实际输出（导入的 DLA 项目）：

```
claims whose wording exceeds their evidence (with the calibrated rewrite):
  - "Our novel framework demonstrates the mechanism."
      -> "Our novel framework may bear on the mechanism."
  - "Sleep-style consolidation is necessary for retention beyond one window."
      -> "Sleep-style consolidation may be related to retention beyond one window."
  - "This establishes the mechanism."
      -> "This is consistent with a role for the mechanism."
```

### 审计：三项

```powershell
researchos audit stats                 # 统计审计：n/CI/effect size、seed 一致性、重复 run、多重比较
researchos mechanism                   # 机制审计：语言强度 vs 证据阶梯
researchos audit log verify            # 事件日志哈希链完整性
```

机制审计在 DLA 项目上直接给出 BLOCKER：

```
MECHANISM audit -> FAIL   6 finding(s), 5 blocking
CAUSAL_LANGUAGE_WITHOUT_INTERVENTION  claim ... speaks at INTERVENTION but its evidence is ...
MECHANISM_CLAIM_OVERREACH             ...
```

### 文献与新颖性

```powershell
# 联网检索（arXiv / Semantic Scholar / Crossref / OpenReview / GitHub / web）
researchos literature search "fast weight writeback alignment retention" --limit 5

# 只用本地（离线）：离线时覆盖率会诚实地显示不足
researchos literature search "write placement" --offline --bib refs.bib

researchos literature map               # 文献图：聚类、最接近的先行工作
researchos literature prior-art         # 三值先验矩阵（? 永不自动变成 "no"）
researchos novelty audit "nobody has compared placement under a matched-energy control"
```

实测联网检索结果：

```
planned 16 queries across 7 families
ok 16 queries executed; 104 papers screened
coverage score 1.00 — 16 queries across 6 families via 5 providers; meets thresholds
```

**一个真实的教训**：查询词太泛（"fast write memory retention"）会把硬件存储器领域的论文拉进来。查得"广"不等于查得"准"——先验矩阵和新颖性结论只能反映**你实际搜过的范围**，所以领域术语要写准。

### 项目自我解释与红队

```powershell
researchos timeline                        # 叙事线（导入的日记会带上真实日期）
researchos timeline summary                # 统计
researchos timeline ask --question "why was this control added?"
researchos timeline ask --question "why was the original claim cancelled?"
researchos timeline claim <claim_id>        # 一条 claim 的完整历史（含理由）
researchos redteam                          # 红队报告：会存到 .researchos/paper/audits/
```

### 论文：先看就绪度，再编译

```powershell
researchos paper readiness     # 7 个维度分别打分，故意没有总分
researchos paper compile --out paper.md
researchos paper audit         # AI 文风 / overclaim 审计
researchos paper voice         # 真实研究叙事线 + 你自己的声音指纹
```

`paper readiness` 实测（DLA 项目）：

```
Evidence Completeness      0.00  0/11 claims are promotable to paper language with at least one evidence id
Statistical Completeness   1.00  NOT EVALUATED: no claim has a linked analysis artifact (诚实标注"未评估")
Mechanism Evidence         0.64  highest evidence rung reached is L1_OBSERVATION
Literature Coverage        0.00  coverage_score 0.00 (0 queries, 6 papers screened, 0 full text)
Claim Grounding            0.00  0/11 claims resolve evidence + experiment
```

编译时如果证据不足，它**拒绝写 Results**，而不是编一段像样的结果：

```
ok   sections: 10 · words: 642
     grounding: 10 numbers checked (0 unmatched), 0/30 load-bearing sentences, 0 blocking violations
warn the compiler refused to write:
  - RESULTS: refused to write a results section because no claim is SUPPORTED yet
  - RELATED_WORK: refused to name prior work because no literature claim has a locator
```

### 每周/每次提交前

```powershell
researchos status                # 看 integrity issues
researchos audit log verify      # 哈希链是否完整
python -m pytest -q              # 183 个测试
```

---

## 四、你会撞到的"拒绝"

这些**不是 bug**，是系统的主要价值。它们都会告诉你怎么修：

| 你会看到 | 含义 | 怎么办 |
|---|---|---|
| `refusing a silent change to guarded research state: core_question …` | 想直接改核心问题 | 走 `task transition request` → 人工 `approve` |
| `a core-state transition request must be attached to a task` | 想做"无主的"方向变更 | 先建 PRIMARY 任务，再带 `--task` 提交 |
| `HYPOTHESIS -> SUPPORTED is not a legal claim transition` | 想跳级 | 先 `TESTED`，补齐证据/分析产物/冲突清零 |
| `SUPPORTED requires evidence at L2_REPRODUCED or above` | 证据等级不够 | 补 reproduced / controlled 级实验 |
| `unresolved conflicts block promotion` | 有未解决矛盾 | `researchos conflict` 相关命令或人工裁决 |
| `refusing to delete … research records are retained` | 想删记录 | 用 `SUPERSEDED` / 改状态，历史不会被抹掉 |
| `UNKNOWN -> FALSE requires explicit_absence_evidence=True` | 想把"不知道"写成"没有" | 找到原文明确说"没有"的位置再填 |
| `numeral '31.7' … has no analysis artifact behind it` | 论文里出现无依据数字 | 补跑分析产出 artifact，或删掉这个数字 |
| `cannot activate … regression against the incumbent failed` | 技能升级破坏了既有基准 | 修好回归，或拒绝这次升级 |
| `principal 'paper_writer' lacks capability 'claim.propose'` | 写作 agent 想自己造 claim | 由 claim_manager 提，人类批准 |

---

## 五、命令速查表

```powershell
# 项目
researchos init <dir> [--name N --domain D]
researchos import <src> [--into DIR --no-approve --max-claims N]
researchos status | dashboard | api --port 8765

# 任务与状态转换
researchos task create "<objective>" --purpose P --priority Q --stop "<cond>" [--touched-core]
researchos task list [--open]
researchos task run <task_id> [--finding "..." --kind K --suggests-core-change --close "summary"]
researchos task transition request|approve|reject|list|preview ...

# 实验 / 证据 / claim
researchos experiment register|list|run ...
researchos evidence list|verify|trace ...
researchos claim list|propose|transition|audit ...

# 文献 / 新颖性
researchos literature search|map|prior-art ...
researchos novelty audit "<claim>"

# 审计 / 时间线 / 红队
researchos audit stats | audit log verify | audit log tail
researchos mechanism [--claim ID]
researchos timeline [show|summary|claim|why|ask]
researchos redteam [--claim ID]

# 技能
researchos skill list|search|evaluate|benchmark|install|update|evolve ...

# 论文
researchos paper compile|audit|readiness|voice

# 其他
researchos agent list           # 15 个 agent 的能力与"不许做什么"
researchos review list|show|decide
```

---

## 六、Python API：接进你自己的脚本

```python
from researchos.kernel import ResearchKernel

k = ResearchKernel.open("D:/my-project")        # 自动向上查找 .researchos/
print(k.research_state().core_question.statement)

# 建一个有边界的任务
task = k.task_manager.create(
    k.principal("planner"),
    objective="检查 metric 定义是否一致",
    purpose="AUDIT", priority="SECONDARY",
    allowed_actions=["audit.create", "analysis.read"],
    stop_condition="定义一致或列出不一致项",
)

# 记录发现（不会碰研究状态）
k.task_manager.add_finding(k.principal("planner"), task.task_id,
                          statement="retention 在 eval 与 train 上定义不同")

# 让 agent 干活：它们只提议，内核负责提交
from researchos.agents import build_agents
agents = build_agents(k)
proposal = agents["claim_manager"].request_approval("<claim_id>", reason="义务看起来已满足")
print(proposal.describe())
```

只读 HTTP 接口 + 看板（给看板 / 笔记本 / 评审人用）：

```powershell
pip install -e ".[api]"
researchos api --root D:\my-project                  # → http://127.0.0.1:8765/
researchos api --root D:\my-project --enable-actions # 额外启用"人在回路"按钮
```

看板是**单个 HTML 文件，零构建、零 npm、零 CDN**（离线可用、五年后仍能打开）。它显示规范要求的全部面板：
核心问题、claim（含其证据允许的措辞）、被否决的 claim、证据覆盖、实验与设计缺口、冲突与信任序、文献图与
最接近先行工作、新颖性结论、开放问题、研究决策、当前任务、技能健康与缺口、研究时间线，以及**论文就绪度的
7 个独立维度（无总分）**。

**它是视图，不是第二权威。**写操作默认关闭；`--enable-actions` 开启后也只在**本机客户端**可用，并且以
human principal 身份走**与 CLI 完全相同的 kernel 门禁**：

| 看板动作 | 仍然会被拒绝的情况 |
|---|---|
| 复核队列 accept / reject / defer | 未知 decision 值 → 422 |
| 批准已提交的状态转换 | 请求不存在 → 409；状态已变化（STALE）→ 409，且记录为 STALE |
| 拒绝状态转换 | 非 human 能力不足 → 拒绝 |
| 重新哈希验证证据 | 文件被改动 → `HASH_MISMATCH` 并生成阻断性 Conflict |

每次点击都会写入 `ui.action` 事件（含动作、客户端地址、origin=dashboard），因此日志里能区分"人点的"和
"命令行执行的"。其余所有变更仍留在 CLI——因为 CLI 里"谁在改"是显式的。

把 `researchos` 接进 CI：

```powershell
researchos audit log verify    # 事件日志被改动就失败
researchos status              # integrity issues 会显示
python -m pytest -q
```

---

## 七、常见问题

**我只想要一个"研究状态看板"** → `researchos status` / `dashboard`，或 `api` + 你自己的前端。不做其他动作也可以。

**我不想联网** → 检索加 `--offline --bib refs.bib`；覆盖率会诚实显示不足，此时新颖性结论必然是 `NOVELTY_UNCERTAIN`。

**我的项目不是 git 仓库** → 完全可以。git 信息只是 provenance 的一部分，缺失会被明确标出（`experiment provenance_missing`），不会伪造。

**我有 PDF 论文** → 装 `pypdf`（已随 dev 依赖）。抽不出文本（扫描件）会记为 `UNPARSED_MATERIAL` 进复核队列，而不是编造摘要。

**如何增加 provider** → 实现 `literature/providers/base.py` 的接口（`search` / `fetch` / `expand`），用 `ProviderRecord` 返回即可；不在核心架构里写死任何 provider。

**如何调 LLM** → `researchos/llm/`。它只做"提议"（查询扩展、假设候选、语言改写），且改写文本里数字必须与原文完全一致，否则整条回复被丢弃。真值路径（模型/内核/证据/claim/分析/论文门禁）不导入它——有测试守着。

**数据存在哪** → 全部在 `.researchos/` 下的 YAML/JSONL 纯文本，可 git diff、可人工编辑、可离线查看。SQLite 只作为可重建缓存，永远不是真值来源。

---

## 八、别期待什么

- **不保证论文正确**，不保证新颖性，不证明机制，不保证实验可复现。它提高可追溯性、减少幻觉与上下文漂移、强化因果与机制纪律、减少 overclaim——并把不确定性显示出来。
- **不替你跑训练**。`experiment run` 只是登记你产出的 artifact（并哈希它），让每个数字都能回溯。
- **不替你写一篇能投的论文**。编译器的职责是"只写有依据的内容"，因此证据不足时它会拒绝写 Results——这通常是它最有用的时刻。
- **导入启发式保守**：宁可漏抽，也不编造。所以复核队列里会有东西。
- **文风审计基于规则**，会漏掉新的 AI 措辞；**语言校准可能过度弱化**（会给出安全但更弱的措辞，而不是"该等级下最强的合法措辞"）。
- **哈希链无法检测尾部截断**：请把 `researchos audit log verify` 输出的 head digest 提交进 git 作为外部锚点。
