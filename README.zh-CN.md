# The Origins of Stochasticity — LLM Uncertainty Quantification

[English](README.md) | **简体中文**

本项目是论文 **[The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models](https://arxiv.org/abs/2606.22792)** 对应的代码仓库，提供大语言模型不确定性量化（Uncertainty Quantification，UQ）的回答生成、方法运行、指标评估与规模效应分析工具。

> Xiang-Jun Ou, Shuang Liang, Xin-Yu Hu, Rong-Hao Huang, Jing Wang, Shao-Qun Zhang. arXiv:2606.22792, 2026.
>
> [论文摘要](https://arxiv.org/abs/2606.22792) · [论文全文](https://arxiv.org/html/2606.22792v1) · [PDF](https://arxiv.org/pdf/2606.22792)

## 项目简介

论文从输入、模型参数、词元分布和解码过程四个层面分析 LLM 的不确定性来源，将 UQ 方法归纳为贝叶斯、集成、共识与单次前向方法，并从排序、校准和覆盖三个角度开展评估。实验涵盖三个模型家族、三个基准数据集、21 种 UQ 方法和两种生成设置，同时研究模型规模与不确定性之间的关系。

本仓库围绕上述实验提供以下模块：

- **回答生成**：使用本地模型或 DeepSeek API 生成贪心回答与多次随机采样，保存正确性标签、词元概率及推理信息。
- **不确定性估计**：通过统一入口选择模型家族、数据集、生成设置和 UQ 方法。
- **统一评估**：计算 AUROC、AUPRC、AUARC、AURC、ECE、MCE 和 Brier，并支持 bootstrap 统计。
- **规模效应分析**：根据逐样本明细汇总各模型的准确率与平均原始 UQ 分数。

## 论文内容与主要发现

### 研究动机与问题

同一个问题可能得到不同的答案；即使最终答案相同，模型给出的中间推理也可能不同。因此，评估 LLM 的可信度既需要观察答案是否正确，也需要理解生成过程中的随机性，以及不确定性分数能否识别错误。论文围绕三个问题展开：如何建立统一的 UQ 评估框架，不同方法在不同任务与生成设置下表现如何，以及模型规模与输出不确定性是否存在关联。

传统的偶然不确定性与认知不确定性二分法提供了概念基础，但难以直接定位 LLM 生成链路中的具体来源。论文进一步沿生成流程分析不确定性，并将来源、量化方法和实际评估条件联系起来。

### 方法发展与分类

![图 1：不确定性量化方法的发展时间线](docs/figures/figure-1-method-timeline.svg)

*图 1．代表性 UQ 方法的发展时间线，按贝叶斯、集成、共识和单次前向四类组织。时间线展示的文献范围大于本仓库的 21 种实验方法。*

| 方法类别 | 基本思路 | 使用时关注的条件 |
| --- | --- | --- |
| 贝叶斯方法 | 通过参数分布或后验近似刻画模型的不确定性 | 参数访问、后验近似与计算开销 |
| 集成方法 | 比较多个模型或多个参数实例的预测差异 | 多模型训练、存储和推理成本 |
| 共识方法 | 利用多次生成之间的语义一致性、聚类或图结构估计不确定性 | 采样次数、语义比较方式与辅助模型 |
| 单次前向方法 | 从一次生成的概率、词元统计或自评信号中提取分数 | 可获得的内部信息，以及额外打分需求 |

论文还从模型可访问性、是否需要训练、资源消耗和评估目标等维度比较方法。实际选择方法时，需同时考虑其估计能力和部署条件。

### 四类不确定性来源

![图 2：LLM 生成过程中的四类不确定性来源](docs/figures/figure-2-uncertainty-sources.png)

*图 2．从输入，经模型前向计算得到词元分布，再由解码得到输出；四个环节分别对应不同的不确定性来源。*

| 来源 | 发生位置 | 直观解释 |
| --- | --- | --- |
| 输入不确定性 | 输入及其表达 | 问题歧义、上下文缺失或提示表达变化可能影响预测 |
| 参数不确定性 | 模型参数 | 不同参数实例或参数估计的不确定性会带来不同的预测分布 |
| 词元级不确定性 | 下一词元概率分布 | 在固定输入与参数下，多个候选词元仍可能具有相近概率 |
| 解码不确定性 | 从分布到输出序列 | 采样与解码策略会使同一模型产生不同的完整回答 |

这一划分强调：观察到的回答差异可能来自生成链路中的不同环节。词元概率分散、答案语义不一致和模型参数差异提供的是不同层面的信号，不能仅凭一个分数就认定其完整刻画了全部来源。

### 统一评估框架

论文同时比较直接回答（Answer-only）和推理增强（Reasoning-augmented）两种生成设置。前者关注最终输出，后者还引入中间推理轨迹，使不确定性的观察对象从答案扩展到生成过程。

评估覆盖三个互补目标：**排序**考察分数能否区分正确与错误回答，**校准**考察置信估计与经验正确率是否一致，**覆盖**考察保留回答与拒答之间的风险权衡。七项具体指标及优化方向见下方“计算评估指标”。某种方法在一个目标上表现较好，并不意味着它在其他目标或所有任务上都同样占优。

主要方法比较使用 Qwen3 的 1.7B、4B，Llama 3.2 的 1B、3B，以及通过 API 访问的 DeepSeek-V3，覆盖知识问答、数学推理与代码生成。规模效应实验进一步比较不同参数规模，详见图 5 和论文第 3.4 节。

### 直接回答设置下的整体排名

![图 3：直接回答设置下 UQ 方法的整体评估排名](docs/figures/figure-3-answer-only-ranking.svg)

*图 3．Answer-only 设置下的整体评估排名。图中数值为汇总后的平均名次，数值越小表示整体排名越靠前，并非准确率或某一项指标的原始分数。*

Deg 和 EigV 的平均名次分别为 **4.99** 和 **5.29**，在该设置下位居前列；MaxSeqProb 和 NumSet 分别为 **6.60** 和 **6.96**。结果支持利用多次回答之间的共识与结构信息评估可信度，同时也说明简单的概率基线仍有比较价值。

整体排名用于概括跨实验条件的表现。选择具体任务的方法时，还应查看论文中的逐模型、逐数据集指标，避免把平均名次理解为每个任务上的固定优劣关系。

### 推理增强设置下的整体排名

![图 4：推理增强设置下 UQ 方法的整体评估排名](docs/figures/figure-4-reasoning-ranking.svg)

*图 4．Reasoning-augmented 设置下的整体评估排名，平均名次越小越好。该设置纳入了利用推理信息的 CoT-UQ 和 Topo-UQ 等方法。*

Deg 和 EigV 仍位居前列，平均名次分别为 **4.51** 和 **5.18**；ConU、NumSet 和 Semantic Density 分别为 **5.67**、**6.29** 和 **6.92**。结合图 3 可见，方法的相对表现会随生成设置发生变化。

引入推理轨迹提供了额外的分析信息，但并不保证某种推理专用方法获得更好的整体排名。论文据此强调，应在目标模型、任务与生成设置上验证 UQ 方法，而不能仅根据方法是否利用推理过程来判断其效果。两张图的参评方法数量不同，其平均名次也应在各自设置内解读。

### 模型规模与输出不确定性

![图 5：不同参数规模下输出不确定性与准确率的关系](docs/figures/figure-5-scaling-law.svg)

*图 5．不同参数规模下的准确率与平均输出不确定性。左列为推理增强设置，右列为直接回答设置；四行依次为 Semantic Entropy、EigV、NumSet 和 BSDetector。*

图中横轴表示准确率，纵轴表示平均不确定性；颜色区分模型家族，形状区分数据集，圆圈大小表示模型参数规模，虚线连接同一模型家族、同一数据集中的不同规模模型。

论文观察到，较大模型通常具有更高的准确率和更低的不确定性估计，呈现出向右下方移动的总体趋势，由此提出 LLM 不确定性具有经验性规模效应。图中也存在随任务、设置和方法变化的局部差异，因此应将这一结果理解为实验中的总体观察，而非对任意模型、任意 UQ 方法都成立的严格定律。不同方法的原始分数尺度也不宜直接横向比较。

本仓库的 `scaling_law/` 提供相应逐样本明细和分组汇总工具，可用于检查准确率与平均原始 UQ 分数之间的关系。

以上图 1–5 均来自[对应论文的 arXiv v1 全文](https://arxiv.org/html/2606.22792v1)，本仓库保存原始 SVG/PNG 资源以便直接展示。图片资源及来源映射见 [docs/figures/README.md](docs/figures/README.md)。

## 实验范围

| 维度 | 内容 |
| --- | --- |
| 模型家族 | Qwen3、Llama 3.2、DeepSeek-V3；命令行标识为 `qwen`、`llama`、`deepseek` |
| 数据集 | TriviaQA（知识问答）、GSM8K（数学推理）、HumanEval（代码生成） |
| 直接回答 | `answer_only`：仅输出最终答案 |
| 推理增强 | `reasoning_augmented`：输出分步推理与最终答案 |

当前生成入口中，DeepSeek 仅支持论文配置中的 `answer_only`。并非所有方法都适用于所有组合，具体以 `registry.json` 和 `python run_uq.py --list` 为准。

21 种方法的统一标识如下：

| 方法组 | 标识 |
| --- | --- |
| 概率与熵 | `max_seq_prob`、`max_tok_prob`、`mean_tok_ent`、`ppl` |
| 语义与图结构 | `num_set`、`eigv`、`deg`、`ecc`、`semantic_entropy`、`semantic_density` |
| 检查、共识与自评 | `bsdetector`、`selfcheckgpt`、`conu`、`mars`、`p_true`、`verbalized_probability` |
| 推理相关 | `cot_uq_me`、`cot_uq_mi`、`cot_uq_sar`、`cot_uq_sp`、`topo_uq` |

上表用于索引代码，论文中的方法分类与理论定义请参阅正文。

## 目录结构

```text
.
├── run_uq.py                  # UQ 统一入口
├── registry.json              # 方法、模型、任务、设置与实现的映射
├── requirements.txt           # 主要运行依赖
├── docs/figures/              # 论文图 1–5 与图片来源说明
├── data/                      # 数据目录（原始基准需自行准备）
├── data_generation/            # 回答生成、输入规范与正确性评估
├── UQ-methods/                 # 各组合的方法脚本及历史近似版本
│   └── lm_polygraph_backend/  # LM-Polygraph 适配与独立依赖
├── uq_runtime/                # 路由、后端执行与结果整理
├── metric_evaluation/         # 七项评估指标及字段映射
└── scaling_law/
    ├── detail/                # 已提供的逐样本 CSV 明细
    ├── summary/               # 已提供的汇总 CSV
    └── summarize_scaling_law.py
```

## 环境安装

建议使用 Python 3.11 创建独立环境，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

本地模型运行需自行准备模型权重，并根据 CUDA 环境选择相应的 PyTorch 构建。部分方法还需要 NLI、句向量或其他辅助模型，相关参数见对应脚本。仅进行指标评估时，可单独安装：

```bash
python -m pip install -r metric_evaluation/requirements.txt
```

LM-Polygraph 后端建议使用另一个独立虚拟环境，安装其固定版本依赖：

```bash
python3 -m venv .venv-polygraph
source .venv-polygraph/bin/activate
python -m pip install --upgrade pip
python -m pip install -r UQ-methods/lm_polygraph_backend/requirements.txt
```

## 使用流程

### 1. 准备数据

生成入口读取本地 JSON 数组或 JSONL 文件，也支持 gzip 压缩。每条记录需包含问题与参考答案；建议提供唯一的 `id`。例如 GSM8K JSONL：

```json
{"id":"example-1","question":"What is 2 + 3?","answer":"5"}
```

常用输入字段包括：

- 问题：`question`、`prompt`、`problem`、`input` 或 `query`。
- 参考答案：`answer`、`answers`、`gt`、`ground_truth` 等；支持答案列表或含 `value`、`aliases` 的对象。
- HumanEval：使用 `prompt` 提供题目，并包含 `test` 和 `entry_point`，可附带 `canonical_solution`。

数据集路径与模型路径均需替换为实际值。项目不会自动下载原始基准。

### 2. 生成模型回答

以下命令使用本地 Qwen3 模型生成 GSM8K 回答：

```bash
python -m data_generation.generate \
  --family qwen \
  --model-path /path/to/Qwen3-model \
  --dataset gsm8k \
  --setting answer_only \
  --input-file data/gsm8k.jsonl \
  --output-file outputs/qwen_gsm8k_stage1.jsonl \
  --num-samples 10 \
  --device cuda
```

每道题生成一个贪心回答和 `--num-samples` 个额外随机回答。输出除 stage1 JSONL 外，还包括同名的 `.meta.json`、`.summary.json` 和 `.correct.jsonl`，分别记录运行配置、汇总信息与正确性标签。

将 `--setting` 改为 `reasoning_augmented` 可生成推理增强回答；推理或代码任务可增大 `--max-new-tokens`。可使用 `--limit` 小规模试跑、`--dry-run` 检查输入与配置、`--resume` 按原配置续跑。

DeepSeek 使用 `--family deepseek`，通过 `--model-path` 指定实际 API 模型 ID，并在环境变量 `DEEPSEEK_API_KEY` 中配置密钥；可用 `--base-url` 指定接口地址。概率字段是否可用取决于 API 返回内容。

HumanEval 默认通过 Docker 执行测试，需要安装并启动 Docker；`--human-eval skip` 会跳过执行，但不会得到可用于完整正确性评估的测试标签。

### 3. 运行 UQ 方法

先查看可用组合和参数：

```bash
python run_uq.py --list
python run_uq.py --help
```

统一入口的 `--implementation` 支持：

| 选项 | 行为 |
| --- | --- |
| `auto`（默认） | 根据方法注册信息选择后端；部分方法会进入 LM-Polygraph |
| `local` | 使用本地运行器，具体脚本与版本以注册表和输出记录为准 |
| `lm-polygraph` | 显式选择 LM-Polygraph 后端 |
| `approx` | 显式选择保留的历史近似实现 |

例如，基于已生成的 stage1 数据运行本地 Semantic Entropy 路由：

```bash
python run_uq.py \
  --implementation local \
  --method semantic_entropy \
  --family qwen \
  --model-path /path/to/Qwen3-model \
  --dataset gsm8k \
  --setting answer_only \
  --input-file outputs/qwen_gsm8k_stage1.jsonl \
  --output-file outputs/semantic_entropy__qwen__gsm8k__answer_only.jsonl
```

可先添加 `--dry-run` 检查实际路由。辅助模型可通过 `--nli-model` 等选项指定；额外后端参数通过 `--` 传入。HumanEval 方法若需要外部正确性标签，可使用 `--correct-file` 指向生成阶段的 `.correct.jsonl`。

本地运行器读取已有模型回答；LM-Polygraph 后端默认 `--mode generate`，读取原始题目并重新生成回答，其输入和执行流程与 stage1 本地路径不同。使用该后端前请在对应环境中查看：

```bash
python run_uq.py --implementation lm-polygraph --help
```

历史近似脚本通常以 `_approx.py` 结尾。部分本地路由也会依据注册表选择近似脚本，因此 `local` 不代表全部为非近似实现。比较实验时，请保存实际脚本、实现类型、模型版本和采样配置；本地输出中的 `uq_run` 记录了方法、分数来源与实现信息。

### 4. 计算评估指标

```bash
python -m metric_evaluation.evaluate \
  --input-file outputs/semantic_entropy__qwen__gsm8k__answer_only.jsonl \
  --method semantic_entropy \
  --output-file outputs/semantic_entropy_metrics.csv \
  --bootstrap-runs 200
```

| 评估角度 | 指标 | 优化方向 |
| --- | --- | --- |
| 排序 | AUROC、AUPRC | 越高越好 |
| 覆盖 | AUARC / AURC | AUARC 越高越好，AURC 越低越好 |
| 校准 | ECE、MCE、Brier | 越低越好 |

评估器根据方法映射区分置信度和不确定性分数。自定义字段可使用 `--score-key` 与 `--score-type confidence` 或 `uncertainty`；外部正确性标签通过 `--correct-file` 指定。

默认使用 `minmax` 校准、10 个均匀分箱和 `legacy_trapezoid` 覆盖计算。校准转换及覆盖积分方式会影响结果，复现实验时应保持一致；相关选项见 `--help`。无效样本默认跳过，可用 `--invalid-policy error` 改为直接报错。

### 5. 汇总规模效应数据

根据仓库提供的逐样本明细生成新的汇总目录：

```bash
python -m scaling_law.summarize_scaling_law \
  --split all \
  --output-dir outputs/scaling_law
```

默认汇总 `semantic_entropy`、`eigv`、`num_set` 和 `bsdetector`，输出各数据集、生成设置、模型和方法的样本数、准确率与平均原始分数。可使用 `--methods` 选择方法，使用 `--validate-sources` 检查明细引用的源文件是否存在。

该脚本进行分组统计；解释规模效应时应结合具体方法的分数方向，并使用一致的实验配置。

## 复现说明

- 模型权重、原始基准与完整生成结果需自行准备；仓库已包含规模效应分析的 CSV 明细与汇总。
- 对比结果时保持模型版本、提示词、采样次数、温度、生成长度、辅助模型和评估配置一致。
- 不同方法对词元概率、完整分布、推理过程或多次回答的要求不同；API 输出不一定满足白盒方法的输入要求。
- 仓库同时保留多种实现与近似版本，具体支持范围和实现来源请结合 `registry.json`、实际路由与方法脚本核对。

## 引用

若本项目对您的研究有帮助，请引用对应论文：

```bibtex
@misc{ou2026originsstochasticity,
  title         = {The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models},
  author        = {Xiang-Jun Ou and Shuang Liang and Xin-Yu Hu and Rong-Hao Huang and Jing Wang and Shao-Qun Zhang},
  year          = {2026},
  eprint        = {2606.22792},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2606.22792},
  url           = {https://arxiv.org/abs/2606.22792}
}
```

## 致谢

本项目使用或适配了 LM-Polygraph、SelfCheckGPT、Hugging Face Transformers 等开源工具。相关方法的理论定义与原始工作请参阅论文参考文献及对应实现。
