# Storyline: SciIR - Bridging Generative AI and Scientific Truth via Semiotic Reasoning

## 1. Introduction: The Semiotic Gap in Generative AI
### 1.1 背景与现状 (Context)
*   **Aesthetic vs. Truth:** 现有的 T2I（文本生成图像）模型（如 Midjourney, Stable Diffusion）在艺术美学和通用领域取得了惊人的效果。
*   **The "Hallucination" Barrier:** 在科学领域，图像不仅仅是视觉信号，更是科学知识的**符号化表征（Semiotic Representation）**。现有模型常犯“科学性幻觉”错误（如错误的分子结构、违背物理定律的示意图、混乱的实验流程）。
*   **缺失的基准:** 缺乏一个大规模、高质量、且经过“科学推理”对齐的数据集和评估标准，阻碍了 AI 在科研教育和跨学科应用中的落地。

### 1.2 核心动机 (Motivation)
*   我们需要一个不仅能画出“像”科学图片，更能理解其背后“科学逻辑”的模型。
*   **理论切入点:** 引入**查尔斯·桑德斯·皮尔士（C.S. Peirce）的符号学三元论**。AI 生成科学图像的过程，本质上是构建**符号（Sign）**、指涉**对象（Object）**并生成**解释元（Interpretant）**的过程。

### 1.3 贡献 (Contributions)
1.  **SciIR Dataset:** 首个包含 100k+ 高质量样本的科学图像推理数据集，源自顶级期刊。
2.  **Theoretical Framework:** 基于符号学三元论划分的三大推理能力维度（Iconic Structure, Indexical Process, Symbolic Law）。
3.  **Automated Pipeline:** 提出“多模型协作”的数据构建与标注流水线（Reasoning → RCoT → Prompt）。
4.  **Sci-Eval Benchmark:** 结合 FID等常规指标，建立基于“关键点核查表（Checklist）”的 VLM 自动化评估体系。

---

## 2. Theoretical Framework: Mapping Science to Semiotics
*本章建立论文的理论地基，论证为何将任务划分为三大赛道。*

我们将科学图像生成视为一种**符号表意（Semiosis）**活动，根据符号与对象的关联方式，定义 SciIR 的核心能力维度：

1.  **像似性 (Iconicity) $\rightarrow$ 实体结构 (Entity Structure)**
    *   *理论:* 符号通过“相似性”指涉对象。
    *   *任务:* 准确重建科学实体（如分子、细胞、仪器）的空间拓扑、几何形态和层次结构。
2.  **指示性 (Indexicality) $\rightarrow$ 科学过程 (Scientific Process)**
    *   *理论:* 符号通过“因果或物理关联”指涉对象。
    *   *任务:* 准确表达时间演化、状态跃迁、因果链条（如反应过程、气象演变）。
3.  **规约性 (Symbolicity) $\rightarrow$ 科学定律 (Scientific Law)**
    *   *理论:* 符号通过“法则或约定”指涉对象。
    *   *任务:* 遵守抽象的科学原理（如能量守恒、化学键规则）以及正确渲染科学符号、单位与图注。

---

## 3. Methodology: Constructing the Semiotic Data Factory
*本章详细介绍如何自动化构建这个 100k 规模的数据集。*

### 3.1 数据获取与预处理 (Data Acquisition & Preprocessing)
*   **Source:** 爬取 *Nature* / *Nature Communications* 等顶级期刊（CC BY 4.0），确保科学知识的权威性。
*   **Visual Processing:**
    *   使用 **YOLO11** 进行子图分割（将多面板图拆解为单原子图）。
    *   标准化填充（Padding）至 1024x1024，并通过 InternVL3.5 剔除低质量图像。

### 3.2 符号化推理标注流水线 (Semiotic Reasoning Annotation Pipeline)
为了让模型学会“推理”而非简单的图文对齐，我们设计了三阶段标注法：

*   **Stage 1: Structured Reasoning Extraction (The Logic)**
    *   利用 Qwen/InternVL 根据图像和论文原文，提取结构化 JSON。
    *   *分类提取:* 明确区分 `Terms` (名词) 和 `Visualization` (视觉特征)，分别对应 Law, Structure, Process 三个维度。
*   **Stage 2: Sci-RCoT Generation (The Narrative)**
    *   **Scientific Reasoning Chain-of-Thought:** 将 JSON 转化为连贯的、可视化的场景描述，填补逻辑与视觉之间的鸿沟。
*   **Stage 3: Prompt Refinement (The Instruction)**
    *   生成用于训练和测试的最终 Prompt，确保术语准确替换描述性语句。

---

## 4. Benchmark Design: The Sci-Eval System
*本章介绍如何验证模型的“科学图像生成能力”。*

### 4.1 评估维度 (Evaluation Tracks)
基于理论框架的五个赛道：
*   **Track 0:** 图像质量 (基础)
*   **Track 1 (Symbol):** 科学定律 (一致性与原理)
*   **Track 2 (Icon):** 实体结构 (空间与拓扑)
*   **Track 3 (Index):** 科学过程 (时间与因果)
*   **Track 4:** 文本渲染 (准确性与规范)

### 4.2 能力-难度矩阵 (Capability-Difficulty Matrix)
*   **Prompt 构建策略:**
    *   *Simple Prompt:* 仅包含单一维度描述。
    *   *Complex Instruction (CoT):* 混合多个维度（如“画一个符合能量守恒(Law)的电致变色(Process)器件结构(Structure)”）。
*   **Difficulty Level:** 基于 Terms 数量和指令复杂度分为 High/Low 两档。

### 4.3 评估方法: VLM-based Checklist
*   **超越像素级对比:** 科学图像的正确性不仅仅在于像素距离（如 FID），还在于关键特征的存在与否。
*   **Evaluator:** 使用 Gemini 3 Pro 作为裁判。
*   **Protocol:** 为每个测试样本生成动态 **Checklist**（例如：“线粒体是否有双层膜？”，“X轴单位是否正确？”），计算**命中率 (Hit Rate)**。

---

## 5. Experiments & Analysis (预设实验)
*本章展示 SciIR 的有效性和现有模型的不足。*

### 5.1 实验设置
*   **Baselines:** Stable Diffusion 3, DALL-E 3, Midjourney v6, Flux.
*   **Fine-tuned Models:** 基于 SciIR 数据集微调的小型模型。

### 5.2 结果分析 (Expected Observations)
*   **General Models Fail on "Symbol" & "Index":** 通用大模型可能在 Track 2 (结构) 上表现尚可（因为见过类似图），但在 Track 1 (定律) 和 Track 3 (过程) 上严重“幻觉”（即符号无法正确指涉对象）。
*   **Effectiveness of Sci-RCoT:** 证明经过 Reason-CoT 训练的模型，在科学逻辑的一致性上显著优于仅使用 Caption 训练的模型。
*   **Difficulty Gap:** 高难度 prompt 下，所有模型性能显著下降，证明 SciIR 作为 Benchmark 的区分度。

---

## 6. Conclusion
*   **Summary:** SciIR 是连接 AI 想象力与科学严谨性的桥梁。
*   **Impact:**
    *   通过**皮尔士符号学**的视角，我们重新定义了科学图像生成的评估标准。
    *   提供了一套可复用的自动化数据工厂方案。
    *   未来展望：助力科研插图自动生成、科学教育可视化及多模态科学发现。

---

### 附录：核心概念映射表 (Key Concept Mapping)

| Semiotic Type | Scientific Dimension   | Key Characteristic | Evaluation Focus |
| :--- |:-----------------------| :--- | :--- |
| **Icon (像似)** | **Entity Structure**   | Similarity, Topology | Shape, Layering, Spatial Relation |
| **Index (指示)** | **Scientific Process** | Causality, Time | State Change, Arrow Logic, Sequence |
| **Symbol (规约)** | **Scientific Law**     | Convention, Rule | Principles (e.g., Conservation), Labels, Units |