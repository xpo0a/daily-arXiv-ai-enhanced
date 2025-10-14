<div id=toc></div>

# Table of Contents

- [cs.CV](#cs.CV) [Total: 13]
- [cs.CL](#cs.CL) [Total: 4]
- [physics.app-ph](#physics.app-ph) [Total: 1]
- [cs.IT](#cs.IT) [Total: 1]
- [stat.ME](#stat.ME) [Total: 2]
- [cs.SE](#cs.SE) [Total: 1]
- [cs.IR](#cs.IR) [Total: 1]
- [cs.LG](#cs.LG) [Total: 1]
- [math.AT](#math.AT) [Total: 1]
- [stat.ML](#stat.ML) [Total: 1]
- [cond-mat.soft](#cond-mat.soft) [Total: 1]
- [q-bio.TO](#q-bio.TO) [Total: 1]
- [cs.CR](#cs.CR) [Total: 1]
- [hep-th](#hep-th) [Total: 1]
- [hep-ph](#hep-ph) [Total: 1]
- [cs.RO](#cs.RO) [Total: 9]
- [cs.AI](#cs.AI) [Total: 1]
- [cs.MA](#cs.MA) [Total: 1]
- [cs.HC](#cs.HC) [Total: 1]


<div id='cs.CV'></div>

# cs.CV [[Back]](#toc)

### [1] [VITA-VLA: Efficiently Teaching Vision-Language Models to Act via Action Expert Distillation](https://arxiv.org/abs/2510.09607v1)
*Shaoqi Dong,Chaoyou Fu,Haihan Gao,Yi-Fan Zhang,Chi Yan,Chu Wu,Xiaoyu Liu,Yunhang Shen,Jing Huo,Deqiang Jiang,Haoyu Cao,Yang Gao,Xing Sun,Ran He,Caifeng Shan*

Main category: cs.CV

TL;DR: 提出了一种基于知识蒸馏的框架，通过将预训练小动作模型的知识转移到视觉语言模型中，为VLM赋予动作执行能力，显著降低训练成本并提升性能。


<details>
  <summary>Details</summary>
Motivation: 视觉语言动作模型虽然通过结合预训练视觉语言模型提升了机器人操作的泛化能力，但从头训练成本高昂。

Method: 采用两阶段训练策略：首先进行轻量级对齐，将VLM隐藏状态映射到小动作模型的动作空间；然后选择性微调语言模型、状态编码器和动作模块，整合多模态输入与精确动作生成。

Result: 在LIBERO数据集上达到97.3%平均成功率（提升11.8%），LIBERO-LONG上达到93.5%（提升24.5%）；真实世界实验中在5个操作任务上达到82.0%成功率（提升17%）。

Conclusion: 动作蒸馏方法能有效使VLM生成精确动作，同时大幅降低训练成本，在仿真和真实环境中均优于现有方法。

Abstract: Vision-Language Action (VLA) models significantly advance robotic
manipulation by leveraging the strong perception capabilities of pretrained
vision-language models (VLMs). By integrating action modules into these
pretrained models, VLA methods exhibit improved generalization. However,
training them from scratch is costly. In this work, we propose a simple yet
effective distillation-based framework that equips VLMs with action-execution
capability by transferring knowledge from pretrained small action models. Our
architecture retains the original VLM structure, adding only an action token
and a state encoder to incorporate physical inputs. To distill action
knowledge, we adopt a two-stage training strategy. First, we perform
lightweight alignment by mapping VLM hidden states into the action space of the
small action model, enabling effective reuse of its pretrained action decoder
and avoiding expensive pretraining. Second, we selectively fine-tune the
language model, state encoder, and action modules, enabling the system to
integrate multimodal inputs with precise action generation. Specifically, the
action token provides the VLM with a direct handle for predicting future
actions, while the state encoder allows the model to incorporate robot dynamics
not captured by vision alone. This design yields substantial efficiency gains
over training large VLA models from scratch. Compared with previous
state-of-the-art methods, our method achieves 97.3% average success rate on
LIBERO (11.8% improvement) and 93.5% on LIBERO-LONG (24.5% improvement). In
real-world experiments across five manipulation tasks, our method consistently
outperforms the teacher model, achieving 82.0% success rate (17% improvement),
which demonstrate that action distillation effectively enables VLMs to generate
precise actions while substantially reducing training costs.

</details>


### [2] [SpaceVista: All-Scale Visual Spatial Reasoning from mm to km](https://arxiv.org/abs/2510.09606v1)
*Peiwen Sun,Shiqiang Lang,Dongming Wu,Yi Ding,Kaituo Feng,Huadai Liu,Zhen Ye,Rui Liu,Yun-Hui Liu,Jianan Wang,Xiangyu Yue*

Main category: cs.CV

TL;DR: 该论文提出了一个全尺度空间推理解决方案，通过构建包含38K视频场景和约100万空间问答对的SpaceVista-1M数据集，开发了SpaceVista-7B模型，解决了依赖室内3D扫描和手动标注、缺乏有效全尺度建模的问题。


<details>
  <summary>Details</summary>
Motivation: 当前空间推理研究主要依赖室内3D扫描和人工标注，且缺乏有效的全尺度场景建模，导致模型容易过拟合到单个场景。论文旨在推进跨多样化场景的全尺度空间推理能力。

Method: 整合结构化空间推理知识系统、尺度感知建模和渐进式训练范式；使用任务特定的专家驱动自动化流水线构建数据集；开发接受密集输入并使用尺度作为锚点的SpaceVista-7B模型。

Result: 在包括SpaceVista-Bench在内的5个基准测试上进行了广泛评估，展示了在所有尺度和场景上的竞争性性能和强泛化能力。

Conclusion: 提出的解决方案成功扩展了MLLMs的全尺度空间智能，数据集、模型和基准测试将公开发布。

Abstract: With the current surge in spatial reasoning explorations, researchers have
made significant progress in understanding indoor scenes, but still struggle
with diverse applications such as robotics and autonomous driving. This paper
aims to advance all-scale spatial reasoning across diverse scenarios by
tackling two key challenges: 1) the heavy reliance on indoor 3D scans and
labor-intensive manual annotations for dataset curation; 2) the absence of
effective all-scale scene modeling, which often leads to overfitting to
individual scenes. In this paper, we introduce a holistic solution that
integrates a structured spatial reasoning knowledge system, scale-aware
modeling, and a progressive training paradigm, as the first attempt to broaden
the all-scale spatial intelligence of MLLMs to the best of our knowledge. Using
a task-specific, specialist-driven automated pipeline, we curate over 38K video
scenes across 5 spatial scales to create SpaceVista-1M, a dataset comprising
approximately 1M spatial QA pairs spanning 19 diverse task types. While
specialist models can inject useful domain knowledge, they are not reliable for
evaluation. We then build an all-scale benchmark with precise annotations by
manually recording, retrieving, and assembling video-based data. However, naive
training with SpaceVista-1M often yields suboptimal results due to the
potential knowledge conflict. Accordingly, we introduce SpaceVista-7B, a
spatial reasoning model that accepts dense inputs beyond semantics and uses
scale as an anchor for scale-aware experts and progressive rewards. Finally,
extensive evaluations across 5 benchmarks, including our SpaceVista-Bench,
demonstrate competitive performance, showcasing strong generalization across
all scales and scenarios. Our dataset, model, and benchmark will be released on
https://peiwensun2000.github.io/mm2km .

</details>


### [3] [Vision Language Models: A Survey of 26K Papers](https://arxiv.org/abs/2510.09586v1)
*Fengming Lin*

Main category: cs.CV

TL;DR: 对2023-2025年CVPR、ICLR和NeurIPS的26,104篇论文进行透明、可重复的研究趋势测量，分析了三大宏观转变：多模态视觉-语言-LLM工作的急剧增长、生成方法的稳步扩展以及3D和视频活动的韧性发展。


<details>
  <summary>Details</summary>
Motivation: 量化计算机视觉和机器学习领域的研究趋势，通过系统化分析大量顶级会议论文，揭示领域内的宏观变化和发展方向。

Method: 对论文标题和摘要进行标准化、短语保护和匹配手工制作的词典，分配最多35个主题标签，挖掘任务、架构、训练机制、目标、数据集和共提及模态的细粒度线索。

Result: 发现三大宏观转变：1) 多模态视觉-语言-LLM工作急剧增长，将经典感知重新定义为指令跟随和多步推理；2) 生成方法稳步扩展，扩散研究围绕可控性、蒸馏和速度进行整合；3) 3D和视频活动保持韧性，组合从NeRFs转向高斯溅射，人类和智能体中心理解日益重要。

Conclusion: 该研究提供了计算机视觉和机器学习领域的系统性趋势分析，揭示了多模态、生成方法和3D/视频理解的主要发展方向，并发布了词典和方法论以供审计和扩展。

Abstract: We present a transparent, reproducible measurement of research trends across
26,104 accepted papers from CVPR, ICLR, and NeurIPS spanning 2023-2025. Titles
and abstracts are normalized, phrase-protected, and matched against a
hand-crafted lexicon to assign up to 35 topical labels and mine fine-grained
cues about tasks, architectures, training regimes, objectives, datasets, and
co-mentioned modalities. The analysis quantifies three macro shifts: (1) a
sharp rise of multimodal vision-language-LLM work, which increasingly reframes
classic perception as instruction following and multi-step reasoning; (2)
steady expansion of generative methods, with diffusion research consolidating
around controllability, distillation, and speed; and (3) resilient 3D and video
activity, with composition moving from NeRFs to Gaussian splatting and a
growing emphasis on human- and agent-centric understanding. Within VLMs,
parameter-efficient adaptation like prompting/adapters/LoRA and lightweight
vision-language bridges dominate; training practice shifts from building
encoders from scratch to instruction tuning and finetuning strong backbones;
contrastive objectives recede relative to cross-entropy/ranking and
distillation. Cross-venue comparisons show CVPR has a stronger 3D footprint and
ICLR the highest VLM share, while reliability themes such as efficiency or
robustness diffuse across areas. We release the lexicon and methodology to
enable auditing and extension. Limitations include lexicon recall and
abstract-only scope, but the longitudinal signals are consistent across venues
and years.

</details>


### [4] [PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs](https://arxiv.org/abs/2510.09507v1)
*Zixin Zhang,Kanghao Chen,Xingwang Lin,Lutao Jiang,Xu Zheng,Yuanhuiyi Lyu,Litao Guo,Yinchuan Li,Ying-Cong Chen*

Main category: cs.CV

TL;DR: PhysToolBench是首个专门评估多模态大语言模型对物理工具理解能力的基准，包含1000多个图像-文本对，评估工具识别、工具理解和工具创造三个难度级别。对32个MLLM的评估显示它们在工具理解方面存在显著缺陷。


<details>
  <summary>Details</summary>
Motivation: 工具使用、理解和创造是人类智能的标志，但现代多模态大语言模型对物理工具的真正理解程度尚未量化。为了填补这一空白，需要建立一个专门的评估基准。

Method: 构建PhysToolBench基准，作为视觉问答数据集，包含1000多个图像-文本对，评估三个难度级别：工具识别（识别工具主要功能）、工具理解（理解工具操作原理）、工具创造（在常规工具不可用时用周围物体创造新工具）。

Result: 对32个MLLM（包括专有、开源、专用具身模型和VLA骨干模型）的综合评估显示，在工具理解方面存在显著缺陷。

Conclusion: 多模态大语言模型在物理工具理解方面存在明显不足，需要进一步改进。作者提供了深入分析并提出了初步解决方案。

Abstract: The ability to use, understand, and create tools is a hallmark of human
intelligence, enabling sophisticated interaction with the physical world. For
any general-purpose intelligent agent to achieve true versatility, it must also
master these fundamental skills. While modern Multimodal Large Language Models
(MLLMs) leverage their extensive common knowledge for high-level planning in
embodied AI and in downstream Vision-Language-Action (VLA) models, the extent
of their true understanding of physical tools remains unquantified. To bridge
this gap, we present PhysToolBench, the first benchmark dedicated to evaluating
the comprehension of physical tools by MLLMs. Our benchmark is structured as a
Visual Question Answering (VQA) dataset comprising over 1,000 image-text pairs.
It assesses capabilities across three distinct difficulty levels: (1) Tool
Recognition: Requiring the recognition of a tool's primary function. (2) Tool
Understanding: Testing the ability to grasp the underlying principles of a
tool's operation. (3) Tool Creation: Challenging the model to fashion a new
tool from surrounding objects when conventional options are unavailable. Our
comprehensive evaluation of 32 MLLMs-spanning proprietary, open-source,
specialized embodied, and backbones in VLAs-reveals a significant deficiency in
tool understanding. Furthermore, we provide an in-depth analysis and propose
preliminary solutions. Code and dataset are publicly available.

</details>


### [5] [SilvaScenes: Tree Segmentation and Species Classification from Under-Canopy Images in Natural Forests](https://arxiv.org/abs/2510.09458v1)
*David-Alexandre Duclos,William Guimont-Martin,Gabriel Jeanson,Arthur Larochelle-Tremblay,Théo Defosse,Frédéric Moore,Philippe Nolet,François Pomerleau,Philippe Giguère*

Main category: cs.CV

TL;DR: 提出了SilvaScenes数据集，用于从林下图像进行树种实例分割，包含24个物种的1476棵树，在五个生物气候区域收集。基准测试显示树分割相对容易（mAP 67.65%），但树种分类仍然具有挑战性（mAP 35.69%）。


<details>
  <summary>Details</summary>
Motivation: 森林管理中的机器人技术需求增长，但在复杂自然环境中感知仍然是一个重大障碍。现有数据集通常关注城市环境或物种有限，无法开发先进的感知系统来支持精准林业、生物多样性监测和林业设备自动化。

Method: 在加拿大魁北克的五个生物气候区域收集林下图像，由林业专家标注了24个物种的1476棵树，创建了SilvaScenes数据集。使用现代深度学习方法进行实例分割的基准测试。

Result: 树分割表现良好，最高平均精度为67.65%，但树种分类仍然具有挑战性，平均精度仅为35.69%。

Conclusion: SilvaScenes数据集填补了森林环境中树种实例分割数据集的空白，为开发更先进的林业感知系统提供了重要资源，同时突显了在复杂自然环境中树种分类的持续挑战。

Abstract: Interest in robotics for forest management is growing, but perception in
complex, natural environments remains a significant hurdle. Conditions such as
heavy occlusion, variable lighting, and dense vegetation pose challenges to
automated systems, which are essential for precision forestry, biodiversity
monitoring, and the automation of forestry equipment. These tasks rely on
advanced perceptual capabilities, such as detection and fine-grained species
classification of individual trees. Yet, existing datasets are inadequate to
develop such perception systems, as they often focus on urban settings or a
limited number of species. To address this, we present SilvaScenes, a new
dataset for instance segmentation of tree species from under-canopy images.
Collected across five bioclimatic domains in Quebec, Canada, SilvaScenes
features 1476 trees from 24 species with annotations from forestry experts. We
demonstrate the relevance and challenging nature of our dataset by benchmarking
modern deep learning approaches for instance segmentation. Our results show
that, while tree segmentation is easy, with a top mean average precision (mAP)
of 67.65%, species classification remains a significant challenge with an mAP
of only 35.69%. Our dataset and source code will be available at
https://github.com/norlab-ulaval/SilvaScenes.

</details>


### [6] [Mono4DEditor: Text-Driven 4D Scene Editing from Monocular Video via Point-Level Localization of Language-Embedded Gaussians](https://arxiv.org/abs/2510.09438v1)
*Jin-Chuan Shi,Chengye Su,Jiajun Wang,Ariel Shamir,Miao Wang*

Main category: cs.CV

TL;DR: Mono4DEditor是一个基于文本提示编辑单目视频重建4D场景的新框架，通过量化CLIP特征增强3D高斯表示，实现语义精确的局部编辑，同时保持未编辑内容的完整性。


<details>
  <summary>Details</summary>
Motivation: 从单目视频重建的4D场景基于文本提示进行编辑在内容创作和虚拟环境中具有广泛应用价值，但面临在复杂动态场景中实现语义精确局部编辑并保持未编辑内容完整性的挑战。

Method: 方法包括：1）用量化CLIP特征增强3D高斯形成语言嵌入动态表示；2）两阶段点级定位策略（CLIP相似度候选选择+空间范围细化）；3）基于扩散的视频编辑模型进行目标编辑，使用流和涂鸦引导确保空间保真度和时间一致性。

Result: 大量实验表明，Mono4DEditor能够在多样化场景和对象类型上实现高质量的文本驱动编辑，同时保持未编辑区域的外观和几何特性，在灵活性和视觉保真度方面超越先前方法。

Conclusion: Mono4DEditor框架成功解决了4D场景编辑中的语义精确性和局部化挑战，为文本驱动的动态场景编辑提供了有效的解决方案。

Abstract: Editing 4D scenes reconstructed from monocular videos based on text prompts
is a valuable yet challenging task with broad applications in content creation
and virtual environments. The key difficulty lies in achieving semantically
precise edits in localized regions of complex, dynamic scenes, while preserving
the integrity of unedited content. To address this, we introduce Mono4DEditor,
a novel framework for flexible and accurate text-driven 4D scene editing. Our
method augments 3D Gaussians with quantized CLIP features to form a
language-embedded dynamic representation, enabling efficient semantic querying
of arbitrary spatial regions. We further propose a two-stage point-level
localization strategy that first selects candidate Gaussians via CLIP
similarity and then refines their spatial extent to improve accuracy. Finally,
targeted edits are performed on localized regions using a diffusion-based video
editing model, with flow and scribble guidance ensuring spatial fidelity and
temporal coherence. Extensive experiments demonstrate that Mono4DEditor enables
high-quality, text-driven edits across diverse scenes and object types, while
preserving the appearance and geometry of unedited areas and surpassing prior
approaches in both flexibility and visual fidelity.

</details>


### [7] [Utilizing dynamic sparsity on pretrained DETR](https://arxiv.org/abs/2510.09380v1)
*Reza Sedghi,Anand Subramoney,David Kappel*

Main category: cs.CV

TL;DR: 本文提出了两种无需重新训练的MLP层稀疏化方法：SIBS（静态指示器稀疏化）和MGS（微门控稀疏化），用于提升DETR模型在目标检测任务中的推理效率。


<details>
  <summary>Details</summary>
Motivation: 基于Transformer的模型在视觉任务（如目标检测）中的推理效率仍然是一个挑战，特别是在DETR的MLP层中存在固有的稀疏性，需要开发方法来利用这种稀疏性而不需要重新训练模型。

Method: 提出了两种方法：1）SIBS：基于固定激活模式的启发式方法预测神经元不活跃性；2）MGS：在预训练DETR上训练的轻量级门控机制，使用小型线性层预测动态稀疏性。

Result: 在COCO数据集上的实验表明，MGS实现了85-95%的激活稀疏度，在保持甚至提升性能的同时显著减少了计算量。

Conclusion: MGS提供了一种实用的、输入自适应的稀疏化方法，能够在无需完整模型重新训练的情况下实现预训练视觉Transformer的高效部署。

Abstract: Efficient inference with transformer-based models remains a challenge,
especially in vision tasks like object detection. We analyze the inherent
sparsity in the MLP layers of DETR and introduce two methods to exploit it
without retraining. First, we propose Static Indicator-Based Sparsification
(SIBS), a heuristic method that predicts neuron inactivity based on fixed
activation patterns. While simple, SIBS offers limited gains due to the
input-dependent nature of sparsity. To address this, we introduce Micro-Gated
Sparsification (MGS), a lightweight gating mechanism trained on top of a
pretrained DETR. MGS predicts dynamic sparsity using a small linear layer and
achieves up to 85 to 95% activation sparsity. Experiments on the COCO dataset
show that MGS maintains or even improves performance while significantly
reducing computation. Our method offers a practical, input-adaptive approach to
sparsification, enabling efficient deployment of pretrained vision transformers
without full model retraining.

</details>


### [8] [BLINK-Twice: You see, but do you observe? A Reasoning Benchmark on Visual Perception](https://arxiv.org/abs/2510.09361v1)
*Junyan Ye,Dongzhi Jiang,Jun He,Baichuan Zhou,Zilong Huang,Zhiyuan Yan,Hongsheng Li,Conghui He,Weijia Li*

Main category: cs.CV

TL;DR: BLINK-Twice是一个以视觉为中心的推理基准，专注于从纯视觉内容进行推理，超越了浅层感知，需要细粒度观察和分析推理。它包含七种视觉挑战类型、自然对抗图像对和带注释的推理链。


<details>
  <summary>Details</summary>
Motivation: 现有的推理基准主要评估基于语言的推理，将视觉输入视为可替换的上下文。为了填补这一空白，需要创建一个基于挑战性感知任务的视觉中心推理基准。

Method: BLINK-Twice集成了三个核心组件：七种视觉挑战类型用于测试视觉推理，自然对抗图像对强制依赖视觉内容，以及带注释的推理链用于细粒度评估推理过程。评估了20个领先的MLLM模型。

Result: BLINK-Twice对当前模型构成了显著挑战。现有的语言空间推理策略（如思维链或自我批评）可以提高性能，但往往导致不稳定和冗余的推理。重复图像观察可以提高性能，主动视觉交互突出了视觉推理新范式的需求。

Conclusion: 该基准强调了从语言基础推理向图像基础推理的转变需求，展示了当前MLLM在视觉推理方面的局限性，并指出了需要新的视觉推理范式。

Abstract: Recently, Multimodal Large Language Models (MLLMs) have made rapid progress,
particularly in enhancing their reasoning capabilities. However, existing
reasoning benchmarks still primarily assess language-based reasoning, often
treating visual input as replaceable context. To address this gap, we introduce
BLINK-Twice, a vision-centric reasoning benchmark grounded in challenging
perceptual tasks. Instead of relying on external knowledge, our tasks require
models to reason from visual content alone, shifting the focus from
language-based to image-grounded reasoning. Compared to prior perception
benchmarks, it moves beyond shallow perception ("see") and requires
fine-grained observation and analytical reasoning ("observe"). BLINK-Twice
integrates three core components: seven types of visual challenges for testing
visual reasoning, natural adversarial image pairs that enforce reliance on
visual content, and annotated reasoning chains for fine-grained evaluation of
the reasoning process rather than final answers alone. We evaluate 20 leading
MLLMs, including 12 foundation models and 8 reasoning-enhanced models.
BLINK-Twice poses a significant challenge to current models. While existing
reasoning strategies in the language space-such as chain-of-thought or
self-criticism can improve performance, they often result in unstable and
redundant reasoning. We observe that repeated image observation improves
performance across models, and active visual interaction, as demonstrated by
models like o3, highlights the need for a new paradigm for vision reasoning.
The dataset is publicly available at https://github.com/PicoTrex/BLINK-Twice

</details>


### [9] [CapGeo: A Caption-Assisted Approach to Geometric Reasoning](https://arxiv.org/abs/2510.09302v1)
*Yuying Li,Siyi Qian,Hao Liang,Leqi Zheng,Ruichuan An,Yongzhen Guo,Wentao Zhang*

Main category: cs.CV

TL;DR: 本文提出CapGeo框架，通过将几何图形转换为文本描述来提升多模态大语言模型的几何推理能力，并构建了CapGeo-Bench数据集来评估几何描述模型的质量。


<details>
  <summary>Details</summary>
Motivation: 现有最先进的多模态大语言模型在几何推理方面表现不佳，瓶颈在于理解几何图形而非推理能力本身。由于几何图形可以用简洁的文本形式准确描述，将视觉内容转换为文本描述是一个有前景的方向。

Method: 引入CapGeo框架，通过辅助描述来桥接视觉和文本模态。同时构建了CapGeo-Bench数据集，包含4,641个精心筛选的图形-描述对，并提出了基于关键点的评估指标来可靠评估几何描述能力。

Result: 实验显示配备描述后模型性能显著提升：Qwen2.5-VL-72B从8.6%提升至59.0%，Claude-Opus-4从44.8%提升至73.0%。关键点评估指标与下游CapGeo性能强相关。

Conclusion: CapGeo框架和基准测试为推进多模态大语言模型的几何推理能力开辟了新途径，表明通过几何描述可以显著提升模型的几何问题解决能力。

Abstract: Geometric reasoning remains a core challenge for Multimodal Large Language
Models (MLLMs). Even the most advanced closed-source systems, such as GPT-O3
and Gemini-2.5-Pro, still struggle to solve geometry problems reliably, despite
exhibiting strong textual reasoning abilities on tasks like the International
Mathematical Olympiad (IMO). This gap suggests that the bottleneck lies in
understanding geometric diagrams rather than reasoning itself. Since geometric
figures can often be faithfully described in concise textual form, converting
visual content into captions offers a promising direction. Motivated by this
insight, we introduce CapGeo, a caption-assisted reasoning framework that
bridges visual and textual modalities. Experiments show substantial
improvements when models are equipped with captions: Qwen2.5-VL-72B improves
from 8.6% (vision-only) to 59.0%, while Claude-Opus-4 rises from 44.8% to
73.0%. To systematically evaluate and identify high-quality geometric
captioning models, we further propose CapGeo-Bench, a dataset of 4,641 curated
figure-caption pairs. Crucially, CapGeo-Bench incorporates a keypoint-based
evaluation metric that correlates strongly with downstream CapGeo performance,
enabling reliable assessment of geometric captioning ability. Together, our
framework and benchmark highlight a new pathway toward advancing geometric
reasoning in MLLMs.

</details>


### [10] [Spotlight on Token Perception for Multimodal Reinforcement Learning](https://arxiv.org/abs/2510.09285v1)
*Siyuan Huang,Xiaoye Qu,Yafu Li,Yun Luo,Zefeng He,Daizong Liu,Yu Cheng*

Main category: cs.CV

TL;DR: 本文提出了一种新颖的多模态强化学习方法VPPO，通过token感知视角分析视觉依赖性，在8个基准测试中显著提升了大型视觉语言模型的推理能力。


<details>
  <summary>Details</summary>
Motivation: 现有强化学习方法在多模态推理中忽视了视觉感知在优化过程中的关键作用，需要从token感知角度重新审视多模态RLVR。

Method: 提出视觉感知策略优化(VPPO)算法，通过双重机制：基于整体视觉依赖性重新加权轨迹优势，并仅对感知关键token进行策略更新。

Result: 在8个感知和推理基准测试中，VPPO相比领先的开源RL调优模型取得了显著提升，在7B和32B模型规模上均验证了有效性。

Conclusion: 研究不仅为分析多模态RLVR建立了新的token级感知视角，还提供了一种新颖有效的优化策略来显著增强LVLMs的多模态推理能力。

Abstract: While Reinforcement Learning with Verifiable Rewards (RLVR) has advanced the
reasoning capabilities of Large Vision-Language Models (LVLMs), most existing
methods in multimodal reasoning neglect the critical role of visual perception
within the RLVR optimization process. In this paper, we undertake a pioneering
exploration of multimodal RLVR through the novel perspective of token
perception, which measures the visual dependency of each generated token. With
a granular analysis of Chain-of-Thought (CoT) processes, we uncover two key
insights: first, token perception in a rollout trajectory is sparsely
distributed, where only a small fraction of tokens have high visual dependency
for visually-grounded reasoning; second, different trajectories exhibit
significant divergence in their overall visual dependency. Based on these
observations, we propose Visually-Perceptive Policy Optimization (VPPO), a
novel policy gradient algorithm that explicitly leverages token perception to
refine the learning signal. Specifically, VPPO achieves this through a dual
mechanism: it reweights a trajectory's advantage by its overall visual
dependency, and focuses policy updates exclusively on perceptually pivotal
tokens. On a comprehensive suite of eight perception and reasoning benchmarks,
VPPO demonstrates substantial gains over leading open-source RL-tuned models,
with its effectiveness consistently validated across 7B and 32B model scales.
Our findings not only establish a new token-level perceptual perspective for
analyzing multimodal RLVR but also present a novel and effective optimization
strategy to significantly enhance the multimodal reasoning capabilities of
LVLMs.

</details>


### [11] [Diagnosing Shoulder Disorders Using Multimodal Large Language Models and Consumer-Grade Cameras](https://arxiv.org/abs/2510.09230v1)
*Jindong Hong,Wencheng Zhang,Shiqin Qiao,Jianhai Chen,Jianing Qiu,Chuanyang Zheng,Qian Xu,Yun Ji,Qianyue Wen,Weiwei Sun,Hao Li,Huizhen Li,Huichao Wang,Kai Wu,Meng Li,Yijun He,Lingjie Luo,Jiankai Sun*

Main category: cs.CV

TL;DR: 该研究提出了一种基于消费级设备视频的肩部疾病诊断框架HMVDx，使用多模态大语言模型分别处理动作理解和疾病诊断任务，显著提高了肩关节损伤诊断的准确性。


<details>
  <summary>Details</summary>
Motivation: 在医疗资源稀缺地区，肩部疾病如冻结肩等常见疾病的早期准确诊断面临挑战，急需低成本、易扩展的辅助诊断方案。

Method: 提出HMVDx混合运动视频诊断框架，将动作理解和疾病诊断两个任务分别由两个多模态大语言模型完成，并提出了基于医疗决策逻辑过程的新评估指标Usability Index。

Result: 实验比较显示，HMVDx在肩关节损伤诊断中的准确率比直接视频诊断提高了79.6%。

Conclusion: 该研究展示了低成本多模态大语言模型在医疗应用中的潜在价值，为未来医学领域视频理解应用提供了重要技术贡献。

Abstract: Shoulder disorders, such as frozen shoulder (a.k.a., adhesive capsulitis),
are common conditions affecting the health of people worldwide, and have a high
incidence rate among the elderly and workers engaged in repetitive shoulder
tasks. In regions with scarce medical resources, achieving early and accurate
diagnosis poses significant challenges, and there is an urgent need for
low-cost and easily scalable auxiliary diagnostic solutions. This research
introduces videos captured by consumer-grade devices as the basis for
diagnosis, reducing the cost for users. We focus on the innovative application
of Multimodal Large Language Models (MLLMs) in the preliminary diagnosis of
shoulder disorders and propose a Hybrid Motion Video Diagnosis framework
(HMVDx). This framework divides the two tasks of action understanding and
disease diagnosis, which are respectively completed by two MLLMs. In addition
to traditional evaluation indicators, this work proposes a novel metric called
Usability Index by the logical process of medical decision-making (action
recognition, movement diagnosis, and final diagnosis). This index evaluates the
effectiveness of MLLMs in the medical field from the perspective of the entire
medical diagnostic pathway, revealing the potential value of low-cost MLLMs in
medical applications for medical practitioners. In experimental comparisons,
the accuracy of HMVDx in diagnosing shoulder joint injuries has increased by
79.6\% compared with direct video diagnosis, a significant technical
contribution to future research on the application of MLLMs for video
understanding in the medical field.

</details>


### [12] [Cattle-CLIP: A Multimodal Framework for Cattle Behaviour Recognition](https://arxiv.org/abs/2510.09203v1)
*Huimin Liu,Jing Gao,Daria Baran,AxelX Montout,Neill W Campbell,Andrew W Dowsey*

Main category: cs.CV

TL;DR: Cattle-CLIP是一个基于多模态深度学习框架的牛只行为识别系统，通过语义线索提升视频特征识别性能，在监督学习和少样本学习场景下均表现优异。


<details>
  <summary>Details</summary>
Motivation: 牛只行为是动物健康、生产力和整体福祉的重要指标。视频监控结合深度学习已成为动物生物识别的主流方法，但现有方法在数据稀缺的行为识别任务中存在性能瓶颈。

Method: 基于大规模图像-语言模型CLIP进行适配，添加时间整合模块，采用定制化数据增强策略和专门文本提示来解决预训练模型与真实牛只监控视频之间的领域差距。

Result: 在监督设置下，Cattle-CLIP在六种行为上的总体准确率达到96.1%，其中进食、饮水和站立反刍行为的召回率接近100%，并在少样本场景中展现出强大的泛化能力。

Conclusion: 该研究展示了多模态学习在农业和动物行为分析中的潜力，特别是在数据稀缺的行为识别任务中具有重要应用价值。

Abstract: Cattle behaviour is a crucial indicator of an individual animal health,
productivity and overall well-being. Video-based monitoring, combined with deep
learning techniques, has become a mainstream approach in animal biometrics, and
it can offer high accuracy in some behaviour recognition tasks. We present
Cattle-CLIP, a multimodal deep learning framework for cattle behaviour
recognition, using semantic cues to improve the performance of video-based
visual feature recognition. It is adapted from the large-scale image-language
model CLIP by adding a temporal integration module. To address the domain gap
between web data used for the pre-trained model and real-world cattle
surveillance footage, we introduce tailored data augmentation strategies and
specialised text prompts. Cattle-CLIP is evaluated under both fully-supervised
and few-shot learning scenarios, with a particular focus on data-scarce
behaviour recognition - an important yet under-explored goal in livestock
monitoring. To evaluate the proposed method, we release the CattleBehaviours6
dataset, which comprises six types of indoor behaviours: feeding, drinking,
standing-self-grooming, standing-ruminating, lying-self-grooming and
lying-ruminating. The dataset consists of 1905 clips collected from our John
Oldacre Centre dairy farm research platform housing 200 Holstein-Friesian cows.
Experiments show that Cattle-CLIP achieves 96.1% overall accuracy across six
behaviours in a supervised setting, with nearly 100% recall for feeding,
drinking and standing-ruminating behaviours, and demonstrates robust
generalisation with limited data in few-shot scenarios, highlighting the
potential of multimodal learning in agricultural and animal behaviour analysis.

</details>


### [13] [Towards Safer and Understandable Driver Intention Prediction](https://arxiv.org/abs/2510.09200v1)
*Mukilan Karuppasamy,Shankar Gangisetty,Shyam Nandan Rai,Carlo Masone,C V Jawahar*

Main category: cs.CV

TL;DR: 该论文提出了一个可解释的驾驶员意图预测框架VCBM，创建了DAAD-X数据集，通过概念瓶颈模型生成时空一致的解释，证明基于transformer的模型比CNN模型更具可解释性。


<details>
  <summary>Details</summary>
Motivation: 随着自动驾驶系统与人类交互增加，决策过程的可解释性对安全驾驶至关重要。深度学习系统需要理解环境和驾驶任务的基础表征，这在基于深度学习的系统中仍是一个重大挑战。

Method: 提出了视频概念瓶颈模型(VCBM)，该框架能够固有地生成时空一致的解释，而不依赖后处理技术。创建了DAAD-X多模态数据集，提供驾驶员决策的层次化文本解释。

Result: 在DAAD-X数据集上的广泛评估表明，基于transformer的模型比传统的基于CNN的模型表现出更大的可解释性。还引入了多标签t-SNE可视化技术来说明多个解释之间的解缠和因果相关性。

Conclusion: 该研究为可解释的驾驶员意图预测提供了新的数据集和框架，证明了transformer模型在可解释性方面的优势，为自动驾驶系统的安全交互提供了重要支持。

Abstract: Autonomous driving (AD) systems are becoming increasingly capable of handling
complex tasks, mainly due to recent advances in deep learning and AI. As
interactions between autonomous systems and humans increase, the
interpretability of decision-making processes in driving systems becomes
increasingly crucial for ensuring safe driving operations. Successful
human-machine interaction requires understanding the underlying representations
of the environment and the driving task, which remains a significant challenge
in deep learning-based systems. To address this, we introduce the task of
interpretability in maneuver prediction before they occur for driver safety,
i.e., driver intent prediction (DIP), which plays a critical role in AD
systems. To foster research in interpretable DIP, we curate the eXplainable
Driving Action Anticipation Dataset (DAAD-X), a new multimodal, ego-centric
video dataset to provide hierarchical, high-level textual explanations as
causal reasoning for the driver's decisions. These explanations are derived
from both the driver's eye-gaze and the ego-vehicle's perspective. Next, we
propose Video Concept Bottleneck Model (VCBM), a framework that generates
spatio-temporally coherent explanations inherently, without relying on post-hoc
techniques. Finally, through extensive evaluations of the proposed VCBM on the
DAAD-X dataset, we demonstrate that transformer-based models exhibit greater
interpretability than conventional CNN-based models. Additionally, we introduce
a multilabel t-SNE visualization technique to illustrate the disentanglement
and causal correlation among multiple explanations. Our data, code and models
are available at: https://mukil07.github.io/VCBM.github.io/

</details>


<div id='cs.CL'></div>

# cs.CL [[Back]](#toc)

### [14] [AutoPR: Let's Automate Your Academic Promotion!](https://arxiv.org/abs/2510.09558v1)
*Qiguang Chen,Zheng Yan,Mingda Yang,Libo Qin,Yixin Yuan,Hanjing Li,Jinhao Liu,Yiyan Ji,Dengyun Peng,Jiannan Guan,Mengkang Hu,Yantao Du,Wanxiang Che*

Main category: cs.CL

TL;DR: AutoPR是一种自动将研究论文转化为准确、吸引人且及时公开内容的新任务，通过PRAgent多智能体框架实现三阶段自动化流程，显著提升学术推广效果。


<details>
  <summary>Details</summary>
Motivation: 随着同行评审研究数量激增，学者依赖社交平台发现研究，作者需要投入大量精力推广工作以确保可见性和引用。为简化流程并减少人力依赖，需要自动化推广解决方案。

Method: 提出PRAgent多智能体框架，包含三个阶段：多模态内容提取、协作合成生成精炼输出、平台特定适配以优化规范、语气和标签实现最大覆盖。

Result: 在PRBench基准测试中，相比直接LLM流水线，PRAgent实现显著改进：总观看时间增加604%，点赞数增长438%，整体参与度至少提升2.9倍。消融研究表明平台建模和目标推广贡献最大。

Conclusion: AutoPR是一个可处理、可衡量的研究问题，为可扩展、有影响力的自动化学术交流提供了路线图。

Abstract: As the volume of peer-reviewed research surges, scholars increasingly rely on
social platforms for discovery, while authors invest considerable effort in
promoting their work to ensure visibility and citations. To streamline this
process and reduce the reliance on human effort, we introduce Automatic
Promotion (AutoPR), a novel task that transforms research papers into accurate,
engaging, and timely public content. To enable rigorous evaluation, we release
PRBench, a multimodal benchmark that links 512 peer-reviewed articles to
high-quality promotional posts, assessing systems along three axes: Fidelity
(accuracy and tone), Engagement (audience targeting and appeal), and Alignment
(timing and channel optimization). We also introduce PRAgent, a multi-agent
framework that automates AutoPR in three stages: content extraction with
multimodal preparation, collaborative synthesis for polished outputs, and
platform-specific adaptation to optimize norms, tone, and tagging for maximum
reach. When compared to direct LLM pipelines on PRBench, PRAgent demonstrates
substantial improvements, including a 604% increase in total watch time, a 438%
rise in likes, and at least a 2.9x boost in overall engagement. Ablation
studies show that platform modeling and targeted promotion contribute the most
to these gains. Our results position AutoPR as a tractable, measurable research
problem and provide a roadmap for scalable, impactful automated scholarly
communication.

</details>


### [15] [Multimodal Policy Internalization for Conversational Agents](https://arxiv.org/abs/2510.09474v1)
*Zhenhailong Wang,Jiateng Liu,Amin Fazel,Ritesh Sarkhel,Xing Fan,Xiang Li,Chenlei Guo,Heng Ji,Ruhi Sarikaya*

Main category: cs.CL

TL;DR: 本文提出了多模态策略内化(MPI)任务，通过TriMPI三阶段训练框架将复杂的多模态策略内化到模型参数中，实现无需推理时包含策略的强策略遵循能力。


<details>
  <summary>Details</summary>
Motivation: 现代对话系统依赖预定义策略，但随着系统扩展，这些策略变得复杂冗长，导致遵循困难且计算成本高。多模态策略管理关键但研究不足，现有方法主要关注文本压缩和安全规则对齐。

Method: 提出TriMPI三阶段框架：1)通过持续预训练注入策略知识；2)监督微调；3)PolicyRollout强化学习，使用策略感知响应进行基于策略的探索。

Result: TriMPI在端到端准确性、泛化性和抗遗忘鲁棒性方面取得显著提升，构建了两个涵盖合成和真实世界决策与工具使用任务的数据集。

Conclusion: 作为多模态策略内化的首个工作，提供了数据集、训练方法和全面评估，为未来研究奠定基础。

Abstract: Modern conversational agents like ChatGPT and Alexa+ rely on predefined
policies specifying metadata, response styles, and tool-usage rules. As these
LLM-based systems expand to support diverse business and user queries, such
policies, often implemented as in-context prompts, are becoming increasingly
complex and lengthy, making faithful adherence difficult and imposing large
fixed computational costs. With the rise of multimodal agents, policies that
govern visual and multimodal behaviors are critical but remain understudied.
Prior prompt-compression work mainly shortens task templates and
demonstrations, while existing policy-alignment studies focus only on
text-based safety rules. We introduce Multimodal Policy Internalization (MPI),
a new task that internalizes reasoning-intensive multimodal policies into model
parameters, enabling stronger policy-following without including the policy
during inference. MPI poses unique data and algorithmic challenges. We build
two datasets spanning synthetic and real-world decision-making and tool-using
tasks and propose TriMPI, a three-stage training framework. TriMPI first
injects policy knowledge via continual pretraining, then performs supervised
finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement
learning extension that augments rollouts with policy-aware responses for
grounded exploration. TriMPI achieves notable gains in end-to-end accuracy,
generalization, and robustness to forgetting. As the first work on multimodal
policy internalization, we provide datasets, training recipes, and
comprehensive evaluations to foster future research. Project page:
https://mikewangwzhl.github.io/TriMPI.

</details>


### [16] [The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach](https://arxiv.org/abs/2510.09424v1)
*Nizar El Ghazal,Antoine Caubrière,Valentin Vielzeuf*

Main category: cs.CL

TL;DR: 本文比较了基于Speech-LLM的端到端口语对话状态跟踪中的上下文管理策略，发现完整口语对话历史输入性能最佳，而注意力池化压缩方法能在保持准确性的同时减少上下文大小。


<details>
  <summary>Details</summary>
Motivation: 研究不同上下文管理策略对口语对话状态跟踪性能的影响，探索如何更有效地利用口语对话历史信息。

Method: 系统评估了三种策略：传统多模态上下文（文本历史+当前口语轮次）、完整口语历史、压缩口语历史方法，并在SpokenWOZ语料库上进行实验。

Result: 完整口语对话历史输入在相似规模模型中表现最佳，显著超越先前方法；注意力池化压缩方法在保持竞争力的准确性的同时减少了上下文大小。

Conclusion: 改进主要源于更有效的上下文利用，完整口语历史策略性能最优，压缩方法提供了良好的权衡选择。

Abstract: This paper presents a comparative study of context management strategies for
end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically
evaluate traditional multimodal context (combining text history and spoken
current turn), full spoken history, and compressed spoken history approaches.
Our experiments on the SpokenWOZ corpus demonstrate that providing the full
spoken conversation as input yields the highest performance among models of
similar size, significantly surpassing prior methods. Furthermore, we show that
attention-pooling-based compression of the spoken history offers a strong
trade-off, maintaining competitive accuracy with reduced context size. Detailed
analysis confirms that improvements stem from more effective context
utilization.

</details>


### [17] [CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2510.09266v1)
*Kaiwen Wei,Xiao Liu,Jie Zhang,Zijian Wang,Ruida Liu,Yuming Yang,Xin Xiao,Xiao Sun,Haoyang Zeng,Changzai Pan,Yidan Zhang,Jiang Zhong,Peijin Wang,Yingchao Feng*

Main category: cs.CL

TL;DR: 提出了CFVBench基准测试，评估多模态检索增强生成模型在视频问答中的表现，发现现有模型在捕捉细粒度多模态细节方面存在瓶颈，并提出了自适应视觉优化框架AVR来提升性能。


<details>
  <summary>Details</summary>
Motivation: 现有视频MRAG基准测试在模态覆盖和格式多样性方面存在局限，主要关注单模态或有限模态任务，或粗粒度场景理解，需要更全面的评估基准。

Method: 构建了CFVBench大规模人工验证基准测试，包含599个公开视频和5,360个开放式问答对；提出了自适应视觉优化框架AVR，通过自适应增加帧采样密度和选择性调用外部工具来提升细粒度多模态理解。

Result: 系统评估了7种检索方法和14个常用MLLMs，发现当前模型（包括GPT5和Gemini）在捕捉瞬态但关键的细粒度多模态细节方面存在困难；AVR框架在所有评估的MLLMs中都能持续提升细粒度多模态理解和性能。

Conclusion: CFVBench揭示了当前多模态模型在细粒度细节捕捉方面的关键瓶颈，而AVR框架提供了一种简单有效的解决方案，能够显著提升模型在复杂多模态任务中的表现。

Abstract: Multimodal Retrieval-Augmented Generation (MRAG) enables Multimodal Large
Language Models (MLLMs) to generate responses with external multimodal
evidence, and numerous video-based MRAG benchmarks have been proposed to
evaluate model capabilities across retrieval and generation stages. However,
existing benchmarks remain limited in modality coverage and format diversity,
often focusing on single- or limited-modality tasks, or coarse-grained scene
understanding. To address these gaps, we introduce CFVBench, a large-scale,
manually verified benchmark constructed from 599 publicly available videos,
yielding 5,360 open-ended QA pairs. CFVBench spans high-density formats and
domains such as chart-heavy reports, news broadcasts, and software tutorials,
requiring models to retrieve and reason over long temporal video spans while
maintaining fine-grained multimodal information. Using CFVBench, we
systematically evaluate 7 retrieval methods and 14 widely-used MLLMs, revealing
a critical bottleneck: current models (even GPT5 or Gemini) struggle to capture
transient yet essential fine-grained multimodal details. To mitigate this, we
propose Adaptive Visual Refinement (AVR), a simple yet effective framework that
adaptively increases frame sampling density and selectively invokes external
tools when necessary. Experiments show that AVR consistently enhances
fine-grained multimodal comprehension and improves performance across all
evaluated MLLMs

</details>


<div id='physics.app-ph'></div>

# physics.app-ph [[Back]](#toc)

### [18] [Self-Resetting Soft Ring Enables Autonomous and Continuous Leaping under Uniform Light](https://arxiv.org/abs/2510.09529v1)
*Fangjie Qi,Caizhi Zhou,Haitao Qing,Haoze Sun,Jie Yin*

Main category: physics.app-ph

TL;DR: 该研究提出了一种毫米级自复位软环机器人，能够在均匀红外光照下实现重复垂直和稳定水平跳跃，无需外部控制。通过几何不对称和质心调节，机器人可在爬行、定向跳跃和垂直跳跃间切换，在多种复杂地形中表现出鲁棒运动能力。


<details>
  <summary>Details</summary>
Motivation: 自然界中跳跃是一种在杂乱、不平或不稳定环境中高效移动的策略，但在软机器人中实现连续自主跳跃仍具挑战性，主要受限于能量存储有限和对人工干预或锁存机制的依赖。

Method: 采用环形液晶弹性体结构，通过扭曲存储弹性能量，当刚性尾部撞击地面时突然释放能量推动机器人。在空中阶段，扭曲体自主解扭为下一周期复位。通过调节几何不对称性和质心位置控制运动模式。

Result: 优化配置可实现超过80倍体高的垂直跳跃和超过3倍体长的定向水平跳跃。机器人能够在斜坡、平行障碍以及草地、湿沙、覆盖物等多种自然杂乱地形中表现出弹性和鲁棒的运动能力。

Conclusion: 这项工作建立了一种基于扭曲机制、光热驱动的软机器人新范式，能够实现自主连续跳跃，在环境导航、群体机器人和非结构化地形导航方面具有应用潜力。

Abstract: Jumping is an efficient locomotion strategy to traverse cluttered, uneven, or
unstable environments in nature, yet replicating continuous, autonomous leaping
in soft robots remains challenging due to limited energy storage and reliance
on human intervention or latches. Here, we report a millimeter-scale,
self-resetting soft ring that achieves repeated vertical and stable horizontal
leaps under uniform infrared illumination without external control. The
ring-shaped liquid crystal elastomer body twists to store elastic energy, which
is suddenly released when a rigid tail strikes the ground, propelling the
robot. During the airborne phase, the twisted body autonomously untwists,
resetting for the next cycle. By tuning geometric asymmetry and the center of
mass, the robot transitions between crawling, directional leaping, and vertical
jumping. Optimized configurations yield vertical jumps exceeding 80 body
heights and directional horizontal leaps over 3 body lengths. Beyond controlled
motion on flat ground, the robot demonstrates resilient and robust locomotion
across slopes, parallel hurdles, and diverse cluttered natural terrains
including grass, wet sand, and mulch. This work establishes a new paradigm of
twisting-enabled, photothermally powered soft robots capable of autonomous,
continuous leaping, with potential applications in environmental navigation,
swarm robotics, and unstructured terrain navigation.

</details>


<div id='cs.IT'></div>

# cs.IT [[Back]](#toc)

### [19] [Serial Polar Automorphism Ensemble Decoders for Physical Unclonable Functions](https://arxiv.org/abs/2510.09220v1)
*Marvin Rübenacke,Sebastian Cammerer,Michael Sullivan,Alexander Keller*

Main category: cs.IT

TL;DR: 提出了一种基于Polar码和低复杂度自同构集成解码(AED)的物理不可克隆函数(PUF)编码方案，相比BCH码减少了43%的码字比特数，同时保持10^-6的块错误率。


<details>
  <summary>Details</summary>
Motivation: PUF应用需要极低的失败率(10^-6以下)和高原始误码率(22%)，这要求设计高效的超低码率编码方案。

Method: 采用Polar码和串行AED方案，重用单个连续消除(SC)解码器进行多次解码尝试；通过级联和递归交织器扩展AED候选数；使用3位量化策略降低SC解码器面积需求。

Result: 在K=312有效载荷比特下，与BCH码基线相比，码字比特数减少1.75倍，同时保持10^-6的块错误率；减少了1.75倍的辅助数据存储需求。

Conclusion: 所提出的编码方案在保持相同错误率的同时，显著减少了码字比特数和芯片面积，为PUF应用提供了高效的编码解决方案。

Abstract: Physical unclonable functions (PUFs) involve challenging practical
applications of error-correcting codes (ECCs), requiring extremely low failure
rates on the order of $10^{-6}$ and below despite raw input bit error rates as
high as 22%. These requirements call for an efficient ultra-low rate code
design. In this work, we propose a novel coding scheme tailored for PUFs based
on Polar codes and a low-complexity version of automorphism ensemble decoding
(AED). Notably, our serial AED scheme reuses a single successive cancellation
(SC) decoder across multiple decoding attempts. By introducing cascaded and
recursive interleavers, we efficiently scale the number of AED candidates
without requiring expensive large multiplexers. An aggressive quantization
strategy of only 3 bits per message further reduces the area requirements of
the underlying SC decoder. The resulting coding scheme achieves the same block
error rate of $10^{-6}$ as our baseline based on Bose-Ray-Chaudhuri-Hocquenghem
(BCH) codes while requiring 1.75x fewer codeword bits to encode the same K =
312 payload bits. This reduction translates directly into 1.75x less helper
data storage and, consequently, a smaller overall chip area.

</details>


<div id='stat.ME'></div>

# stat.ME [[Back]](#toc)

### [20] [Defensive Model Expansion for Robust Bayesian Inference](https://arxiv.org/abs/2510.09598v1)
*Antonio R. Linero*

Main category: stat.ME

TL;DR: 该论文提出了一种防御性模型扩展方法，通过在参数模型基础上添加强收缩的非参数组件，实现自动适应：当参数模型正确时恢复参数效率，当模型误设时激活灵活组件捕获缺失信号。


<details>
  <summary>Details</summary>
Motivation: 应用研究人员对非参数方法存在顾虑，担心在小样本中失去功效或在简单模型足够时过拟合。论文旨在证明当非参数模型强烈收缩向参数子模型时，这些担忧是不必要的。

Method: 考虑在参数模型基础上扩展一个向零强收缩的非参数组件。采用贝叶斯非参数模型锚定到线性回归，包括高斯过程回归和贝叶斯加性回归树的变体。

Result: 证明了这些模型在参数模型成立时能一致识别正确的参数子模型，并为回归系数提供渐近有效推断。模拟显示"通用BART"模型在参数模型成立时与正确指定的线性回归表现相同，在存在非线性效应时显著优于线性回归。

Conclusion: 提出"防御性模型扩展"作为防止模型误设的实用范式，使模型既能保持参数效率，又能灵活适应复杂模式。

Abstract: Some applied researchers hesitate to use nonparametric methods, worrying that
they will lose power in small samples or overfit the data when simpler models
are sufficient. We argue that at least some of these concerns are unfounded
when nonparametric models are strongly shrunk towards parametric submodels. We
consider expanding a parametric model with a nonparametric component that is
heavily shrunk toward zero. This construction allows the model to adapt
automatically: if the parametric model is correct, the nonparametric component
disappears, recovering parametric efficiency, while if it is misspecified, the
flexible component activates to capture the missing signal. We show that this
adaptive behavior follows from simple and general conditions. Specifically, we
prove that Bayesian nonparametric models anchored to linear regression,
including variants of Gaussian processes regression and Bayesian additive
regression trees, consistently identify the correct parametric submodel when it
holds and give asymptotically efficient inference for regression coefficients.
In simulations, we find that the "general BART" model performs identically to
correctly specified linear regression when the parametric model holds, and
substantially outperform it when nonlinear effects are present. This suggests a
practical paradigm: "defensive model expansion" as a safeguard against model
misspecification.

</details>


### [21] [The bixplot: A variation on the boxplot suited for bimodal data](https://arxiv.org/abs/2510.09276v1)
*Camille M. Montalcini,Peter J. Rousseeuw*

Main category: stat.ME

TL;DR: 本文提出了一种名为bixplot的可视化方法扩展，专门用于检测和显示双峰和多峰分布，通过构建确保连续簇的单变量聚类方法来识别数据中潜在的有意义子组。


<details>
  <summary>Details</summary>
Motivation: 箱线图及相关可视化方法被广泛用于单变量数据的初步探索，但现有方法在检测和显示双峰和多峰性方面存在局限，需要专门工具来更好地识别数据中的子组结构。

Method: 构建了一种单变量聚类方法，确保簇的连续性（即没有簇的成员位于另一个簇内部），且每个簇包含至少给定数量的唯一成员，从而创建bixplot可视化显示。

Result: bixplot显示有助于识别和解释数据中潜在的有意义子组，同时显示单个数据值以引起对孤立点的注意。该方法已在Python和R中实现，并在多个真实数据集上展示了其多种选项。

Conclusion: bixplot作为箱线图的扩展，有效促进了双峰和多峰分布的检测与可视化，为探索数据中的子组结构提供了有力工具。

Abstract: Boxplots and related visualization methods are widely used exploratory tools
for taking a first look at collections of univariate variables. In this note an
extension is provided that is specifically designed to detect and display
bimodality and multimodality when the data warrant it. For this purpose a
univariate clustering method is constructed that ensures contiguous clusters,
meaning that no cluster has members inside another cluster, and such that each
cluster contains at least a given number of unique members. The resulting
bixplot display facilitates the identification and interpretation of
potentially meaningful subgroups underlying the data. The bixplot also displays
the individual data values, which can draw attention to isolated points.
Implementations of the bixplot are available in both Python and R, and their
many options are illustrated on several real datasets. For instance, an
external variable can be visualized by color gradations inside the display.

</details>


<div id='cs.SE'></div>

# cs.SE [[Back]](#toc)

### [22] [A Semantic Framework for Patient Digital Twins in Chronic Care](https://arxiv.org/abs/2510.09134v1)
*Amal Elgammal,Bernd J. Krämer,Michael P. Papazoglou,Mira Raheem*

Main category: cs.SE

TL;DR: 本文提出了患者医疗数字孪生（PMDT）框架，这是一个基于本体的患者模型，整合了生理、心理、行为、基因组等多模态健康数据，支持语义互操作性、自动推理和隐私保护。


<details>
  <summary>Details</summary>
Motivation: 当前数字孪生应用多为器官特异性或局限于孤立数据类型，缺乏统一且隐私保护的框架。需要整合多模态健康数据以实现精准、自适应和预防性决策。

Method: 采用OWL 2.0实现本体驱动的患者模型，围绕模块化蓝图（患者、疾病诊断、治疗随访、轨迹、安全、路径、不良事件）构建，通过专家研讨会、问卷和真实世界免疫治疗患者试点研究进行迭代优化和验证。

Result: 评估确认了本体覆盖度、推理正确性、可用性和GDPR合规性。PMDT能够统一异构数据，操作能力问题，支持描述性、预测性和规范性分析。

Conclusion: PMDT通过弥合数据碎片化和语义标准化方面的差距，为下一代数字健康生态系统提供了经过验证的基础，将慢性病护理转变为主动、持续优化和公平的管理模式。

Abstract: Personalized chronic care requires the integration of multimodal health data
to enable precise, adaptive, and preventive decision-making. Yet most current
digital twin (DT) applications remain organ-specific or tied to isolated data
types, lacking a unified and privacy-preserving foundation. This paper
introduces the Patient Medical Digital Twin (PMDT), an ontology-driven in
silico patient framework that integrates physiological, psychosocial,
behavioral, and genomic information into a coherent, extensible model.
Implemented in OWL 2.0, the PMDT ensures semantic interoperability, supports
automated reasoning, and enables reuse across diverse clinical contexts. Its
ontology is structured around modular Blueprints (patient, disease and
diagnosis, treatment and follow-up, trajectories, safety, pathways, and adverse
events), formalized through dedicated conceptual views. These were iteratively
refined and validated through expert workshops, questionnaires, and a pilot
study in the EU H2020 QUALITOP project with real-world immunotherapy patients.
Evaluation confirmed ontology coverage, reasoning correctness, usability, and
GDPR compliance. Results demonstrate the PMDT's ability to unify heterogeneous
data, operationalize competency questions, and support descriptive, predictive,
and prescriptive analytics in a federated, privacy-preserving manner. By
bridging gaps in data fragmentation and semantic standardization, the PMDT
provides a validated foundation for next-generation digital health ecosystems,
transforming chronic care toward proactive, continuously optimized, and
equitable management.

</details>


<div id='cs.IR'></div>

# cs.IR [[Back]](#toc)

### [23] [MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval](https://arxiv.org/abs/2510.09510v1)
*Siyue Zhang,Yuan Gao,Xiao Zhou,Yilun Zhao,Tingyu Song,Arman Cohan,Anh Tuan Luu,Chen Zhao*

Main category: cs.IR

TL;DR: MRMR是首个需要密集推理的专家级多学科多模态检索基准，包含1502个查询，涵盖23个领域，通过引入多领域专家级查询、推理密集型任务和图像-文本交错序列，显著提升了多模态检索的挑战性。


<details>
  <summary>Details</summary>
Motivation: 现有检索基准主要关注通用领域，缺乏对专家级多学科知识和密集推理能力的评估，需要构建更真实、更具挑战性的多模态检索基准。

Method: 构建包含1502个查询的基准，涵盖23个专业领域，引入矛盾检索新任务，采用图像-文本交错序列的查询和文档结构，评估4类多模态检索系统和14个前沿模型。

Result: Qwen3-Embedding模型结合LLM生成的图像描述表现最佳，但多模态模型在推理密集型任务上仍有不足，显示多模态检索模型仍有很大改进空间。

Conclusion: MRMR基准为推进多模态检索在更真实和具有挑战性场景中的发展铺平了道路，揭示了当前模型在专家级推理任务上的局限性。

Abstract: We introduce MRMR, the first expert-level multidisciplinary multimodal
retrieval benchmark requiring intensive reasoning. MRMR contains 1,502 queries
spanning 23 domains, with positive documents carefully verified by human
experts. Compared to prior benchmarks, MRMR introduces three key advancements.
First, it challenges retrieval systems across diverse areas of expertise,
enabling fine-grained model comparison across domains. Second, queries are
reasoning-intensive, with images requiring deeper interpretation such as
diagnosing microscopic slides. We further introduce Contradiction Retrieval, a
novel task requiring models to identify conflicting concepts. Finally, queries
and documents are constructed as image-text interleaved sequences. Unlike
earlier benchmarks restricted to single images or unimodal documents, MRMR
offers a realistic setting with multi-image queries and mixed-modality corpus
documents. We conduct an extensive evaluation of 4 categories of multimodal
retrieval systems and 14 frontier models on MRMR. The text embedding model
Qwen3-Embedding with LLM-generated image captions achieves the highest
performance, highlighting substantial room for improving multimodal retrieval
models. Although latest multimodal models such as Ops-MM-Embedding perform
competitively on expert-domain queries, they fall short on reasoning-intensive
tasks. We believe that MRMR paves the way for advancing multimodal retrieval in
more realistic and challenging scenarios.

</details>


<div id='cs.LG'></div>

# cs.LG [[Back]](#toc)

### [24] [Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](https://arxiv.org/abs/2510.09201v1)
*Yumin Choi,Dongki Kim,Jinheon Baek,Sung Ju Hwang*

Main category: cs.LG

TL;DR: 本文提出多模态提示优化问题，并开发了MPO框架，通过联合优化文本和非文本提示来提升多模态大语言模型的性能。


<details>
  <summary>Details</summary>
Motivation: 当前提示优化方法局限于文本模态，限制了多模态大语言模型的潜力，因此需要扩展到多模态空间。

Method: 提出多模态提示优化器(MPO)，通过保持对齐的联合更新来优化多模态提示，并利用贝叶斯选择策略指导候选提示选择。

Result: 在图像、视频、分子等多种模态上的实验表明，MPO优于领先的纯文本优化方法。

Conclusion: 多模态提示优化是实现多模态大语言模型潜力的关键步骤。

Abstract: Large Language Models (LLMs) have shown remarkable success, and their
multimodal expansions (MLLMs) further unlock capabilities spanning images,
videos, and other modalities beyond text. However, despite this shift, prompt
optimization approaches, designed to reduce the burden of manual prompt
crafting while maximizing performance, remain confined to text, ultimately
limiting the full potential of MLLMs. Motivated by this gap, we introduce the
new problem of multimodal prompt optimization, which expands the prior
definition of prompt optimization to the multimodal space defined by the pairs
of textual and non-textual prompts. To tackle this problem, we then propose the
Multimodal Prompt Optimizer (MPO), a unified framework that not only performs
the joint optimization of multimodal prompts through alignment-preserving
updates but also guides the selection process of candidate prompts by
leveraging earlier evaluations as priors in a Bayesian-based selection
strategy. Through extensive experiments across diverse modalities that go
beyond text, such as images, videos, and even molecules, we demonstrate that
MPO outperforms leading text-only optimization methods, establishing multimodal
prompt optimization as a crucial step to realizing the potential of MLLMs.

</details>


<div id='math.AT'></div>

# math.AT [[Back]](#toc)

### [25] [Parametrized Topological Complexity for a Multi-Robot System with Variable Tasks](https://arxiv.org/abs/2510.09323v1)
*Gopal Chandra Dutta,Amit Kumar Paul,Subhankar Sau*

Main category: math.AT

TL;DR: 本文研究多机器人在未知障碍环境中的广义运动规划问题，通过拓扑复杂性理论确定碰撞避免运动规划算法的最小不稳定性要求。


<details>
  <summary>Details</summary>
Motivation: 扩展Farber等人的序列参数化拓扑复杂性框架，解决异构机器人系统在未知障碍环境中的运动规划问题，其中每个机器人需要按顺序访问不同数量的目标状态。

Method: 构建适当的纤维化数学模型，通过拓扑复杂性理论分析问题，包括奇偶维数环境空间的详细分析、上同调计算和运动规划算法的显式构造。

Result: 确定了广义设置下的拓扑复杂性不变量，该不变量捕获了在参数依赖约束下设计无碰撞运动规划算法所需的最小算法不稳定性。

Conclusion: 该研究为异构多机器人系统在未知环境中的运动规划提供了理论基础，通过拓扑复杂性量化了算法设计的根本限制。

Abstract: We study a generalized motion planning problem involving multiple autonomous
robots navigating in a $d$-dimensional Euclidean space in the presence of a set
of obstacles whose positions are unknown a priori. Each robot is required to
visit sequentially a prescribed set of target states, with the number of
targets varying between robots. This heterogeneous setting generalizes the
framework considered in the prior works on sequential parametrized topological
complexity by Farber and the second author of this article. To determine the
topological complexity of our problem, we formulate it mathematically by
constructing an appropriate fibration. Our main contribution is the
determination of this invariant in the generalized setting, which captures the
minimal algorithmic instability required for designing collision-free motion
planning algorithms under parameter-dependent constraints. We provide a
detailed analysis for both odd and even-dimensional ambient spaces, including
the essential cohomological computations and explicit constructions of
corresponding motion planning algorithms.

</details>


<div id='stat.ML'></div>

# stat.ML [[Back]](#toc)

### [26] [Interpretable Generative and Discriminative Learning for Multimodal and Incomplete Clinical Data](https://arxiv.org/abs/2510.09513v1)
*Albert Belenguer-Llorens,Carlos Sevilla-Salcedo,Janaina Mourao-Miranda,Vanessa Gómez-Verdejo*

Main category: stat.ML

TL;DR: 提出一种贝叶斯方法处理多模态临床数据中的不完整视图和有限样本问题，结合生成式和判别式学习实现自动插补缺失视图和鲁棒推理。


<details>
  <summary>Details</summary>
Motivation: 现实临床问题通常具有多模态数据特征，但存在视图不完整和样本量有限的问题，这对机器学习算法构成重大挑战。

Method: 采用贝叶斯方法，集成(1)生成式公式捕捉跨视图关系的半监督策略，和(2)判别式任务导向公式识别特定下游目标的相关信息。

Result: 该方法能够捕获和解开生物、心理和社会人口等多模态之间的复杂交互作用，在临床数据中表现出明显潜力。

Conclusion: 这种双重生成-判别式公式既提供一般理解又提供任务特定见解，实现缺失视图的自动插补和跨数据源的鲁棒推理。

Abstract: Real-world clinical problems are often characterized by multimodal data,
usually associated with incomplete views and limited sample sizes in their
cohorts, posing significant limitations for machine learning algorithms. In
this work, we propose a Bayesian approach designed to efficiently handle these
challenges while providing interpretable solutions. Our approach integrates (1)
a generative formulation to capture cross-view relationships with a
semi-supervised strategy, and (2) a discriminative task-oriented formulation to
identify relevant information for specific downstream objectives. This dual
generative-discriminative formulation offers both general understanding and
task-specific insights; thus, it provides an automatic imputation of the
missing views while enabling robust inference across different data sources.
The potential of this approach becomes evident when applied to the multimodal
clinical data, where our algorithm is able to capture and disentangle the
complex interactions among biological, psychological, and sociodemographic
modalities.

</details>


<div id='cond-mat.soft'></div>

# cond-mat.soft [[Back]](#toc)

### [27] [Toggling stiffness via multistability](https://arxiv.org/abs/2510.09511v1)
*Hugo de Souza Oliveira,Michele Curatolo,Renate Sachse,Edoardo Milana*

Main category: cond-mat.soft

TL;DR: 本文提出了一种多稳态机械超材料，通过结构设计实现可切换的刚度效应，其中有效剪切刚度在不同稳定构型之间离散切换。该材料可用于软机器人和智能结构中的自适应系统。


<details>
  <summary>Details</summary>
Motivation: 开发能够通过结构设计而非材料成分实现可编程机械响应的机械超材料，特别是实现可切换刚度效应的多稳态系统，为软机器人和智能结构提供自适应解决方案。

Method: 使用替代梁模型进行力学分析，通过改变支撑梁的细长比或引入局部铰链来调节旋转传递，从而控制刚度比。通过3D打印原型进行实验验证。

Result: 实验验证了数值预测，确认了不同几何形状下一致的刚度切换行为。成功演示了利用此效应的单片软离合器，实现了可编程的逐步刚度调制。

Conclusion: 这项工作建立了使用多稳态超材料实现可切换刚度的设计策略，为软机器人和智能结构中的自适应、轻量化和自主系统铺平了道路。

Abstract: Mechanical metamaterials enable unconventional and programmable mechanical
responses through structural design rather than material composition. In this
work, we introduce a multistable mechanical metamaterial that exhibits a
toggleable stiffness effect, where the effective shear stiffness switches
discretely between stable configurations. The mechanical analysis of surrogate
beam models of the unit cell reveal that this behavior originates from the
rotation transmitted by the support beams to the curved beam, which governs the
balance between bending and axial deformation. The stiffness ratio between the
two states of the unit cell can be tuned by varying the slenderness of the
support beams or by incorporating localized hinges that modulate rotational
transfer. Experiments on 3D-printed prototypes validate the numerical
predictions, confirming consistent stiffness toggling across different
geometries. Finally, we demonstrate a monolithic soft clutch that leverages
this effect to achieve programmable, stepwise stiffness modulation. This work
establishes a design strategy for toggleable stiffness using multistable
metamaterials, paving the way for adaptive, lightweight, and autonomous systems
in soft robotics and smart structures.

</details>


<div id='q-bio.TO'></div>

# q-bio.TO [[Back]](#toc)

### [28] [Unsupervised full-field Bayesian inference of orthotropic hyperelasticity from a single biaxial test: a myocardial case study](https://arxiv.org/abs/2510.09498v1)
*Rogier P. Krijnen,Akshay Joshi,Siddhant Kumar,Mathias Peirlinck*

Main category: q-bio.TO

TL;DR: 提出了一种基于贝叶斯推理的无监督方法EUCLID，用于从单个异质双轴拉伸试验中识别高度非线性正交各向异性材料模型参数，并量化不确定性。


<details>
  <summary>Details</summary>
Motivation: 传统均质组织测试需要多种变形模式（如三轴剪切和双轴拉伸），这需要多个样本和大量操作，存在样本间变异性和操作损伤问题。

Method: 采用EUCLID无监督方法，结合贝叶斯推理方法和三维连续体单元，从单个异质双轴拉伸试验中识别高度非线性正交各向异性本构模型参数。

Result: 该方法在不同噪声水平下能够定量推断合成心肌组织板的材料模型参数，与真实模拟结果吻合良好，并提供了相应的可信区间。

Conclusion: 该方法展示了从单个双轴拉伸试验中表征高度非线性和正交各向异性材料模型的潜力，并具备不确定性量化能力。

Abstract: Fully capturing this behavior in traditional homogenized tissue testing
requires the excitation of multiple deformation modes, i.e. combined triaxial
shear tests and biaxial stretch tests. Inherently, such multimodal experimental
protocols necessitate multiple tissue samples and extensive sample
manipulations. Intrinsic inter-sample variability and manipulation-induced
tissue damage might have an adverse effect on the inversely identified tissue
behavior. In this work, we aim to overcome this gap by focusing our attention
to the use of heterogeneous deformation profiles in a parameter estimation
problem. More specifically, we adapt EUCLID, an unsupervised method for the
automated discovery of constitutive models, towards the purpose of parameter
identification for highly nonlinear, orthotropic constitutive models using a
Bayesian inference approach and three-dimensional continuum elements. We
showcase its strength to quantitatively infer, with varying noise levels, the
material model parameters of synthetic myocardial tissue slabs from a single
heterogeneous biaxial stretch test. This method shows good agreement with the
ground-truth simulations and with corresponding credibility intervals. Our work
highlights the potential for characterizing highly nonlinear and orthotropic
material models from a single biaxial stretch test with uncertainty
quantification.

</details>


<div id='cs.CR'></div>

# cs.CR [[Back]](#toc)

### [29] [Goal-oriented Backdoor Attack against Vision-Language-Action Models via Physical Objects](https://arxiv.org/abs/2510.09269v1)
*Zirun Zhou,Zhengyang Xiao,Haochuan Xu,Jing Sun,Di Wang,Jingfeng Zhang*

Main category: cs.CR

TL;DR: 该论文提出了一种针对视觉-语言-动作模型的目标导向后门攻击方法GoBA，通过在训练数据中注入物理对象作为触发器，使模型在遇到物理触发器时执行预定义的目标动作，而正常输入下表现正常。


<details>
  <summary>Details</summary>
Motivation: 现有VLA模型依赖未筛选的训练数据集存在安全隐患，当前后门攻击大多假设白盒访问且仅导致任务失败而非执行特定动作。本文揭示更实际的威胁：攻击者可通过在训练数据中注入物理对象作为触发器来操控VLA模型。

Method: 基于LIBERO基准提出BadLIBERO数据集，包含多样化的物理触发器和目标导向的后门动作。采用三级评估方法将受害VLA在GoBA下的动作分为三种状态：无动作、尝试执行、成功执行。

Result: 实验表明，当物理触发器存在时，GoBA使受害VLA在97%的输入中成功实现后门目标，同时在干净输入上造成零性能下降。动作轨迹和触发器颜色显著影响攻击性能，而触发器大小影响很小。

Conclusion: GoBA展示了物理后门攻击对VLA模型的实际威胁，强调了训练数据安全的重要性。该方法成功率高且不影响正常性能，为VLA模型的安全研究提供了重要参考。

Abstract: Recent advances in vision-language-action (VLA) models have greatly improved
embodied AI, enabling robots to follow natural language instructions and
perform diverse tasks. However, their reliance on uncurated training datasets
raises serious security concerns. Existing backdoor attacks on VLAs mostly
assume white-box access and result in task failures instead of enforcing
specific actions. In this work, we reveal a more practical threat: attackers
can manipulate VLAs by simply injecting physical objects as triggers into the
training dataset. We propose goal-oriented backdoor attacks (GoBA), where the
VLA behaves normally in the absence of physical triggers but executes
predefined and goal-oriented actions in the presence of physical triggers.
Specifically, based on a popular VLA benchmark LIBERO, we introduce BadLIBERO
that incorporates diverse physical triggers and goal-oriented backdoor actions.
In addition, we propose a three-level evaluation that categorizes the victim
VLA's actions under GoBA into three states: nothing to do, try to do, and
success to do. Experiments show that GoBA enables the victim VLA to
successfully achieve the backdoor goal in 97 percentage of inputs when the
physical trigger is present, while causing zero performance degradation on
clean inputs. Finally, by investigating factors related to GoBA, we find that
the action trajectory and trigger color significantly influence attack
performance, while trigger size has surprisingly little effect. The code and
BadLIBERO dataset are accessible via the project page at
https://goba-attack.github.io/.

</details>


<div id='hep-th'></div>

# hep-th [[Back]](#toc)

### [30] [Quantization of charged fields in the presence of intense electromagnetic fields](https://arxiv.org/abs/2510.09447v1)
*Álvaro Álvarez-Domínguez*

Main category: hep-th

TL;DR: 该论文应用弯曲时空中的量子场论技术研究外部场中的粒子产生，重点关注施温格效应（强电场产生粒子-反粒子对）。通过发展非平凡背景中带电场量子化的理论框架，分析量子真空定义的模糊性，并研究不同量子化方案如何实现幺正时间演化。


<details>
  <summary>Details</summary>
Motivation: 研究施温格效应及其在黑洞物理中的意义，探索外部电磁场和引力场在定义粒子和真空态中的基本作用，揭示平直时空直觉的局限性，并为广义相对论与量子场论的概念鸿沟搭建桥梁。

Method: 应用弯曲时空量子场论技术，发展带电场在非平凡背景中的量子化理论框架，将宇宙学中的低能态概念扩展到施温格设置，推广量子弗拉索夫方程，分析不同量子化方案的幺正性，并采用操作视角研究量子模糊性的物理意义。

Result: 证明了施温格效应在当前条件下阻止了光形成黑洞，研究了费米子电荷超辐射现象，展示了量子效应如何导致黑洞放电（无经典类比的过程）。揭示了量子模糊性与不同相互作用和测量模式相关，具有真实的物理意义。

Conclusion: 该论文强调了外部电磁场和引力场在定义粒子和真空态中的基本作用，识别了具有黑洞物理意义的纯量子现象，为构建一致的时空量子描述提供了新工具，有助于弥合广义相对论与量子场论之间的概念差距。

Abstract: This thesis applies techniques from quantum field theory in curved spacetimes
to study particle creation in external fields, focusing on the Schwinger effect
(i.e., the production of particle-antiparticle pairs by intense electric
fields). Although experimental verification remains out of reach, theoretical
analysis advances our understanding of this phenomenon and its broader
implications.
  The work develops the theoretical framework for quantizing charged fields in
nontrivial backgrounds, addressing the ambiguities in defining the quantum
vacuum and extending the concept of states of low energy from cosmology to the
Schwinger setting. It examines how different quantizations allow for unitary
time evolution, and generalizes the quantum Vlasov equation to encompass a
wider range of schemes. An operational perspective reveals that quantum
ambiguities have genuine physical meaning, being linked to different modes of
interaction and measurement. The study also analyzes dynamical transitions
between static regimes and their impact on observables, with applications to
analog cosmological expansion in Bose-Einstein condensates and the Schwinger
effect itself. In the context of black holes, the thesis shows that the
Schwinger effect prevents the formation of black holes from light under current
conditions and investigates fermionic charge superradiance, demonstrating how
quantum effects lead to black-hole discharge (a process without classical
analogue).
  Overall, the thesis underscores the fundamental role of external
electromagnetic and gravitational fields in defining particles and vacua,
revealing the limits of flat-spacetime intuition and identifying purely quantum
phenomena with implications for black-hole physics. It contributes to bridging
the conceptual gap between general relativity and quantum field theory and
offers new tools toward a consistent quantum description of spacetime.

</details>


<div id='hep-ph'></div>

# hep-ph [[Back]](#toc)

### [31] [Lie symmetry analysis of the two-Higgs-doublet model field equations](https://arxiv.org/abs/2510.09542v1)
*M. Aa. Solberg*

Main category: hep-ph

TL;DR: 本文应用李对称性分析研究双希格斯二重态模型(2HDM)的欧拉-拉格朗日方程，确定了其标量李点对称性。研究发现2HDM中不存在标量李点散度对称性或非变分李点对称性，并重新推导了已知的严格变分李点对称性。


<details>
  <summary>Details</summary>
Motivation: 研究2HDM的对称性结构，因为变分对称性通常在量子化过程中保持不变，且变分李对称性会产生守恒定律。

Method: 使用偏微分方程的李对称性分析方法，分析2HDM的欧拉-拉格朗日方程，确定李点对称性。

Result: 确认2HDM中只有严格变分李点对称性，不存在散度对称性或非变分对称性。证明了三个可简化粒子物理模型李对称性计算的通用结果。

Conclusion: 李对称性分析是确定李对称性的广泛应用方法，可处理多变量、多参数和重参数化自由度的模型，缺失的离散对称性可通过所得李对称性代数的自同构群识别。

Abstract: We apply Lie symmetry analysis of partial differential equations (PDEs) to
the Euler-Lagrange equations of the two-Higgs-doublet model (2HDM), to
determine its scalar Lie point symmetries. A Lie point symmetry is a
structure-preserving transformation of the spacetime variables and the fields
of the model, which is also continuous and connected to the identity.
Symmetries of PDEs may in general be divided into strict variational
symmetries, divergence symmetries and non-variational symmetries, where the
first two are collectively referred to as variational symmetries. Variational
symmetries are usually preserved under quantization, and variational Lie
symmetries yield conservation laws. We demonstrate that there are no scalar Lie
point divergence symmetries or non-variational Lie point symmetries in the
2HDM, and re-derive its well-known strict variational Lie point symmetries,
thus confirming the consistency of our implementation of Lie's method.
Moreover, we prove three general results which may simplify Lie symmetry
calculations for a wide class of particle physics models. Lie symmetry analysis
of PDEs is a broadly applicable method for determining Lie symmetries. As
demonstrated here by example, it can be applied to models with many variables,
parameters and reparametrization freedom, while any missing discrete symmetries
may be identified through the automorphism groups of the resulting Lie symmetry
algebras.

</details>


<div id='cs.RO'></div>

# cs.RO [[Back]](#toc)

### [32] [Zero-shot Structure Learning and Planning for Autonomous Robot Navigation using Active Inference](https://arxiv.org/abs/2510.09574v1)
*Daria de tinguy,Tim Verbelen,Emilio Gamba,Bart Dhoedt*

Main category: cs.RO

TL;DR: AIMAPP是一个基于主动推理的生物启发式自主导航框架，将建图、定位和决策统一在单一生成模型中，无需预定义地图或训练即可在陌生环境中进行探索和导航。


<details>
  <summary>Details</summary>
Motivation: 解决机器人在陌生环境中同时进行探索、定位和规划的问题，无需依赖预定义地图或大量训练，实现完全自监督的导航。

Method: 采用主动推理方法，结合海马体导航机制，使用拓扑推理、位置细胞编码和情景记忆来指导行为。在线构建稀疏拓扑地图，动态学习状态转移，通过最小化期望自由能来规划动作。

Result: 开发了ROS兼容的导航系统，在各种大规模真实和模拟环境中表现出鲁棒性能，能够适应模糊观测、环境变化和传感器噪声。

Conclusion: AIMAPP提供了一个生物启发的模块化解决方案，可在非结构化环境中实现可扩展的自监督导航，展示了在不确定条件下的强大适应能力。

Abstract: Autonomous navigation in unfamiliar environments requires robots to
simultaneously explore, localise, and plan under uncertainty, without relying
on predefined maps or extensive training. We present a biologically inspired,
Active Inference-based framework, Active Inference MAPping and Planning
(AIMAPP). This model unifies mapping, localisation, and decision-making within
a single generative model. Inspired by hippocampal navigation, it uses
topological reasoning, place-cell encoding, and episodic memory to guide
behaviour. The agent builds and updates a sparse topological map online, learns
state transitions dynamically, and plans actions by minimising Expected Free
Energy. This allows it to balance goal-directed and exploratory behaviours. We
implemented a ROS-compatible navigation system that is sensor and
robot-agnostic, capable of integrating with diverse hardware configurations. It
operates in a fully self-supervised manner, is resilient to drift, and supports
both exploration and goal-directed navigation without any pre-training. We
demonstrate robust performance in large-scale real and simulated environments
against state-of-the-art planning models, highlighting the system's
adaptability to ambiguous observations, environmental changes, and sensor
noise. The model offers a biologically inspired, modular solution to scalable,
self-supervised navigation in unstructured settings. AIMAPP is available at
https://github.com/decide-ugent/AIMAPP.

</details>


### [33] [Guiding Energy-Efficient Locomotion through Impact Mitigation Rewards](https://arxiv.org/abs/2510.09543v1)
*Chenghao Wang,Arjun Viswanathan,Eric Sihite,Alireza Ramezani*

Main category: cs.RO

TL;DR: 该论文提出了一种结合冲击缓解因子(IMF)和对抗运动先验(AMP)的方法，使强化学习策略能够同时学习动物的显性运动轨迹和隐性被动动力学，实现了高达32%的能源效率提升。


<details>
  <summary>Details</summary>
Motivation: 动物通过其隐含的被动动力学实现节能运动，但现有的模仿学习方法主要捕捉显性步态模式，而忽略了隐性被动动力学。

Method: 通过引入基于物理信息的冲击缓解因子(IMF)作为奖励项，并将其与对抗运动先验(AMP)结合，使强化学习策略能够学习动物的显性运动轨迹和隐性被动动力学。

Result: 在AMP和手工设计的奖励结构中都实现了高达32%的能源效率提升，通过运输成本(CoT)来衡量。

Conclusion: 该方法成功地将被动动力学整合到运动模仿学习中，显著提高了机器人的能源效率。

Abstract: Animals achieve energy-efficient locomotion by their implicit passive
dynamics, a marvel that has captivated roboticists for decades.Recently,
methods incorporated Adversarial Motion Prior (AMP) and Reinforcement learning
(RL) shows promising progress to replicate Animals' naturalistic motion.
However, such imitation learning approaches predominantly capture explicit
kinematic patterns, so-called gaits, while overlooking the implicit passive
dynamics. This work bridges this gap by incorporating a reward term guided by
Impact Mitigation Factor (IMF), a physics-informed metric that quantifies a
robot's ability to passively mitigate impacts. By integrating IMF with AMP, our
approach enables RL policies to learn both explicit motion trajectories from
animal reference motion and the implicit passive dynamic. We demonstrate energy
efficiency improvements of up to 32%, as measured by the Cost of Transport
(CoT), across both AMP and handcrafted reward structure.

</details>


### [34] [Dynamic Quadrupedal Legged and Aerial Locomotion via Structure Repurposing](https://arxiv.org/abs/2510.09526v1)
*Chenghao Wang,Kaushik Venkatesh Krishnamurthy,Shreyansh Pitroda,Adarsh Salagame,Ioannis Mandralis,Eric Sihite,Alireza Ramezani,Morteza Gharib*

Main category: cs.RO

TL;DR: 本文介绍了Husky v.2多模态地面-空中机器人的硬件设计，该机器人通过结构重构实现了动态四足行走和悬停飞行功能。


<details>
  <summary>Details</summary>
Motivation: 解决多模态机器人在不同操作模式下需求冲突的挑战，实现地面移动和空中飞行的集成。

Method: 采用姿态操纵和推力矢量技术，通过结构重构将腿部结构重新用于动态四足行走和飞行。

Result: 成功实现了动态四足行走和悬停功能，验证了结构重构方法的有效性。

Conclusion: Husky v.2机器人通过创新的结构重构设计，成功解决了多模态机器人面临的模式冲突问题，为地面-空中机器人提供了可行的解决方案。

Abstract: Multi-modal ground-aerial robots have been extensively studied, with a
significant challenge lying in the integration of conflicting requirements
across different modes of operation. The Husky robot family, developed at
Northeastern University, and specifically the Husky v.2 discussed in this
study, addresses this challenge by incorporating posture manipulation and
thrust vectoring into multi-modal locomotion through structure repurposing.
This quadrupedal robot features leg structures that can be repurposed for
dynamic legged locomotion and flight. In this paper, we present the hardware
design of the robot and report primary results on dynamic quadrupedal legged
locomotion and hovering.

</details>


### [35] [Autonomous Soft Robotic Guidewire Navigation via Imitation Learning](https://arxiv.org/abs/2510.09497v1)
*Noah Barnes,Ji Woong Kim,Lingyun Di,Hannah Qu,Anuruddha Bhattacharjee,Miroslaw Janowski,Dheeraj Gandhi,Bailey Felix,Shaopeng Jiang,Olivia Young,Mark Fuge,Ryan D. Sochol,Jeremy D. Brown,Axel Krieger*

Main category: cs.RO

TL;DR: 提出基于Transformer的模仿学习框架，用于软体机器人导丝在血管内导航，在动脉瘤定位任务中达到83%的成功率。


<details>
  <summary>Details</summary>
Motivation: 解决血管内手术中机器人导丝导航的建模和控制挑战，提高血管内导航的精确性和安全性。

Method: 开发基于Transformer的模仿学习框架，包含目标条件、相对动作输出和自动对比染料注射，在36种不同分叉几何结构上训练647个演示。

Result: 在未见过的血管几何结构上，模型能够自主将机器人尖端导航至动脉瘤位置，成功率达到83%，优于多个基线方法。

Conclusion: 该方法能够实现可泛化的软体机器人导丝导航，为血管内手术自动化提供了有前景的解决方案。

Abstract: In endovascular surgery, endovascular interventionists push a thin tube
called a catheter, guided by a thin wire to a treatment site inside the
patient's blood vessels to treat various conditions such as blood clots,
aneurysms, and malformations. Guidewires with robotic tips can enhance
maneuverability, but they present challenges in modeling and control.
Automation of soft robotic guidewire navigation has the potential to overcome
these challenges, increasing the precision and safety of endovascular
navigation. In other surgical domains, end-to-end imitation learning has shown
promising results. Thus, we develop a transformer-based imitation learning
framework with goal conditioning, relative action outputs, and automatic
contrast dye injections to enable generalizable soft robotic guidewire
navigation in an aneurysm targeting task. We train the model on 36 different
modular bifurcated geometries, generating 647 total demonstrations under
simulated fluoroscopy, and evaluate it on three previously unseen vascular
geometries. The model can autonomously drive the tip of the robot to the
aneurysm location with a success rate of 83% on the unseen geometries,
outperforming several baselines. In addition, we present ablation and baseline
studies to evaluate the effectiveness of each design and data collection
choice. Project website: https://softrobotnavigation.github.io/

</details>


### [36] [FOGMACHINE -- Leveraging Discrete-Event Simulation and Scene Graphs for Modeling Hierarchical, Interconnected Environments under Partial Observations from Mobile Agents](https://arxiv.org/abs/2510.09483v1)
*Lars Ohnemus,Nils Hantke,Max Weißer,Kai Furmans*

Main category: cs.RO

TL;DR: FOGMACHINE是一个开源框架，将动态场景图与离散事件模拟相结合，用于建模复杂环境中的对象动态、智能体观察和交互，特别关注不确定性传播和多智能体行为。


<details>
  <summary>Details</summary>
Motivation: 当前动态场景图方法难以捕捉随机动态、部分可观察性和多智能体活动，而这些对于具身AI在不确定性和延迟感知下的行动至关重要。

Method: 通过融合动态场景图和离散事件模拟来建模对象动态、智能体观察和交互，支持大规模仿真。

Result: 在城市场景实验中展示了真实的时间和空间模式，同时揭示了在稀疏观察下信念估计的挑战。

Conclusion: FOGMACHINE通过结合结构化表示和高效模拟，为复杂不确定环境中的基准测试、模型训练和具身AI发展建立了有效工具。

Abstract: Dynamic Scene Graphs (DSGs) provide a structured representation of
hierarchical, interconnected environments, but current approaches struggle to
capture stochastic dynamics, partial observability, and multi-agent activity.
These aspects are critical for embodied AI, where agents must act under
uncertainty and delayed perception. We introduce FOGMACHINE , an open-source
framework that fuses DSGs with discrete-event simulation to model object
dynamics, agent observations, and interactions at scale. This setup enables the
study of uncertainty propagation, planning under limited perception, and
emergent multi-agent behavior. Experiments in urban scenarios illustrate
realistic temporal and spatial patterns while revealing the challenges of
belief estimation under sparse observations. By combining structured
representations with efficient simulation, FOGMACHINE establishes an effective
tool for benchmarking, model training, and advancing embodied AI in complex,
uncertain environments.

</details>


### [37] [Bridging Research and Practice in Simulation-based Testing of Industrial Robot Navigation Systems](https://arxiv.org/abs/2510.09396v1)
*Sajad Khatiri,Francisco Eli Vina Barrientos,Maximilian Wulf,Paolo Tonella,Sebastiano Panichella*

Main category: cs.RO

TL;DR: 本文介绍了将无人机测试框架Surrealist应用于ANYmal四足机器人工业检测的工业应用，通过基于搜索的算法自动生成具有挑战性的避障场景，在工业评估中成功测试了五种专有算法并验证了其价值。


<details>
  <summary>Details</summary>
Motivation: 传统测试方法难以覆盖动态环境中机器人导航的全部操作需求，需要更有效的测试框架来确保机器人导航的鲁棒性。

Method: 采用基于搜索的算法自动生成具有挑战性的避障场景，将Surrealist框架从无人机应用扩展到ANYmal四足机器人工业检测领域。

Result: 在试点阶段，生成的测试套件揭示了一个实验算法的关键弱点（成功率40.3%），并证明了另一个算法的优越鲁棒性（成功率71.2%）。在六个月的工业评估中，成功测试了五种专有算法，正式调查确认该框架增强了开发过程、发现关键故障、提供客观基准并加强了整体验证流程。

Conclusion: Surrealist框架在工业环境中的成功应用证明其能够有效发现传统测试方法遗漏的故障，为机器人导航算法的开发和验证提供了有价值的工具。

Abstract: Ensuring robust robotic navigation in dynamic environments is a key
challenge, as traditional testing methods often struggle to cover the full
spectrum of operational requirements. This paper presents the industrial
adoption of Surrealist, a simulation-based test generation framework originally
for UAVs, now applied to the ANYmal quadrupedal robot for industrial
inspection. Our method uses a search-based algorithm to automatically generate
challenging obstacle avoidance scenarios, uncovering failures often missed by
manual testing. In a pilot phase, generated test suites revealed critical
weaknesses in one experimental algorithm (40.3% success rate) and served as an
effective benchmark to prove the superior robustness of another (71.2% success
rate). The framework was then integrated into the ANYbotics workflow for a
six-month industrial evaluation, where it was used to test five proprietary
algorithms. A formal survey confirmed its value, showing it enhances the
development process, uncovers critical failures, provides objective benchmarks,
and strengthens the overall verification pipeline.

</details>


### [38] [Placeit! A Framework for Learning Robot Object Placement Skills](https://arxiv.org/abs/2510.09267v1)
*Amina Ferrad,Johann Huber,François Hélénon,Julien Gleyze,Mahdi Khoramshahi,Stéphane Doncieux*

Main category: cs.RO

TL;DR: Placeit!是一个基于进化计算的框架，用于为刚性物体自动生成有效的放置位置，支持从桌面放置到堆叠和插入等多种任务，显著优于现有方法，在实际部署中达到90%的成功率。


<details>
  <summary>Details</summary>
Motivation: 机器人学习面临获取大规模高质量数据的瓶颈，传统方法需要大量人工劳动。受Graspit!使用仿真自动生成抓取姿态的启发，需要开发类似方法解决物体放置问题。

Method: 采用进化计算框架，通过质量多样性优化生成多样化的有效放置姿态，支持多种放置场景包括桌面放置、堆叠和插入。

Result: 在所有场景下显著优于最先进方法，基于该框架构建的拾放管道在120次真实世界部署中达到90%的成功率。

Conclusion: Placeit!是开放环境拾放任务的强大工具，也是为训练基于仿真的机器人基础模型生成所需数据的宝贵引擎。

Abstract: Robotics research has made significant strides in learning, yet mastering
basic skills like object placement remains a fundamental challenge. A key
bottleneck is the acquisition of large-scale, high-quality data, which is often
a manual and laborious process. Inspired by Graspit!, a foundational work that
used simulation to automatically generate dexterous grasp poses, we introduce
Placeit!, an evolutionary-computation framework for generating valid placement
positions for rigid objects. Placeit! is highly versatile, supporting tasks
from placing objects on tables to stacking and inserting them. Our experiments
show that by leveraging quality-diversity optimization, Placeit! significantly
outperforms state-of-the-art methods across all scenarios for generating
diverse valid poses. A pick&place pipeline built on our framework achieved a
90% success rate over 120 real-world deployments. This work positions Placeit!
as a powerful tool for open-environment pick-and-place tasks and as a valuable
engine for generating the data needed to train simulation-based foundation
models in robotics.

</details>


### [39] [HANDO: Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation](https://arxiv.org/abs/2510.09221v1)
*Jingyuan Sun,Chaoran Wang,Mingyu Zhang,Cui Miao,Hongyu Ji,Zihan Qu,Han Sun,Bing Wang,Qingyi Si*

Main category: cs.RO

TL;DR: HANDO是一个用于配备机械臂的腿式机器人的分层框架，包含自主探索层和全身运动操作层，实现人类中心的移动操作任务


<details>
  <summary>Details</summary>
Motivation: 在非结构化环境中实现无缝的运动操作需要机器人结合自主探索和全身控制来进行物理交互

Method: 采用两层框架：第一层使用目标导向的自主探索策略导航到语义指定目标；第二层使用统一的全身运动操作策略协调手臂和腿部进行精确交互

Result: 已完成导航模块的初步部署，将继续推进全身运动操作的更精细部署

Conclusion: HANDO框架为腿式机器人提供了在动态环境中执行复杂移动操作任务的有效解决方案

Abstract: Seamless loco-manipulation in unstructured environments requires robots to
leverage autonomous exploration alongside whole-body control for physical
interaction. In this work, we introduce HANDO (Hierarchical Autonomous
Navigation and Dexterous Omni-loco-manipulation), a two-layer framework
designed for legged robots equipped with manipulators to perform human-centered
mobile manipulation tasks. The first layer utilizes a goal-conditioned
autonomous exploration policy to guide the robot to semantically specified
targets, such as a black office chair in a dynamic environment. The second
layer employs a unified whole-body loco-manipulation policy to coordinate the
arm and legs for precise interaction tasks-for example, handing a drink to a
person seated on the chair. We have conducted an initial deployment of the
navigation module, and will continue to pursue finer-grained deployment of
whole-body loco-manipulation.

</details>


### [40] [Decentralized Multi-Robot Relative Navigation in Unknown, Structurally Constrained Environments under Limited Communication](https://arxiv.org/abs/2510.09188v1)
*Zihao Mao,Yunheng Wang,Yunting Ji,Yi Yang,Wenjie Song*

Main category: cs.RO

TL;DR: 提出了一种完全去中心化的分层相对导航框架，在未知、结构受限且无GPS的环境中实现战略远见和战术敏捷性，无需统一坐标系。


<details>
  <summary>Details</summary>
Motivation: 解决多机器人导航在全局战略远见和局部战术敏捷性之间的基本权衡问题，特别是在通信受限环境下。集中式方法通信开销大，分布式方法缺乏全局意识容易陷入死锁和拓扑陷阱。

Method: 采用分层架构：战略层通过机会性相遇构建和交换轻量级拓扑地图，培养涌现的全局意识；战术层基于局部度量信息，采用基于采样的逃逸点策略实时生成动态可行轨迹。

Result: 广泛的仿真和真实世界实验表明，该系统在成功率和工作效率方面显著优于其他方法，特别是在通信受限且具有复杂拓扑结构的环境中。

Conclusion: 该分层相对导航框架成功实现了战略远见和战术敏捷性的平衡，在通信受限的复杂环境中表现出色，为多机器人导航提供了有效的解决方案。

Abstract: Multi-robot navigation in unknown, structurally constrained, and GPS-denied
environments presents a fundamental trade-off between global strategic
foresight and local tactical agility, particularly under limited communication.
Centralized methods achieve global optimality but suffer from high
communication overhead, while distributed methods are efficient but lack the
broader awareness to avoid deadlocks and topological traps. To address this, we
propose a fully decentralized, hierarchical relative navigation framework that
achieves both strategic foresight and tactical agility without a unified
coordinate system. At the strategic layer, robots build and exchange
lightweight topological maps upon opportunistic encounters. This process
fosters an emergent global awareness, enabling the planning of efficient,
trap-avoiding routes at an abstract level. This high-level plan then inspires
the tactical layer, which operates on local metric information. Here, a
sampling-based escape point strategy resolves dense spatio-temporal conflicts
by generating dynamically feasible trajectories in real time, concurrently
satisfying tight environmental and kinodynamic constraints. Extensive
simulations and real-world experiments demonstrate that our system
significantly outperforms in success rate and efficiency, especially in
communication-limited environments with complex topological structures.

</details>


<div id='cs.AI'></div>

# cs.AI [[Back]](#toc)

### [41] [Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges](https://arxiv.org/abs/2510.09404v1)
*Christian Bluethgen,Dave Van Veen,Daniel Truhn,Jakob Nikolas Kather,Michael Moor,Malgorzata Polacin,Akshay Chaudhari,Thomas Frauenfelder,Curtis P. Langlotz,Michael Krauthammer,Farhad Nooralahzadeh*

Main category: cs.AI

TL;DR: 本文综述了基于大语言模型（LLM）的智能体系统在放射学中的应用，探讨了如何通过外部工具和反馈机制增强LLM能力，实现从半自动化工作流到自适应智能体的不同自主程度。


<details>
  <summary>Details</summary>
Motivation: 放射学具有多模态数据流和协调工作流的特点，非常适合应用能够适应上下文并自动化复杂重复任务的智能体系统。虽然LLM在放射学单个任务中已表现出色，但单独使用LLM未能充分利用其在复杂多步骤工作流中的潜力。

Method: 通过为LLM配备外部工具和反馈机制，使其能够驱动表现出不同程度自主性的系统，从半自动化工作流到能够管理复杂过程的自适应智能体。

Result: LLM驱动的智能体系统在放射学中具有广阔应用前景，能够处理依赖多个信息源动态上下文的复杂决策过程。

Conclusion: LLM驱动的智能体系统为放射学工作流自动化提供了重要机遇，但需要解决错误级联、工具使用效率和医疗IT集成等挑战。

Abstract: Building agents, systems that perceive and act upon their environment with a
degree of autonomy, has long been a focus of AI research. This pursuit has
recently become vastly more practical with the emergence of large language
models (LLMs) capable of using natural language to integrate information,
follow instructions, and perform forms of "reasoning" and planning across a
wide range of tasks. With its multimodal data streams and orchestrated
workflows spanning multiple systems, radiology is uniquely suited to benefit
from agents that can adapt to context and automate repetitive yet complex
tasks. In radiology, LLMs and their multimodal variants have already
demonstrated promising performance for individual tasks such as information
extraction and report summarization. However, using LLMs in isolation
underutilizes their potential to support complex, multi-step workflows where
decisions depend on evolving context from multiple information sources.
Equipping LLMs with external tools and feedback mechanisms enables them to
drive systems that exhibit a spectrum of autonomy, ranging from semi-automated
workflows to more adaptive agents capable of managing complex processes. This
review examines the design of such LLM-driven agentic systems, highlights key
applications, discusses evaluation methods for planning and tool use, and
outlines challenges such as error cascades, tool-use efficiency, and health IT
integration.

</details>


<div id='cs.MA'></div>

# cs.MA [[Back]](#toc)

### [42] [Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy](https://arxiv.org/abs/2510.09469v1)
*Bharath Muppasani,Ritirupa Dey,Biplav Srivastava,Vignesh Narayanan*

Main category: cs.MA

TL;DR: 提出了一种结合去中心化路径规划和轻量级中心化协调器的混合框架，用于多智能体路径规划问题，通过强化学习进行去中心化规划，利用最小化信息共享实现有效冲突解决。


<details>
  <summary>Details</summary>
Motivation: 传统中心化算法在大型场景中计算成本高，而分布式学习方法虽然可扩展性好但解决方案质量较低，需要平衡解决方案质量和计算效率。

Method: 使用强化学习进行去中心化路径规划，配合轻量级中心化协调器动态共享静态冲突单元标志或简短冲突轨迹等最小化信息。

Result: 该方法在减少智能体间信息共享的同时，能够在大规模场景中持续找到可行的无碰撞解决方案。

Conclusion: 混合框架有效解决了多智能体路径规划中的可扩展性和解决方案质量之间的权衡问题。

Abstract: Multi-agent pathfinding (MAPF) remains a critical problem in robotics and
autonomous systems, where agents must navigate shared spaces efficiently while
avoiding conflicts. Traditional centralized algorithms that have global
information, such as Conflict-Based Search (CBS), provide high-quality
solutions but become computationally expensive in large-scale scenarios due to
the combinatorial explosion of conflicts that need resolution. Conversely,
distributed approaches that have local information, particularly learning-based
methods, offer better scalability by operating with relaxed information
availability, yet often at the cost of solution quality. To address these
limitations, we propose a hybrid framework that combines decentralized path
planning with a lightweight centralized coordinator. Our framework leverages
reinforcement learning (RL) for decentralized planning, enabling agents to
adapt their planning based on minimal, targeted alerts--such as static
conflict-cell flags or brief conflict tracks--that are dynamically shared
information from the central coordinator for effective conflict resolution. We
empirically study the effect of the information available to an agent on its
planning performance. Our approach reduces the inter-agent information sharing
compared to fully centralized and distributed methods, while still consistently
finding feasible, collision-free solutions--even in large-scale scenarios
having higher agent counts.

</details>


<div id='cs.HC'></div>

# cs.HC [[Back]](#toc)

### [43] [Differential Analysis of Pseudo Haptic Feedback: Novel Comparative Study of Visual and Auditory Cue Integration for Psychophysical Evaluation](https://arxiv.org/abs/2510.09570v1)
*Nishant Gautam,Somya Sharma,Peter Corcoran,Kaspar Althoefer*

Main category: cs.HC

TL;DR: 本研究通过心理物理学实验量化了视觉和听觉刺激如何结合在普通平板设备上诱发伪触觉压力感受，发现多感官整合能有效增强伪触觉反馈效果。


<details>
  <summary>Details</summary>
Motivation: 伪触觉技术利用精心设计的视觉或听觉线索来欺骗大脑"感受"从未物理施加的力，为传统触觉硬件提供了低成本替代方案。本研究旨在探索视觉和听觉刺激如何协同工作来增强伪触觉体验。

Method: 使用基于Unity的Rollball游戏，参与者（n=4）在三种不同纹理的地形上引导虚拟球，同时通过Robotous RFT40力-扭矩传感器实时捕获手指力。每个地形配有不同的滚动声音配置文件，碰撞时触发额外的"敲击"音效以增强真实感。

Result: 触觉力随线索强度系统增加：仅视觉试验为0.40N、0.79N和0.88N，仅音频试验为0.41N、0.81N和0.90N。更高的音频频率和更密集的视觉纹理都引发了更强的肌肉激活，它们的组合进一步减少了感知表面变化所需的力量。

Conclusion: 研究结果表明，消费级等长设备可以在没有专门执行器的情况下可靠地诱导和测量分级伪触觉反馈，为经济实惠的康复工具、训练模拟器和辅助界面开辟了道路。

Abstract: Pseudo-haptics exploit carefully crafted visual or auditory cues to trick the
brain into "feeling" forces that are never physically applied, offering a
low-cost alternative to traditional haptic hardware. Here, we present a
comparative psychophysical study that quantifies how visual and auditory
stimuli combine to evoke pseudo-haptic pressure sensations on a commodity
tablet. Using a Unity-based Rollball game, participants (n = 4) guided a
virtual ball across three textured terrains while their finger forces were
captured in real time with a Robotous RFT40 force-torque sensor. Each terrain
was paired with a distinct rolling-sound profile spanning 440 Hz - 4.7 kHz, 440
Hz - 13.1 kHz, or 440 Hz - 8.9 kHz; crevice collisions triggered additional
"knocking" bursts to heighten realism. Average tactile forces increased
systematically with cue intensity: 0.40 N, 0.79 N and 0.88 N for visual-only
trials and 0.41 N, 0.81 N and 0.90 N for audio-only trials on Terrains 1-3,
respectively. Higher audio frequencies and denser visual textures both elicited
stronger muscle activation, and their combination further reduced the force
needed to perceive surface changes, confirming multisensory integration. These
results demonstrate that consumer-grade isometric devices can reliably induce
and measure graded pseudo-haptic feedback without specialized actuators,
opening a path toward affordable rehabilitation tools, training simulators and
assistive interfaces.

</details>
