<div id=toc></div>

# Table of Contents

- [cs.CV](#cs.CV) [Total: 18]
- [cs.CL](#cs.CL) [Total: 3]
- [cs.HC](#cs.HC) [Total: 1]
- [stat.ME](#stat.ME) [Total: 2]
- [cs.GR](#cs.GR) [Total: 1]
- [cs.LG](#cs.LG) [Total: 2]
- [cs.AI](#cs.AI) [Total: 2]
- [cs.RO](#cs.RO) [Total: 11]
- [econ.GN](#econ.GN) [Total: 1]
- [quant-ph](#quant-ph) [Total: 2]
- [math-ph](#math-ph) [Total: 1]
- [cond-mat.mtrl-sci](#cond-mat.mtrl-sci) [Total: 1]


<div id='cs.CV'></div>

# cs.CV [[Back]](#toc)

### [1] [MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning](https://arxiv.org/abs/2510.08567v1)
*Tajamul Ashraf,Umair Nawaz,Abdelrahman M. Shaker,Rao Anwer,Philip Torr,Fahad Shahbaz Khan,Salman Khan*

Main category: cs.CV

TL;DR: 该论文提出了一个视觉中心的多模态智能体调优框架，通过自动合成多模态轨迹、生成逐步偏好对，训练VLM控制器实现强大的工具使用推理。


<details>
  <summary>Details</summary>
Motivation: 解决当前视觉语言模型作为控制器时面临的高质量多模态轨迹稀缺和手动标注成本高的问题，提升多模态工具使用的推理能力。

Method: 构建M-TRACE数据集（28.5K多模态任务，177K验证轨迹），开发MATRIX Agent控制器，引入Pref-X偏好对数据集（11K自动生成），通过逐步偏好学习进行优化。

Result: 在Agent-X、GTA和GAIA三个基准测试中，MATRIX持续超越开源和闭源VLM，展示了可扩展且有效的多模态工具使用能力。

Conclusion: 该框架通过自动数据合成和偏好学习，显著提升了多模态智能体的工具使用推理性能，为大规模多模态任务提供了有效解决方案。

Abstract: Vision language models (VLMs) are increasingly deployed as controllers with
access to external tools for complex reasoning and decision-making, yet their
effectiveness remains limited by the scarcity of high-quality multimodal
trajectories and the cost of manual annotation. We address this challenge with
a vision-centric agent tuning framework that automatically synthesizes
multimodal trajectories, generates step-wise preference pairs, and trains a VLM
controller for robust tool-use reasoning. Our pipeline first constructs
M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified
trajectories, enabling imitation-based trajectory tuning. Building on this, we
develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool
reasoning. To achieve finer alignment, we further introduce Pref-X, a set of
11K automatically generated preference pairs, and optimize MATRIX on it via
step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA,
MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating
scalable and effective multimodal tool use. Our data and code is avaliable at
https://github.com/mbzuai-oryx/MATRIX.

</details>


### [2] [NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](https://arxiv.org/abs/2510.08565v1)
*Changyao Tian,Hao Li,Gen Luo,Xizhou Zhu,Weijie Su,Hanming Deng,Jinguo Zhu,Jie Shao,Ziran Zhu,Yunpeng Liu,Lewei Lu,Wenhai Wang,Hongsheng Li,Jifeng Dai*

Main category: cs.CV

TL;DR: 该论文提出了一种端到端的原生多模态大语言模型（MLLM）训练方法NaViL，系统研究了在数据约束下的设计空间和扩展特性，发现视觉编码器与LLM之间存在正相关的扩展关系。


<details>
  <summary>Details</summary>
Motivation: 现有MLLM通常采用组合式训练，将预训练视觉编码器与预训练LLM连接，但这种分离训练难以探索多模态扩展特性。论文旨在研究端到端原生训练的设计空间和扩展规律。

Method: 系统研究MLLM的各种设计选择，获得最佳元架构以平衡性能和训练成本；探索原生MLLM的扩展特性，提出NaViL模型及简单有效的训练方案。

Result: 在14个多模态基准测试中，NaViL展现出与现有MLLM竞争的性能，证实了视觉编码器与LLM的正相关扩展关系。

Conclusion: 研究结果为未来原生MLLM研究提供了深入见解，证明了端到端原生训练的有效性和扩展潜力。

Abstract: Compositional training has been the de-facto paradigm in existing Multimodal
Large Language Models (MLLMs), where pre-trained vision encoders are connected
with pre-trained LLMs through continuous multimodal pre-training. However, the
multimodal scaling property of this paradigm remains difficult to explore due
to the separated training. In this paper, we focus on the native training of
MLLMs in an end-to-end manner and systematically study its design space and
scaling property under a practical setting, i.e., data constraint. Through
careful study of various choices in MLLM, we obtain the optimal
meta-architecture that best balances performance and training cost. After that,
we further explore the scaling properties of the native MLLM and indicate the
positively correlated scaling relationship between visual encoders and LLMs.
Based on these findings, we propose a native MLLM called NaViL, combined with a
simple and cost-effective recipe. Experimental results on 14 multimodal
benchmarks confirm the competitive performance of NaViL against existing MLLMs.
Besides that, our findings and results provide in-depth insights for the future
study of native MLLMs.

</details>


### [3] [SciVideoBench: Benchmarking Scientific Video Reasoning in Large Multimodal Models](https://arxiv.org/abs/2510.08559v1)
*Andong Deng,Taojiannan Yang,Shoubin Yu,Lincoln Spencer,Mohit Bansal,Chen Chen,Serena Yeung-Levy,Xiaohan Wang*

Main category: cs.CV

TL;DR: SciVideoBench是一个专门评估科学视频推理能力的基准测试，包含1000个来自前沿科学实验视频的多选题，涵盖25+专业学科，要求模型具备领域知识、时空感知和复杂逻辑推理能力。


<details>
  <summary>Details</summary>
Motivation: 当前视频基准主要针对通用场景，依赖感知/识别能力，推理任务相对简单，无法有效评估先进的多模态认知技能。科学领域的复杂视频推理仍然是一个重要且具有挑战性的前沿问题。

Method: 构建SciVideoBench基准，包含1000个精心设计的多选题，源自前沿科学实验视频，涵盖25+专业学科，通过半自动系统验证。每个问题需要复杂的领域知识、精确的时空感知和精细的逻辑推理。

Result: 评估显示最先进的专有和开源LMMs（包括Gemini 2.5 Pro和Qwen2.5-VL）存在显著的性能缺陷，表明在视频推理能力方面还有很大的提升空间。

Conclusion: SciVideoBench为LMMs的未来发展提供了有价值的见解和明确方向，推动真正有能力的多模态AI共同科学家的演进，帮助推动前沿AI在更广泛科学领域的边界。

Abstract: Large Multimodal Models (LMMs) have achieved remarkable progress across
various capabilities; however, complex video reasoning in the scientific domain
remains a significant and challenging frontier. Current video benchmarks
predominantly target general scenarios where perception/recognition is heavily
relied on, while with relatively simple reasoning tasks, leading to saturation
and thus failing to effectively evaluate advanced multimodal cognitive skills.
To address this critical gap, we introduce SciVideoBench, a rigorous benchmark
specifically designed to assess advanced video reasoning in scientific
contexts. SciVideoBench consists of 1,000 carefully crafted multiple-choice
questions derived from cutting-edge scientific experimental videos spanning
over 25 specialized academic subjects and verified by a semi-automatic system.
Each question demands sophisticated domain-specific knowledge, precise
spatiotemporal perception, and intricate logical reasoning, effectively
challenging models' higher-order cognitive abilities. Our evaluation highlights
significant performance deficits in state-of-the-art proprietary and
open-source LMMs, including Gemini 2.5 Pro and Qwen2.5-VL, indicating
substantial room for advancement in video reasoning capabilities. Detailed
analyses of critical factors such as reasoning complexity and visual grounding
provide valuable insights and clear direction for future developments in LMMs,
driving the evolution of truly capable multimodal AI co-scientists. We hope
SciVideoBench could fit the interests of the community and help to push the
boundary of cutting-edge AI for border science.

</details>


### [4] [Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation](https://arxiv.org/abs/2510.08553v1)
*Yunzhe Xu,Yiyuan Pan,Zhe Liu*

Main category: cs.CV

TL;DR: Memoir提出了一种基于想象的记忆检索机制，用于视觉语言导航中的记忆持久性问题。该方法使用世界模型想象未来状态作为查询，选择性检索环境观察和行为历史，显著提升了导航性能。


<details>
  <summary>Details</summary>
Motivation: 现有的记忆持久性视觉语言导航方法存在关键限制：缺乏有效的记忆访问机制，要么依赖完整内存整合，要么使用固定范围查找；主要存储环境观察而忽略了包含有价值决策策略的导航行为模式。

Method: 1) 语言条件世界模型想象未来状态，用于编码经验和生成检索查询；2) 混合视角级内存，将观察和行为模式锚定到视角，实现混合检索；3) 经验增强导航模型，通过专用编码器整合检索知识。

Result: 在10个不同测试场景的多样化记忆持久性VLN基准测试中，Memoir在所有场景都取得显著改进：在IR2R上比最佳记忆持久性基线提升5.4% SPL，同时实现8.3倍训练加速和74%推理内存减少。

Conclusion: 预测性检索环境和行为记忆能够实现更有效的导航，分析表明这种想象引导范式具有巨大的提升空间（73.3% vs 93.4%上限）。

Abstract: Vision-and-Language Navigation (VLN) requires agents to follow natural
language instructions through environments, with memory-persistent variants
demanding progressive improvement through accumulated experience. Existing
approaches for memory-persistent VLN face critical limitations: they lack
effective memory access mechanisms, instead relying on entire memory
incorporation or fixed-horizon lookup, and predominantly store only
environmental observations while neglecting navigation behavioral patterns that
encode valuable decision-making strategies. We present Memoir, which employs
imagination as a retrieval mechanism grounded by explicit memory: a world model
imagines future navigation states as queries to selectively retrieve relevant
environmental observations and behavioral histories. The approach comprises: 1)
a language-conditioned world model that imagines future states serving dual
purposes: encoding experiences for storage and generating retrieval queries; 2)
Hybrid Viewpoint-Level Memory that anchors both observations and behavioral
patterns to viewpoints, enabling hybrid retrieval; and 3) an
experience-augmented navigation model that integrates retrieved knowledge
through specialized encoders. Extensive evaluation across diverse
memory-persistent VLN benchmarks with 10 distinctive testing scenarios
demonstrates Memoir's effectiveness: significant improvements across all
scenarios, with 5.4% SPL gains on IR2R over the best memory-persistent
baseline, accompanied by 8.3x training speedup and 74% inference memory
reduction. The results validate that predictive retrieval of both environmental
and behavioral memories enables more effective navigation, with analysis
indicating substantial headroom (73.3% vs 93.4% upper bound) for this
imagination-guided paradigm. Code at https://github.com/xyz9911/Memoir.

</details>


### [5] [ARTDECO: Towards Efficient and High-Fidelity On-the-Fly 3D Reconstruction with Structured Scene Representation](https://arxiv.org/abs/2510.08551v1)
*Guanghao Li,Kerui Ren,Linning Xu,Zhewen Zheng,Changjian Jiang,Xin Gao,Bo Dai,Jian Pu,Mulin Yu,Jiangmiao Pang*

Main category: cs.CV

TL;DR: ARTDECO是一个统一的3D重建框架，结合了前馈模型的高效性和SLAM管道的可靠性，通过3D基础模型进行姿态估计和点预测，使用高斯解码器将多尺度特征转换为结构化3D高斯，采用分层高斯表示和LoD感知渲染策略，在保持交互性能的同时提供高质量重建。


<details>
  <summary>Details</summary>
Motivation: 解决单目图像序列实时3D重建中的关键权衡问题：基于场景优化的方法精度高但计算昂贵，而前馈基础模型可实现实时推理但精度和鲁棒性不足。

Method: 使用3D基础模型进行姿态估计和点预测，结合高斯解码器将多尺度特征转换为结构化3D高斯，设计分层高斯表示和LoD感知渲染策略以提高渲染保真度并减少冗余。

Result: 在8个不同的室内外基准测试中，ARTDECO实现了与SLAM相当的交互性能，与前馈系统相似的鲁棒性，以及接近基于场景优化的重建质量。

Conclusion: ARTDECO为实时数字化真实世界环境提供了一条实用路径，既能保持准确的几何结构，又能提供高视觉保真度。

Abstract: On-the-fly 3D reconstruction from monocular image sequences is a
long-standing challenge in computer vision, critical for applications such as
real-to-sim, AR/VR, and robotics. Existing methods face a major tradeoff:
per-scene optimization yields high fidelity but is computationally expensive,
whereas feed-forward foundation models enable real-time inference but struggle
with accuracy and robustness. In this work, we propose ARTDECO, a unified
framework that combines the efficiency of feed-forward models with the
reliability of SLAM-based pipelines. ARTDECO uses 3D foundation models for pose
estimation and point prediction, coupled with a Gaussian decoder that
transforms multi-scale features into structured 3D Gaussians. To sustain both
fidelity and efficiency at scale, we design a hierarchical Gaussian
representation with a LoD-aware rendering strategy, which improves rendering
fidelity while reducing redundancy. Experiments on eight diverse indoor and
outdoor benchmarks show that ARTDECO delivers interactive performance
comparable to SLAM, robustness similar to feed-forward systems, and
reconstruction quality close to per-scene optimization, providing a practical
path toward on-the-fly digitization of real-world environments with both
accurate geometry and high visual fidelity. Explore more demos on our project
page: https://city-super.github.io/artdeco/.

</details>


### [6] [MM-HELIX: Boosting Multimodal Long-Chain Reflective Reasoning with Holistic Platform and Adaptive Hybrid Policy Optimization](https://arxiv.org/abs/2510.08540v1)
*Xiangyu Zhao,Junming Lin,Tianhao Liang,Yifan Zhou,Wenhao Chai,Yuzhe Gu,Weiyun Wang,Kai Chen,Gen Luo,Wenwei Zhang,Junchi Yan,Hua Yang,Haodong Duan,Xue Yang*

Main category: cs.CV

TL;DR: 该论文研究了多模态大语言模型在长链反思推理方面的能力不足，通过构建MM-HELIX基准测试发现现有模型性能缺陷，并提出自适应混合策略优化方法显著提升了模型推理能力。


<details>
  <summary>Details</summary>
Motivation: 当前多模态大语言模型在数学和逻辑推理任务中表现出色，但其长链反思推理能力（解决复杂现实问题所需）尚未得到充分探索，这限制了模型处理复杂问题的能力。

Method: 首先构建MM-HELIX多模态基准测试（1,260个样本），然后开发步骤引导响应生成管道创建MM-HELIX-100K数据集，最后提出自适应混合策略优化方法，将离线监督和在线优化动态统一到单一训练阶段。

Result: 在Qwen2.5-VL-7B基准模型上，该方法在MM-HELIX基准测试上实现了+18.6%的准确率提升，在通用数学和逻辑任务上平均性能提升+5.7%，显示出良好的泛化能力。

Conclusion: 研究表明多模态大语言模型的反思推理能力可以有效学习和泛化，为开发更强大的多模态大语言模型铺平了道路。

Abstract: While current Multimodal Large Language Models (MLLMs) have demonstrated
proficiency in reasoning tasks such as mathematics and logic, their capacity
for long-chain reflective reasoning, a prerequisite for solving complex
real-world problems, remains largely underexplored. In this work, we first
conduct an extensive empirical investigation to evaluate this capability.
Leveraging a carefully designed data synthesis engine, we construct MM-HELIX, a
multimodal benchmark consisting 1,260 samples of 42 challenging synthetic tasks
that require iterative thinking and backtracking. Empirical results on this
benchmark reveal that existing MLLMs exhibit significant performance deficits
in long-chain reflective reasoning. To address this limitation, we generate
post-training data and further explore learning paradigms for exploiting such
data. We first develop the Step-Elicited Response Generation pipeline to create
MM-HELIX-100K, a large-scale dataset of 100k high-quality, reflective reasoning
traces for instruction-tuning stage. Given that standard Reinforcement Learning
fails on complex tasks due to sparse reward signals and catastrophic forgetting
after Supervised Fine-Tuning, we propose Adaptive Hybrid Policy Optimization
(AHPO), a novel training strategy that dynamically unifies offline supervision
and online optimization into a single stage. This strategy enables the model to
learn from expert data when rewards are sparse and conduct independent
exploration once proficient. When applied to the Qwen2.5-VL-7B baseline, our
method achieves a +18.6\% accuracy improvement on MM-HELIX benchmark and
demonstrates strong generalization with a +5.7\% average performance gain on
general mathematic and logic tasks. Our work demonstrate that reflective
reasoning in MLLMs can be effectively learned and generalized, paving the way
for developing more capable MLLMs.

</details>


### [7] [SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2510.08531v1)
*Hongxing Li,Dingming Li,Zixuan Wang,Yuchen Yan,Hang Wu,Wenqi Zhang,Yongliang Shen,Weiming Lu,Jun Xiao,Yueting Zhuang*

Main category: cs.CV

TL;DR: 本文提出了一种渐进式训练框架SpatialLadder，通过三阶段训练方法显著提升了视觉语言模型的空间推理能力，在多个基准测试中超越了GPT-4o和Gemini-2.0-Flash。


<details>
  <summary>Details</summary>
Motivation: 现有视觉语言模型在空间推理方面存在根本性挑战，主要原因是缺乏从感知到理解的层次化基础。作者认为直接学习空间推理而不建立层次化基础是现有方法的局限性。

Method: 提出了三阶段渐进式训练框架：1）通过目标定位建立空间感知；2）通过多维空间任务发展空间理解；3）通过带可验证奖励的强化学习加强复杂推理。基于构建的SpatialLadder-26k多模态数据集进行训练。

Result: SpatialLadder模型（30亿参数）在空间推理基准测试中实现了最先进性能，相比基础模型平均提升23.4%，超越GPT-4o 20.8%，超越Gemini-2.0-Flash 10.1%。在域外基准测试中保持强泛化能力，提升7.2%。

Conclusion: 从感知到推理的渐进式训练对于构建鲁棒的空间智能至关重要，该方法为视觉语言模型的空间推理能力提供了有效的解决方案。

Abstract: Spatial reasoning remains a fundamental challenge for Vision-Language Models
(VLMs), with current approaches struggling to achieve robust performance
despite recent advances. We identify that this limitation stems from a critical
gap: existing methods attempt to learn spatial reasoning directly without
establishing the hierarchical foundations of perception and understanding. To
address this challenge, we present a comprehensive methodology for building
spatial intelligence progressively. We introduce SpatialLadder-26k, a
multimodal dataset containing 26,610 samples spanning object localization,
single image, multi-view, and video spatial reasoning tasks, constructed
through a standardized pipeline that ensures systematic coverage across
modalities. Building on this dataset, we design a three-stage progressive
training framework that (1) establishes spatial perception through object
localization, (2) develops spatial understanding through multi-dimensional
spatial tasks, and (3) strengthens complex reasoning via reinforcement learning
with verifiable rewards. This approach yields SpatialLadder, a 3B-parameter
model that achieves state-of-the-art performance on spatial reasoning
benchmarks, with 23.4% average improvement over the base model, surpassing
GPT-4o by 20.8% and Gemini-2.0-Flash by 10.1%. Notably, SpatialLadder maintains
strong generalization with 7.2% improvement on out-of-domain benchmarks,
demonstrating that progressive training from perception to reasoning is
essential for robust spatial intelligence.

</details>


### [8] [FlexTraj: Image-to-Video Generation with Flexible Point Trajectory Control](https://arxiv.org/abs/2510.08527v1)
*Zhiyuan Zhang,Can Wang,Dongdong Chen,Jing Liao*

Main category: cs.CV

TL;DR: FlexTraj是一个用于图像到视频生成的框架，通过统一的基于点的运动表示实现灵活的点轨迹控制，支持密集和稀疏轨迹控制，采用序列拼接方案实现快速收敛和高效推理。


<details>
  <summary>Details</summary>
Motivation: 现有的图像到视频生成方法在轨迹控制方面存在局限性，需要开发一个能够支持多粒度、对齐无关的轨迹控制的统一框架。

Method: 引入统一的基于点的运动表示，使用分割ID、时间一致的轨迹ID和可选颜色通道编码每个点；采用高效的序列拼接方案而非token拼接或ControlNet；采用退火训练策略逐步减少对完整监督和对齐条件的依赖。

Result: 实验结果表明FlexTraj能够实现多粒度、对齐无关的轨迹控制，支持运动克隆、基于拖拽的图像到视频、运动插值、相机重定向、灵活动作控制和网格动画等多种应用。

Conclusion: FlexTraj提供了一个高效、可控的图像到视频生成框架，在轨迹控制方面表现出色，具有广泛的应用前景。

Abstract: We present FlexTraj, a framework for image-to-video generation with flexible
point trajectory control. FlexTraj introduces a unified point-based motion
representation that encodes each point with a segmentation ID, a temporally
consistent trajectory ID, and an optional color channel for appearance cues,
enabling both dense and sparse trajectory control. Instead of injecting
trajectory conditions into the video generator through token concatenation or
ControlNet, FlexTraj employs an efficient sequence-concatenation scheme that
achieves faster convergence, stronger controllability, and more efficient
inference, while maintaining robustness under unaligned conditions. To train
such a unified point trajectory-controlled video generator, FlexTraj adopts an
annealing training strategy that gradually reduces reliance on complete
supervision and aligned condition. Experimental results demonstrate that
FlexTraj enables multi-granularity, alignment-agnostic trajectory control for
video generation, supporting various applications such as motion cloning,
drag-based image-to-video, motion interpolation, camera redirection, flexible
action control and mesh animations.

</details>


### [9] [Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression](https://arxiv.org/abs/2510.08512v1)
*Nikolaos Stathoulopoulos,Christoforos Kanellakis,George Nikolakopoulos*

Main category: cs.CV

TL;DR: 提出基于语义场景图的3D点云深度压缩框架，在保持结构和语义保真度的同时实现高达98%的数据压缩，支持多机器人位姿图优化和地图融合等下游应用。


<details>
  <summary>Details</summary>
Motivation: 在带宽受限和连接不稳定的多机器人系统中，大型复杂点云数据传输效率低下会降低系统性能，需要高效的压缩方法。

Method: 将点云分解为语义一致的分块，使用基于特征线性调制的语义感知编码器编码为紧凑潜在表示，通过基于折叠的解码器在潜在特征和图节点属性指导下进行重构。

Result: 在SemanticKITTI和nuScenes数据集上实现最先进的压缩率，数据大小减少高达98%，同时保持结构和语义保真度。

Conclusion: 该框架不仅实现了高效的点云压缩，还支持下游应用，在轨迹精度和地图对齐方面与原始LiDAR扫描结果相当。

Abstract: Efficient transmission of 3D point cloud data is critical for advanced
perception in centralized and decentralized multi-agent robotic systems,
especially nowadays with the growing reliance on edge and cloud-based
processing. However, the large and complex nature of point clouds creates
challenges under bandwidth constraints and intermittent connectivity, often
degrading system performance. We propose a deep compression framework based on
semantic scene graphs. The method decomposes point clouds into semantically
coherent patches and encodes them into compact latent representations with
semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A
folding-based decoder, guided by latent features and graph node attributes,
enables structurally accurate reconstruction. Experiments on the SemanticKITTI
and nuScenes datasets show that the framework achieves state-of-the-art
compression rates, reducing data size by up to 98% while preserving both
structural and semantic fidelity. In addition, it supports downstream
applications such as multi-robot pose graph optimization and map merging,
achieving trajectory accuracy and map alignment comparable to those obtained
with raw LiDAR scans.

</details>


### [10] [MoA-VR: A Mixture-of-Agents System Towards All-in-One Video Restoration](https://arxiv.org/abs/2510.08508v1)
*Lu Liu,Chunlei Cai,Shaocheng Shen,Jianfeng Liang,Weimin Ouyang,Tianxiao Ye,Jian Mao,Huiyu Duan,Jiangchao Yao,Xiaoyun Zhang,Qiang Hu,Guangtao Zhai*

Main category: cs.CV

TL;DR: MoA-VR是一个基于多智能体协作的视频修复系统，通过三个协调的智能体（退化识别、路由修复、质量评估）模仿人类专家的处理流程，有效处理复杂多样的视频退化问题。


<details>
  <summary>Details</summary>
Motivation: 现实世界视频常因采集和传输条件不同而遭受复杂退化（如噪声、压缩伪影、低光失真），现有方法需要人工选择专用模型或采用单一架构，无法泛化处理各种退化类型。

Method: 构建大规模高分辨率视频退化识别基准，使用视觉语言模型进行退化识别；引入基于大语言模型的自适应路由器学习修复策略；构建Res-VQ数据集并设计专用VLM视频质量评估模型。

Result: 大量实验表明，MoA-VR能有效处理多样化和复合退化，在客观指标和感知质量方面均优于现有基线方法。

Conclusion: 该研究展示了多模态智能和模块化推理在通用视频修复系统中的整合潜力，为处理复杂视频退化问题提供了新思路。

Abstract: Real-world videos often suffer from complex degradations, such as noise,
compression artifacts, and low-light distortions, due to diverse acquisition
and transmission conditions. Existing restoration methods typically require
professional manual selection of specialized models or rely on monolithic
architectures that fail to generalize across varying degradations. Inspired by
expert experience, we propose MoA-VR, the first
\underline{M}ixture-\underline{o}f-\underline{A}gents \underline{V}ideo
\underline{R}estoration system that mimics the reasoning and processing
procedures of human professionals through three coordinated agents: Degradation
Identification, Routing and Restoration, and Restoration Quality Assessment.
Specifically, we construct a large-scale and high-resolution video degradation
recognition benchmark and build a vision-language model (VLM) driven
degradation identifier. We further introduce a self-adaptive router powered by
large language models (LLMs), which autonomously learns effective restoration
strategies by observing tool usage patterns. To assess intermediate and final
processed video quality, we construct the \underline{Res}tored
\underline{V}ideo \underline{Q}uality (Res-VQ) dataset and design a dedicated
VLM-based video quality assessment (VQA) model tailored for restoration tasks.
Extensive experiments demonstrate that MoA-VR effectively handles diverse and
compound degradations, consistently outperforming existing baselines in terms
of both objective metrics and perceptual quality. These results highlight the
potential of integrating multimodal intelligence and modular reasoning in
general-purpose video restoration systems.

</details>


### [11] [InstructX: Towards Unified Visual Editing with MLLM Guidance](https://arxiv.org/abs/2510.08485v1)
*Chong Mou,Qichao Sun,Yanze Wu,Pengze Zhang,Xinghui Li,Fulong Ye,Songtao Zhao,Qian He*

Main category: cs.CV

TL;DR: InstructX是一个统一的多模态大语言模型框架，用于图像和视频编辑。研究发现图像训练可以产生视频编辑能力，无需显式监督，解决了视频训练数据稀缺的问题。通过整合模态特定的MLLM特征，在单一模型中统一处理图像和视频编辑任务。


<details>
  <summary>Details</summary>
Motivation: 随着多模态大语言模型在视觉理解和推理方面取得进展，研究者希望利用它们提升扩散模型的编辑性能。但目前大多数研究缺乏对MLLM设计选择的深入分析，且MLLM与扩散模型在视频编辑等困难任务中的整合仍面临挑战。

Method: 提出了InstructX统一框架，全面研究MLLM与扩散模型在指令驱动编辑任务中的整合。分析图像和视频在统一建模中的协作与区别，利用图像训练产生视频编辑能力，整合模态特定的MLLM特征。

Result: 大量实验表明，该方法能够处理广泛的图像和视频编辑任务，并实现了最先进的性能。图像训练能够产生视频编辑能力，无需显式监督，缓解了视频训练数据稀缺的限制。

Conclusion: InstructX框架成功统一了图像和视频编辑任务，通过模态特定的MLLM特征整合，在单一模型中实现了高效的跨模态编辑能力，为多模态编辑系统提供了新的解决方案。

Abstract: With recent advances in Multimodal Large Language Models (MLLMs) showing
strong visual understanding and reasoning, interest is growing in using them to
improve the editing performance of diffusion models. Despite rapid progress,
most studies lack an in-depth analysis of MLLM design choices. Moreover, the
integration of MLLMs and diffusion models remains an open challenge in some
difficult tasks, such as video editing. In this paper, we present InstructX, a
unified framework for image and video editing. Specifically, we conduct a
comprehensive study on integrating MLLMs and diffusion models for
instruction-driven editing across diverse tasks. Building on this study, we
analyze the cooperation and distinction between images and videos in unified
modeling. (1) We show that training on image data can lead to emergent video
editing capabilities without explicit supervision, thereby alleviating the
constraints imposed by scarce video training data. (2) By incorporating
modality-specific MLLM features, our approach effectively unifies image and
video editing tasks within a single model. Extensive experiments demonstrate
that our method can handle a broad range of image and video editing tasks and
achieves state-of-the-art performance.

</details>


### [12] [The Visual Iconicity Challenge: Evaluating Vision-Language Models on Sign Language Form-Meaning Mapping](https://arxiv.org/abs/2510.08482v1)
*Onur Keleş,Aslı Özyürek,Gerardo Ortega,Kadir Gökgö,Esam Ghaleb*

Main category: cs.CV

TL;DR: 该论文提出了视觉象似性挑战基准，用于评估视觉语言模型在手语象似性任务上的表现，包括音系形式预测、透明度和象似性评级三个任务。


<details>
  <summary>Details</summary>
Motivation: 手语中的象似性（语言形式与意义的相似性）为视觉基础化提供了自然测试平台，研究旨在评估VLMs从动态人体动作中恢复这种映射的能力。

Method: 引入基于视频的视觉象似性挑战基准，采用心理语言学测量方法，在零样本和少样本设置下评估13个最先进的VLMs在荷兰手语上的表现，并与人类基线比较。

Result: VLMs在音系形式预测上能恢复部分手形和位置细节但低于人类表现；在透明度任务上远低于人类基线；只有顶级模型与人类象似性评级有中等相关性。有趣的是，音系形式预测能力更强的模型与人类象似性判断相关性更好。

Conclusion: 研究验证了这些诊断任务的有效性，并表明需要以人为本的信号和具身学习方法，以建模象似性并改进多模态模型中的视觉基础化。

Abstract: Iconicity, the resemblance between linguistic form and meaning, is pervasive
in signed languages, offering a natural testbed for visual grounding. For
vision-language models (VLMs), the challenge is to recover such essential
mappings from dynamic human motion rather than static context. We introduce the
\textit{Visual Iconicity Challenge}, a novel video-based benchmark that adapts
psycholinguistic measures to evaluate VLMs on three tasks: (i) phonological
sign-form prediction (e.g., handshape, location), (ii) transparency (inferring
meaning from visual form), and (iii) graded iconicity ratings. We assess $13$
state-of-the-art VLMs in zero- and few-shot settings on Sign Language of the
Netherlands and compare them to human baselines. On \textit{phonological form
prediction}, VLMs recover some handshape and location detail but remain below
human performance; on \textit{transparency}, they are far from human baselines;
and only top models correlate moderately with human \textit{iconicity ratings}.
Interestingly, \textit{models with stronger phonological form prediction
correlate better with human iconicity judgment}, indicating shared sensitivity
to visually grounded structure. Our findings validate these diagnostic tasks
and motivate human-centric signals and embodied learning methods for modelling
iconicity and improving visual grounding in multimodal models.

</details>


### [13] [Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools](https://arxiv.org/abs/2510.08480v1)
*Zhenlong Yuan,Xiangyan Qu,Chengxuan Qian,Rui Chen,Jing Tang,Lei Sun,Xiangxiang Chu,Dapeng Zhang,Yiwei Wang,Yujun Cai,Shuo Li*

Main category: cs.CV

TL;DR: Video-STAR是一个用于开放词汇动作识别的框架，通过上下文子动作分解和工具增强强化学习，解决多模态大语言模型在语义相似动作区分上的局限性。


<details>
  <summary>Details</summary>
Motivation: 多模态大语言模型依赖文本先验，在开放词汇场景中难以区分语义相似的动作，需要更细粒度的视觉推理能力。

Method: 将动作分解为判别性子动作进行细粒度匹配，同时动态调用领域特定工具进行跨模态交错推理，通过分层奖励机制平衡工具使用效率、子动作相关性和结构一致性。

Result: 在HMDB-51、UCF-101、SSv2、Kinetics-400和Kinetics-600数据集上达到最先进性能，在区分细粒度动作和处理跨模态幻觉方面优于现有方法。

Conclusion: Video-STAR通过子动作分解和工具增强推理，显著提升了开放词汇动作识别的性能，验证了其优秀的鲁棒性和泛化能力。

Abstract: Multimodal large language models (MLLMs) have demonstrated remarkable
potential in bridging visual and textual reasoning, yet their reliance on
text-centric priors often limits their ability to disentangle semantically
similar actions in open-vocabulary scenarios. To address this, we propose
Video-STAR, a framework that harmonizes contextual sub-motion decomposition
with tool-augmented reinforcement learning for open-vocabulary action
recognition (OVAR). Unlike prior methods that treat actions as monolithic
entities, our approach innovatively decomposes actions into discriminative
sub-motions for fine-grained matching while dynamically invoking
domain-specific tools for cross-modal interleaving, thereby enabling
category-specific reasoning capacity and reducing cross-modal hallucination.
Moreover, by designing a hierarchical reward that balances tool-usage
efficiency, sub-motion relevance, and structural coherence in reasoning, our
method autonomously leverages external tools to prioritize sub-motion patterns
without explicit supervision, transmitting from text-centric reasoning to
visually grounded inference. Extensive evaluations on HMDB-51, UCF-101, SSv2,
Kinetics-400, and Kinetics-600 datasets demonstrate our state-of-the-art
performance, outperforming existing methods in distinguishing fine-grained
actions and handling cross-modal hallucination, validating our excellent
robustness and generalization.

</details>


### [14] [Hierarchical Spatial Algorithms for High-Resolution Image Quantization and Feature Extraction](https://arxiv.org/abs/2510.08449v1)
*Noor Islam S. Mohammad*

Main category: cs.CV

TL;DR: 本文提出了一个模块化空间图像处理框架，集成了灰度量化、色彩亮度增强、图像锐化、双向变换管道和几何特征提取等功能，在多个数据集上表现出稳健的确定性性能。


<details>
  <summary>Details</summary>
Motivation: 开发一个综合性的图像处理框架，能够处理从基础量化到高级特征提取的多种任务，满足实时图像分析和计算机视觉应用的需求。

Method: 采用模块化设计，包括：灰度图像的8级离散量化、RGB和YCrCb色彩空间的直方图均衡化、HSV值通道的亮度调整、3×3卷积核的图像锐化、结合反锐化掩模、伽马校正和噪声放大的双向变换管道，以及Canny边缘检测、霍夫线估计、Harris角点检测和形态学窗口定位等几何特征提取方法。

Result: 双向变换管道在正向和反向过程中分别达到76.10%和74.80%的准确率；霍夫线估计成功检测到51.50°的台球杆对齐角度；球杆隔离与真实图像的相似度达到81.87%。

Conclusion: 该模块化框架在多样化数据集上表现出稳健的确定性性能，展示了在实时图像分析和计算机视觉应用中的潜力。

Abstract: This study introduces a modular framework for spatial image processing,
integrating grayscale quantization, color and brightness enhancement, image
sharpening, bidirectional transformation pipelines, and geometric feature
extraction. A stepwise intensity transformation quantizes grayscale images into
eight discrete levels, producing a posterization effect that simplifies
representation while preserving structural detail. Color enhancement is
achieved via histogram equalization in both RGB and YCrCb color spaces, with
the latter improving contrast while maintaining chrominance fidelity.
Brightness adjustment is implemented through HSV value-channel manipulation,
and image sharpening is performed using a 3 * 3 convolution kernel to enhance
high-frequency details. A bidirectional transformation pipeline that integrates
unsharp masking, gamma correction, and noise amplification achieved accuracy
levels of 76.10% and 74.80% for the forward and reverse processes,
respectively. Geometric feature extraction employed Canny edge detection,
Hough-based line estimation (e.g., 51.50{\deg} for billiard cue alignment),
Harris corner detection, and morphological window localization. Cue isolation
further yielded 81.87\% similarity against ground truth images. Experimental
evaluation across diverse datasets demonstrates robust and deterministic
performance, highlighting its potential for real-time image analysis and
computer vision.

</details>


### [15] [UniVideo: Unified Understanding, Generation, and Editing for Videos](https://arxiv.org/abs/2510.08377v1)
*Cong Wei,Quande Liu,Zixuan Ye,Qiulin Wang,Xintao Wang,Pengfei Wan,Kun Gai,Wenhu Chen*

Main category: cs.CV

TL;DR: UniVideo是一个统一的多模态视频生成和编辑框架，采用双流设计结合MLLM和MMDiT，支持多种视频任务并在单一指令范式下联合训练，实现了任务组合和跨任务泛化能力。


<details>
  <summary>Details</summary>
Motivation: 现有的统一多模态模型主要局限于图像领域，需要扩展到视频领域来处理复杂的多模态指令并保持视觉一致性。

Method: 采用双流设计：多模态大语言模型(MLLM)用于理解指令，多模态DiT(MMDiT)用于视频生成，在单一多模态指令范式下联合训练多种视频任务。

Result: 在文本/图像到视频生成、上下文视频生成和编辑任务中达到或超过最先进的特定任务基线，支持任务组合和跨任务泛化。

Conclusion: UniVideo展示了统一视频建模的可行性，支持任务组合和跨模态能力迁移，为未来视频生成研究提供了新方向。

Abstract: Unified multimodal models have shown promising results in multimodal content
generation and editing but remain largely limited to the image domain. In this
work, we present UniVideo, a versatile framework that extends unified modeling
to the video domain. UniVideo adopts a dual-stream design, combining a
Multimodal Large Language Model (MLLM) for instruction understanding with a
Multimodal DiT (MMDiT) for video generation. This design enables accurate
interpretation of complex multimodal instructions while preserving visual
consistency. Built on this architecture, UniVideo unifies diverse video
generation and editing tasks under a single multimodal instruction paradigm and
is jointly trained across them. Extensive experiments demonstrate that UniVideo
matches or surpasses state-of-the-art task-specific baselines in
text/image-to-video generation, in-context video generation and in-context
video editing. Notably, the unified design of UniVideo enables two forms of
generalization. First, UniVideo supports task composition, such as combining
editing with style transfer, by integrating multiple capabilities within a
single instruction. Second, even without explicit training on free-form video
editing, UniVideo transfers its editing capability from large-scale image
editing data to this setting, handling unseen instructions such as
green-screening characters or changing materials within a video. Beyond these
core capabilities, UniVideo also supports visual-prompt-based video generation,
where the MLLM interprets visual prompts and guides the MMDiT during synthesis.
To foster future research, we will release our model and code.

</details>


### [16] [Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge](https://arxiv.org/abs/2510.08316v1)
*Yu Huang,Zelin Peng,Changsong Wen,Xiaokang Yang,Wei Shen*

Main category: cs.CV

TL;DR: 提出了一种基于语义基础学习范式的3D功能分割方法，通过跨模态亲和力转移将2D视觉基础模型的丰富语义知识迁移到3D领域，显著提升了3D功能分割的精度。


<details>
  <summary>Details</summary>
Motivation: 现有的3D功能分割方法通常使用点云编码器作为通用特征提取器，但忽视了3D数据固有的挑战（如稀疏性、噪声和几何模糊性），导致学习到的3D特征缺乏清晰且语义一致的功能边界。

Method: 提出了跨模态亲和力转移预训练策略，将3D编码器与提升的2D语义对齐，并联合优化重建、亲和力和多样性；在此基础上设计了跨模态功能分割变换器，集成多模态提示与预训练特征来生成精确的提示感知分割图。

Result: 在标准基准测试上的广泛实验表明，该框架在3D功能分割任务上取得了新的最先进结果。

Conclusion: 通过将2D视觉基础模型的语义知识迁移到3D领域，能够有效解决3D功能分割中的语义边界不清晰问题，为机器人操作、具身AI等应用提供了更可靠的功能分割能力。

Abstract: Affordance segmentation aims to parse 3D objects into functionally distinct
parts, bridging recognition and interaction for applications in robotic
manipulation, embodied AI, and AR. While recent studies leverage visual or
textual prompts to guide this process, they often rely on point cloud encoders
as generic feature extractors, overlooking the intrinsic challenges of 3D data
such as sparsity, noise, and geometric ambiguity. As a result, 3D features
learned in isolation frequently lack clear and semantically consistent
functional boundaries. To address this bottleneck, we propose a
semantic-grounded learning paradigm that transfers rich semantic knowledge from
large-scale 2D Vision Foundation Models (VFMs) into the 3D domain.
Specifically, We introduce Cross-Modal Affinity Transfer (CMAT), a pre-training
strategy that aligns a 3D encoder with lifted 2D semantics and jointly
optimizes reconstruction, affinity, and diversity to yield semantically
organized representations. Building on this backbone, we further design the
Cross-modal Affordance Segmentation Transformer (CAST), which integrates
multi-modal prompts with CMAT-pretrained features to generate precise,
prompt-aware segmentation maps. Extensive experiments on standard benchmarks
demonstrate that our framework establishes new state-of-the-art results for 3D
affordance segmentation.

</details>


### [17] [A Multimodal Depth-Aware Method For Embodied Reference Understanding](https://arxiv.org/abs/2510.08278v1)
*Fevziye Irem Eyiokur,Dogucan Yaman,Hazım Kemal Ekenel,Alexander Waibel*

Main category: cs.CV

TL;DR: 提出了一种新颖的具身参考理解框架，通过结合LLM数据增强、深度图模态和深度感知决策模块，在模糊场景中更准确地识别目标物体。


<details>
  <summary>Details</summary>
Motivation: 解决现有开放词汇物体检测方法在存在多个候选物体的模糊场景中经常失败的问题，需要同时处理语言指令和指向线索。

Method: 联合利用LLM数据增强、深度图模态和深度感知决策模块，实现语言和具身线索的鲁棒集成。

Result: 在两个数据集上的实验结果表明，该方法显著优于现有基线，实现了更准确可靠的参考检测。

Conclusion: 所提出的ERU框架能够有效处理复杂或杂乱环境中的歧义问题，提高具身参考理解的性能。

Abstract: Embodied Reference Understanding requires identifying a target object in a
visual scene based on both language instructions and pointing cues. While prior
works have shown progress in open-vocabulary object detection, they often fail
in ambiguous scenarios where multiple candidate objects exist in the scene. To
address these challenges, we propose a novel ERU framework that jointly
leverages LLM-based data augmentation, depth-map modality, and a depth-aware
decision module. This design enables robust integration of linguistic and
embodied cues, improving disambiguation in complex or cluttered environments.
Experimental results on two datasets demonstrate that our approach
significantly outperforms existing baselines, achieving more accurate and
reliable referent detection.

</details>


### [18] [Beyond Textual CoT: Interleaved Text-Image Chains with Deep Confidence Reasoning for Image Editing](https://arxiv.org/abs/2510.08157v1)
*Zhentao Zou,Zhengrong Yue,Kunpeng Du,Binlei Bao,Hanting Li,Haizhen Xie,Guozheng Xu,Yue Zhou,Yali Wang,Jie Hu,Xue Jiang,Xinghao Chen*

Main category: cs.CV

TL;DR: 提出MURE框架，通过多模态交错文本-图像推理链解决复杂图像编辑问题，引入MMDC机制减少幻觉，在三个基准测试中显著提升编辑效果


<details>
  <summary>Details</summary>
Motivation: 现有方法在处理复杂对象交叉和细粒度空间关系时缺乏显式推理过程，纯文本或坐标增强的CoT无法充分表示复杂视觉布局和像素级细节

Method: MURE框架采用原生多模态交错文本-图像推理链，将复杂编辑任务分解为相互依赖的子任务，每个步骤包含文本描述和对应视觉提示（如位置掩码），并引入MMDC机制通过奖励模型剪枝低质量推理路径

Result: 构建了首个CoT-Edit-14K数据集，包含14K高质量编辑样本，在三个图像编辑基准测试中取得显著改进

Conclusion: 多模态交错推理链能够更精确地分解复杂编辑任务，通过视觉推理路径的深度置信度评估确保高质量编辑结果，为图像编辑提供了更有效的推理框架

Abstract: Image editing with natural language has gained significant popularity, yet
existing methods struggle with intricate object intersections and fine-grained
spatial relationships due to the lack of an explicit reasoning process. While
Chain-of-Thought (CoT) has been explored to enhance reasoning, purely textual
CoT or CoT augmented with coordinate information is fundamentally limited in
its ability to represent intricate visual layouts and lacks the necessary
visual cues to guide the generation of fine-grained, pixel-level details. To
address these challenges, we propose Multimodal Reasoning Edit (MURE), a novel
framework that shifts the visual editing process from purely text-based
reasoning to a series of interleaved textual and visual rationales. Our
framework performs image editing using a natively multimodal, interleaved
text-image CoT. This approach generates a step-by-step chain of reasoning where
a textual description is followed by a corresponding visual cue, such as a
positional mask that defined intended edited regions or a representation of new
content. Furthermore, to mitigate the hallucination phenomenon of large
language models, we introduce Multimodal Deep Confidence (MMDC) reasoning
paradigm. This paradigm explores a tree of visual reasoning paths at each step.
By pruning low-quality branches using a deep confidence score from a reward
model, it ensures the model consistently follows a high-quality trajectory
towards the final edited result. The proposed method decomposes complex editing
tasks into interdependent sub-tasks, achieving greater precision at each stage
and yielding high-fidelity edited results. We define the formulation for
interleaved text-image chains and release the first CoT-Edit-14K dataset,
comprising 14K high-quality editing examples. Extensive experiments show that
our method yields significant improvements across three image editing
benchmarks.

</details>


<div id='cs.CL'></div>

# cs.CL [[Back]](#toc)

### [19] [Which Heads Matter for Reasoning? RL-Guided KV Cache Compression](https://arxiv.org/abs/2510.08525v1)
*Wenjie Du,Li Jiang,Keda Tao,Xue Liu,Huan Wang*

Main category: cs.CL

TL;DR: RLKV是一个新颖的KV缓存压缩框架，使用强化学习识别推理模型中对推理质量至关重要的注意力头，只对这些关键头保留完整KV缓存，对其他头应用压缩缓存，在实现20-50%缓存减少的同时保持接近无损的性能。


<details>
  <summary>Details</summary>
Motivation: 现有的KV缓存压缩方法在推理模型上表现不佳：token丢弃方法破坏了推理完整性，而头重分配方法错误地压缩了推理关键头，导致性能显著下降。

Method: 提出RLKV框架，使用强化学习直接优化每个头的缓存使用与推理质量之间的关系，识别推理关键头，然后对这些关键头分配完整KV缓存，对其他头应用压缩的常量KV缓存。

Result: 实验表明只有一小部分注意力头对推理至关重要，RLKV在实现20-50%缓存减少的同时，性能优于基线方法，接近未压缩结果。

Conclusion: RLKV成功识别了推理模型中的功能异质性，通过选择性压缩实现了高效的KV缓存管理，为推理模型的高效部署提供了有效解决方案。

Abstract: Reasoning large language models exhibit complex reasoning behaviors through
the extended chain-of-thought generation, creating unprecedented Key-Value (KV)
cache overhead during the decoding phase. Existing KV cache compression methods
underperform on reasoning models: token-dropping methods break reasoning
integrity by discarding critical information, while head-reallocating methods
mistakenly compress reasoning-critical heads since they are designed for
retrieval tasks, resulting in significant performance degradation as
compression rates increase. We hypothesize that KV heads exhibit functional
heterogeneity in reasoning models-some heads are critical for chain-of-thought
consistency while others are compressible. To validate and exploit this
insight, we propose RLKV, a novel reasoning-critical head identification
framework, which uses reinforcement learning to directly optimize the
relationship between each head's cache usage and reasoning quality. As RLKV
produces rewards from actual generated samples during training, it naturally
identifies heads relevant to reasoning behaviors. We then allocate full KV
cache to these heads while applying compressed constant KV cache to others for
efficient inference. Our experiments reveal that only a small fraction of
attention heads is essential for reasoning, enabling our KV compression
approach to outperform baseline methods while achieving 20-50% cache reduction
with near lossless performance compared to uncompressed results.

</details>


### [20] [ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping](https://arxiv.org/abs/2510.08457v1)
*Shuang Chen,Yue Guo,Yimeng Ye,Shijue Huang,Wenbo Hu,Haoxi Li,Manyuan Zhang,Jiayu Chen,Song Guo,Nanyun Peng*

Main category: cs.CL

TL;DR: ARES是一个统一的开源自适应推理框架，通过动态分配探索努力来解决多模态大推理模型在简单问题上过度思考、在复杂问题上探索不足的问题。


<details>
  <summary>Details</summary>
Motivation: 现有MLRMs在简单问题上产生不必要的冗长推理轨迹，而在挑战性问题上探索不足导致错过解决方案，需要平衡推理效率。

Method: 采用两阶段训练流程：自适应冷启动阶段根据问题难度成比例生成推理轨迹；第二阶段使用自适应熵策略优化，利用高窗口熵token作为探索触发器，结合分层熵奖励和动态KL控制决定探索程度。

Result: 在多种数学、逻辑和多模态基准测试中，ARES实现了优越的性能和推理效率，在显著降低推理成本的同时缩小了与领先商业系统的差距。

Conclusion: ARES框架通过自适应推理机制有效平衡了简单和复杂问题的推理资源分配，提高了多模态大推理模型的整体效率和性能。

Abstract: Recent advances in multimodal large reasoning models (MLRMs) have
substantially improved their ability to solve complex textual and visual tasks.
However, these models tend to overthink on simple problems, producing
unnecessarily lengthy reasoning traces, while under-exploring on challenging
ones, leading to missed solutions. To address this imbalance, we propose ARES,
a unified open-source framework for adaptive reasoning that dynamically
allocates exploration effort based on task difficulty. Our approach is
motivated by two key empirical findings: (i) while single-token entropy is
noisy, high window-entropy (HWE) tokens (token-level entropies averaged under a
sliding window) can reliably capture reasoning-critical moments; and (ii)
reducing HWE usage benefits easy problems, while increasing it is essential for
solving hard ones. Building on these insights, ARES introduces a two-stage
training pipeline. In the Adaptive Cold-Start stage, we curate multimodal and
textual data paired with reasoning traces of length proportional to problem
difficulty, equipping the model with initial difficulty awareness. In the
second stage, we develop Adaptive Entropy Policy Optimization (AEPO), which
uses HWE tokens as exploration triggers to decide when to explore, and a
hierarchical entropy reward with dynamic KL control to decide how much to
explore. Extensive experiments demonstrate that ARES achieves superior
performance and reasoning efficiency across diverse mathematical, logical, and
multimodal benchmarks, while closing the gap to leading commercial systems
under significantly lower inference costs.

</details>


### [21] [ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code](https://arxiv.org/abs/2510.08163v1)
*Jian Xie,Zhendong Chu,Aoxiao Zhong,Kai Zhang,Mingzhe Han,Xin Fang,Jialie Shen,Qingsong Wen*

Main category: cs.CL

TL;DR: ARM2是一个通过强化学习框架和长度感知优化来平衡推理性能与效率的统一模型，可减少70%以上的token使用量，同时保持与传统推理模型相当的性能。


<details>
  <summary>Details</summary>
Motivation: 大型推理模型存在"过度思考"问题，在简单任务上生成不必要的冗长推理。现有方法如长度惩罚或路由机制通常是启发式且任务特定的，缺乏自适应推理的通用框架。

Method: ARM2采用强化学习框架，结合长度感知优化，支持多种推理格式。除了自然语言推理外，还集成视觉理解和可执行代码，实现多模态应用。

Result: 实验表明ARM2在保持与传统GRPO训练推理模型相当性能的同时，平均减少70%以上的token使用量。

Conclusion: ARM2通过自适应推理框架有效解决了过度思考问题，在保持性能的同时显著提高推理效率，并通过广泛分析验证了其设计合理性。

Abstract: Large Reasoning Models (LRMs) often suffer from the ``over-thinking''
problem, generating unnecessarily long reasoning on simple tasks. Some
strategies have been proposed to mitigate this issue, such as length penalties
or routing mechanisms, but they are typically heuristic and task-specific,
lacking a general framework for adaptive reasoning. In this paper, we present
ARM2, a unified model that adaptively balances reasoning performance and
efficiency across multiple formats through a reinforcement learning framework
augmented with length-aware optimization. Beyond conventional natural language
inference, ARM2 integrates vision understanding, extending its applicability to
multimodal. Moreover, ARM2 integrates executable code into reasoning, enabling
substantial reductions in token cost while preserving task performance compared
to long CoT. Experiments demonstrate that ARM2 achieves performance on par with
traditional reasoning models trained with GRPO, while reducing token usage by
over 70% on average. We further conduct extensive analyses to validate the
effectiveness of ARM2 and the soundness of its design.

</details>


<div id='cs.HC'></div>

# cs.HC [[Back]](#toc)

### [22] [Sentiment Matters: An Analysis of 200 Human-SAV Interactions](https://arxiv.org/abs/2510.08202v1)
*Lirui Guo,Michael G. Burke,Wynita M. Griggs*

Main category: cs.HC

TL;DR: 本文介绍了包含200个人类与共享自动驾驶车辆交互的数据集，包括2136次对话交换和心理学调查数据，通过两个案例研究展示了数据集的实用性：识别SAV接受度的关键预测因子，以及比较LLM与传统情感分析方法的性能。


<details>
  <summary>Details</summary>
Motivation: 共享自动驾驶车辆（SAVs）可能成为交通系统的重要组成部分，因此有效的人类-SAV交互成为一个重要研究领域。本文旨在通过创建人类-SAV对话数据集来推动这一领域的发展。

Method: 创建了包含文本数据（2136次人类-SAV对话交换）和实证数据（心理因素调查结果）的开源数据集。使用随机森林建模和弦图识别关键预测因子，并比较基于LLM的情感分析工具与传统TextBlob方法的性能。

Result: 研究发现响应情感极性（感知积极性）是SAV接受度和服务质量感知的关键影响因素。即使简单的零样本LLM提示也比传统词典方法更接近用户报告的情感，但仍存在局限性。

Conclusion: 本研究为设计对话式SAV界面提供了新的见解，并为进一步探索高级情感建模、自适应用户交互和多模态对话系统奠定了基础。

Abstract: Shared Autonomous Vehicles (SAVs) are likely to become an important part of
the transportation system, making effective human-SAV interactions an important
area of research. This paper introduces a dataset of 200 human-SAV interactions
to further this area of study. We present an open-source human-SAV
conversational dataset, comprising both textual data (e.g., 2,136 human-SAV
exchanges) and empirical data (e.g., post-interaction survey results on a range
of psychological factors). The dataset's utility is demonstrated through two
benchmark case studies: First, using random forest modeling and chord diagrams,
we identify key predictors of SAV acceptance and perceived service quality,
highlighting the critical influence of response sentiment polarity (i.e.,
perceived positivity). Second, we benchmark the performance of an LLM-based
sentiment analysis tool against the traditional lexicon-based TextBlob method.
Results indicate that even simple zero-shot LLM prompts more closely align with
user-reported sentiment, though limitations remain. This study provides novel
insights for designing conversational SAV interfaces and establishes a
foundation for further exploration into advanced sentiment modeling, adaptive
user interactions, and multimodal conversational systems.

</details>


<div id='stat.ME'></div>

# stat.ME [[Back]](#toc)

### [23] [Doubly Robust Estimation with Stabilized Weights for Binary Proximal Outcomes in Micro-Randomized Trials](https://arxiv.org/abs/2510.08359v1)
*Jinho Cha,Eunchan Cha*

Main category: stat.ME

TL;DR: 提出了一种双重稳健的估计平均偏移效应（DR-EMEE）方法，结合了每决策逆概率加权和结果回归，使用稳定化和截断权重，在微随机化试验中提高了估计效率和稳健性。


<details>
  <summary>Details</summary>
Motivation: 微随机化试验中，标准的逆概率加权估计器在小样本或极端随机化情况下存在偏差和不稳定性，而估计平均偏移效应虽然提高了效率但缺乏双重稳健性。

Method: 提出DR-EMEE方法，结合每决策逆概率加权和结果回归，使用稳定化和截断权重，证明了双重稳健性和渐近效率，并提供有限样本方差校正，扩展到机器学习干扰估计器。

Result: 在模拟中，DR-EMEE降低了均方根误差，提高了覆盖率，相比IPW实现了高达两倍的效率提升，相比EMEE提升了5-10%的效率。在HeartSteps、PAMAP2和mHealth数据集应用中，在随机化和观察性设置下均实现了稳定高效的推断。

Conclusion: DR-EMEE方法在微随机化试验中提供了更稳健和高效的因果效应估计，适用于移动健康干预评估。

Abstract: Micro-randomized trials (MRTs) are increasingly used to evaluate mobile
health interventions with binary proximal outcomes. Standard inverse
probability weighting (IPW) estimators are unbiased but unstable in small
samples or under extreme randomization. Estimated mean excursion effect (EMEE)
improves efficiency but lacks double robustness. We propose a doubly robust
EMEE (DR-EMEE) with stabilized and truncated weights, combining per-decision
IPW and outcome regression. We prove double robustness, asymptotic efficiency,
and provide finite-sample variance corrections, with extensions to machine
learning nuisance estimators. In simulations, DR-EMEE reduces root mean squared
error, improves coverage, and achieves up to twofold efficiency gains over IPW
and five to ten percent over EMEE. Applications to HeartSteps, PAMAP2, and
mHealth datasets confirm stable and efficient inference across both randomized
and observational settings.

</details>


### [24] [Bayesian Profile Regression with Linear Mixed Models (Profile-LMM) applied to Longitudinal Exposome Data](https://arxiv.org/abs/2510.08304v1)
*Matteo Amestoy,Mark van de Wiel,Jeroen Lakerveld,Wessel van Wieringen*

Main category: stat.ME

TL;DR: 提出了一种新的统计框架，将贝叶斯轮廓回归扩展到纵向数据分析中，通过结合轮廓回归和线性混合模型来处理暴露组数据中的多重共线性、纵向测量和复杂交互作用问题。


<details>
  <summary>Details</summary>
Motivation: 暴露组（多种非遗传因素）对健康结果有重要影响，但分析暴露组面临方法学挑战：暴露间高度共线性、纵向重复测量特性、与个体特征的复杂交互作用。

Method: 提出轮廓-线性混合模型（profile-LMM），将轮廓回归（通过聚类处理共线性）整合到线性混合模型（处理纵向数据）中，同时考虑潜在暴露簇与个体特征的交互作用。

Result: 通过模拟数据验证，该方法能准确识别模型参数并恢复真实的潜在暴露簇结构；在Lifelines队列的纵向数据应用中，成功识别出与舒张压显著相关的暴露组合。

Conclusion: 所提出的profile-LMM框架有效解决了暴露组分析中的关键方法学挑战，为纵向暴露组研究提供了实用的统计工具。

Abstract: Exposure to diverse non-genetic factors, known as the exposome, is a critical
determinant of health outcomes. However, analyzing the exposome presents
significant methodological challenges, including: high collinearity among
exposures, the longitudinal nature of repeated measurements, and potential
complex interactions with individual characteristics. In this paper, we address
these challenges by proposing a novel statistical framework that extends
Bayesian profile regression. Our method integrates profile regression, which
handles collinearity by clustering exposures into latent profiles, into a
linear mixed model (LMM), a framework for longitudinal data analysis. This
profile-LMM approach effectively accounts for within-person variability over
time while also incorporating interactions between the latent exposure clusters
and individual characteristics. We validate our method using simulated data,
demonstrating its ability to accurately identify model parameters and recover
the true latent exposure cluster structure. Finally, we apply this approach to
a large longitudinal data set from the Lifelines cohort to identify
combinations of exposures that are significantly associated with diastolic
blood pressure.

</details>


<div id='cs.GR'></div>

# cs.GR [[Back]](#toc)

### [25] [X2Video: Adapting Diffusion Models for Multimodal Controllable Neural Video Rendering](https://arxiv.org/abs/2510.08530v1)
*Zhitong Huang,Mohan Zhang,Renhan Wang,Rui Tang,Hao Zhu,Jing Liao*

Main category: cs.GR

TL;DR: X2Video是首个基于内在通道（反照率、法线、粗糙度、金属度、辐照度）生成逼真视频的扩散模型，支持参考图像和文本提示的多模态控制，通过混合自注意力确保时间一致性，递归采样方法生成长视频。


<details>
  <summary>Details</summary>
Motivation: 现有视频生成模型缺乏对颜色、材质、几何和光照的精确控制能力，需要开发能够通过内在通道和多模态输入实现精确视频渲染的方法。

Method: 扩展XRGB图像生成模型到视频生成，采用混合自注意力确保时间一致性，开发掩码交叉注意力分离全局和局部文本提示，使用递归采样方法结合关键帧预测和帧插值生成长视频。

Result: X2Video能够生成时间一致、逼真的长视频，有效支持多模态控制，包括参考图像、全局和局部文本提示，并支持通过参数调整编辑颜色、材质、几何和光照。

Conclusion: X2Video通过内在通道引导和多模态控制，实现了对视频内容的精确操控，在视频生成质量、时间一致性和编辑灵活性方面表现出色。

Abstract: We present X2Video, the first diffusion model for rendering photorealistic
videos guided by intrinsic channels including albedo, normal, roughness,
metallicity, and irradiance, while supporting intuitive multi-modal controls
with reference images and text prompts for both global and local regions. The
intrinsic guidance allows accurate manipulation of color, material, geometry,
and lighting, while reference images and text prompts provide intuitive
adjustments in the absence of intrinsic information. To enable these
functionalities, we extend the intrinsic-guided image generation model XRGB to
video generation by employing a novel and efficient Hybrid Self-Attention,
which ensures temporal consistency across video frames and also enhances
fidelity to reference images. We further develop a Masked Cross-Attention to
disentangle global and local text prompts, applying them effectively onto
respective local and global regions. For generating long videos, our novel
Recursive Sampling method incorporates progressive frame sampling, combining
keyframe prediction and frame interpolation to maintain long-range temporal
consistency while preventing error accumulation. To support the training of
X2Video, we assembled a video dataset named InteriorVideo, featuring 1,154
rooms from 295 interior scenes, complete with reliable ground-truth intrinsic
channel sequences and smooth camera trajectories. Both qualitative and
quantitative evaluations demonstrate that X2Video can produce long, temporally
consistent, and photorealistic videos guided by intrinsic conditions.
Additionally, X2Video effectively accommodates multi-modal controls with
reference images, global and local text prompts, and simultaneously supports
editing on color, material, geometry, and lighting through parametric tuning.
Project page: https://luckyhzt.github.io/x2video

</details>


<div id='cs.LG'></div>

# cs.LG [[Back]](#toc)

### [26] [Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models](https://arxiv.org/abs/2510.08492v1)
*Sharut Gupta,Shobhita Sundaram,Chenyu Wang,Stefanie Jegelka,Phillip Isola*

Main category: cs.LG

TL;DR: UML是一种模态无关的训练范式，通过交替处理不同模态的输入并共享参数，利用未配对的辅助多模态数据来增强目标模态的表征学习，无需显式配对数据。


<details>
  <summary>Details</summary>
Motivation: 传统多模态学习器依赖配对数据集，但忽略了利用未配对的辅助多模态数据直接增强目标模态表征学习的潜力。

Method: 提出UML（Unpaired Multimodal Learner），采用模态无关的训练范式，单一模型交替处理不同模态输入并共享参数，利用不同模态是共享底层现实投影的假设。

Result: 理论上在线性数据生成假设下，未配对辅助数据能产生比单模态训练更具信息量的表征；实证表明使用文本、音频或图像等未配对辅助模态数据能持续提升图像和音频等单模态目标的下游性能。

Conclusion: UML证明了利用未配对多模态数据增强单模态表征学习的有效性，为多模态学习提供了新的训练范式。

Abstract: Traditional multimodal learners find unified representations for tasks like
visual question answering, but rely heavily on paired datasets. However, an
overlooked yet potentially powerful question is: can one leverage auxiliary
unpaired multimodal data to directly enhance representation learning in a
target modality? We introduce UML: Unpaired Multimodal Learner, a
modality-agnostic training paradigm in which a single model alternately
processes inputs from different modalities while sharing parameters across
them. This design exploits the assumption that different modalities are
projections of a shared underlying reality, allowing the model to benefit from
cross-modal structure without requiring explicit pairs. Theoretically, under
linear data-generating assumptions, we show that unpaired auxiliary data can
yield representations strictly more informative about the data-generating
process than unimodal training. Empirically, we show that using unpaired data
from auxiliary modalities -- such as text, audio, or images -- consistently
improves downstream performance across diverse unimodal targets such as image
and audio. Our project page: https://unpaired-multimodal.github.io/

</details>


### [27] [Expressive Value Learning for Scalable Offline Reinforcement Learning](https://arxiv.org/abs/2510.08218v1)
*Nicolas Espinosa-Dice,Kiante Brantley,Wen Sun*

Main category: cs.LG

TL;DR: EVOR是一种可扩展的离线强化学习方法，通过结合表达性策略和表达性价值函数，避免了传统方法中的蒸馏或反向传播时间问题，在训练时通过流匹配学习最优正则化Q函数，在推理时通过拒绝采样进行策略提取。


<details>
  <summary>Details</summary>
Motivation: 解决离线强化学习在机器人领域应用中的可扩展性问题，避免传统方法依赖计算昂贵的反向传播时间或引入复合错误的策略蒸馏。

Method: 提出EVOR方法，在训练时使用流匹配学习正则化Q函数，在推理时通过拒绝采样从表达性价值函数中提取策略，实现高效优化和计算可扩展搜索。

Result: 在多样化的离线RL任务上，EVOR超越了基线方法，证明了将表达性价值学习整合到离线RL中的优势。

Conclusion: EVOR提供了一种无需蒸馏或反向传播时间的可扩展离线RL解决方案，通过表达性价值函数和策略的有效整合，提升了离线强化学习的性能。

Abstract: Reinforcement learning (RL) is a powerful paradigm for learning to make
sequences of decisions. However, RL has yet to be fully leveraged in robotics,
principally due to its lack of scalability. Offline RL offers a promising
avenue by training agents on large, diverse datasets, avoiding the costly
real-world interactions of online RL. Scaling offline RL to increasingly
complex datasets requires expressive generative models such as diffusion and
flow matching. However, existing methods typically depend on either
backpropagation through time (BPTT), which is computationally prohibitive, or
policy distillation, which introduces compounding errors and limits scalability
to larger base policies. In this paper, we consider the question of how to
develop a scalable offline RL approach without relying on distillation or
backpropagation through time. We introduce Expressive Value Learning for
Offline Reinforcement Learning (EVOR): a scalable offline RL approach that
integrates both expressive policies and expressive value functions. EVOR learns
an optimal, regularized Q-function via flow matching during training. At
inference-time, EVOR performs inference-time policy extraction via rejection
sampling against the expressive value function, enabling efficient
optimization, regularization, and compute-scalable search without retraining.
Empirically, we show that EVOR outperforms baselines on a diverse set of
offline RL tasks, demonstrating the benefit of integrating expressive value
learning into offline RL.

</details>


<div id='cs.AI'></div>

# cs.AI [[Back]](#toc)

### [28] [How to Teach Large Multimodal Models New Skills](https://arxiv.org/abs/2510.08564v1)
*Zhen Zhu,Yiming Gong,Yao Xiao,Yaoyao Liu,Derek Hoiem*

Main category: cs.AI

TL;DR: 该论文研究了如何在不遗忘先前能力的情况下教授大型多模态模型新技能，发现通过选择性更新特定层（自注意力投影层或MLP的Gate&Up层）可以有效学习新技能同时保持原有性能。


<details>
  <summary>Details</summary>
Motivation: 研究如何在连续微调中让大型多模态模型学习新技能而不遗忘先前能力，解决模型在窄域微调后出现的性能下降问题。

Method: 在五个目标技能上进行顺序微调，同时在八个保留基准上监控通用能力，通过输出令牌分布变化分析遗忘现象，并测试两种简单的调优方法：仅更新自注意力投影层或仅更新MLP的Gate&Up层。

Result: 发现窄域微调后的明显"遗忘"现象在后续阶段可以部分恢复，识别出两种有效的调优方法能在获得强大目标增益的同时基本保持保留性能。

Conclusion: 通过选择性更新特定层可以实现在学习新技能的同时最小化对先前能力的遗忘，为大型多模态模型的持续学习提供了实用解决方案。

Abstract: How can we teach large multimodal models (LMMs) new skills without erasing
prior abilities? We study sequential fine-tuning on five target skills while
monitoring general ability on eight held-out benchmarks across three model
families. We observe that apparent "forgetting" on held-out tasks after narrow
fine-tuning can partly recover at later stages. We trace this behavior to a
measurable shift in the output token distribution, manifested through a simple
counting-bias probe that co-varies with forgetting. Guided by this picture, we
identify two simple, robust tuning recipes that learn strongly while limiting
drift: (i) updating only the self-attention projection layers, and (ii)
updating only the MLP Gate&Up while freezing the Down projection. Across models
and tasks, these choices deliver strong target gains while largely preserving
held-out performance. Code is available at
https://github.com/jessemelpolio/LMM_CL

</details>


### [29] [Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling](https://arxiv.org/abs/2510.08470v1)
*Bianca-Mihaela Ganescu,Suchir Salhan,Andrew Caines,Paula Buttery*

Main category: cs.AI

TL;DR: 本文提出了一种轻量级解码器架构，通过动态门控机制自适应融合语言和视觉信息，在BabyLM Challenge 2025视觉赛道的数据约束下实现了竞争性或更优的性能。


<details>
  <summary>Details</summary>
Motivation: 在认知合理的数据量限制下训练视觉语言模型，需要重新思考模型如何整合多模态信息，特别是在BabyLM Challenge 2025的严格数据约束下。

Method: 采用轻量级解码器架构，包含：(1) 基于token的动态门控机制用于自适应融合语言和视觉线索；(2) 特征调制和通道注意力以最大化有限视觉信息的效用；(3) 辅助对比目标用于视觉基础。

Result: 在五个基准测试(BLiMP、BLiMP Supplement、EWoK、Winoground和VQA)上表现出竞争性或优于多模态基线的性能。动态门控机制发现了可解释的模式，无需显式监督即可为内容词偏好视觉线索，为功能词偏好语言线索。

Conclusion: 尽管存在挑战约束的限制（如全局图像嵌入造成的信息瓶颈和数据集分割导致的训练不稳定），但动态门控被确立为高效多模态学习的强大工具，即使在严格约束下也能提供可解释性和性能。

Abstract: Training vision-language models on cognitively-plausible amounts of data
requires rethinking how models integrate multimodal information. Within the
constraints of the Vision track for the BabyLM Challenge 2025, we propose a
lightweight decoder-based architecture with (1) token-wise dynamic gating for
adaptive fusion of linguistic and visual cues, (2) feature modulation and
channel attention to maximise the utility of limited visual information and (3)
auxiliary contrastive objectives for visual grounding. Evaluation on five
benchmarks (BLiMP, BLiMP Supplement, EWoK, Winoground and VQA) shows
competitive or superior performance to multimodal baselines. More notably, our
dynamic gate discovers interpretable patterns without explicit supervision,
favouring visual cues for content words and linguistic cues for function words.
While we identify limitations in the Challenge constraints, such as the
information bottleneck created by global image embeddings and training
instability from the dataset split, our findings establish dynamic gating as a
powerful tool for efficient multimodal learning, offering both interpretability
and performance even under severe constraints.

</details>


<div id='cs.RO'></div>

# cs.RO [[Back]](#toc)

### [30] [BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation](https://arxiv.org/abs/2510.08572v1)
*Rocktim Jyoti Das,Harsh Singh,Diana Turmakhan,Muhammad Abdullah Sohail,Mingfei Han,Preslav Nakov,Fabio Pizzati,Ivan Laptev*

Main category: cs.RO

TL;DR: BLAZER是一个从自动生成数据中学习机器人操作策略的框架，利用LLM规划器的零样本能力在模拟环境中自动生成多样化操作任务的演示数据，成功示例用于微调LLM以提升规划能力，并实现从模拟到传感器操作的无缝迁移。


<details>
  <summary>Details</summary>
Motivation: 机器人领域缺乏互联网规模的多任务演示数据，现有数据集受限于人工收集和整理，限制了策略的泛化性和鲁棒性发展。

Method: 基于LLM规划器的零样本能力自动生成模拟环境中的操作任务演示，利用成功示例微调LLM，无需人工监督即可提升规划能力。

Result: BLAZER显著提升了模拟和真实环境中的零样本操作性能，在训练任务池外的任务上也有改进，并能实现LLM模型的下采样。

Conclusion: BLAZER框架通过自动生成训练数据有效解决了机器人领域数据稀缺问题，为开发更通用和鲁棒的机器人策略提供了可行路径。

Abstract: Scaling data and models has played a pivotal role in the remarkable progress
of computer vision and language. Inspired by these domains, recent efforts in
robotics have similarly focused on scaling both data and model size to develop
more generalizable and robust policies. However, unlike vision and language,
robotics lacks access to internet-scale demonstrations across diverse robotic
tasks and environments. As a result, the scale of existing datasets typically
suffers from the need for manual data collection and curation. To address this
problem, here we propose BLAZER, a framework that learns manipulation policies
from automatically generated training data. We build on the zero-shot
capabilities of LLM planners and automatically generate demonstrations for
diverse manipulation tasks in simulation. Successful examples are then used to
finetune an LLM and to improve its planning capabilities without human
supervision. Notably, while BLAZER training requires access to the simulator's
state, we demonstrate direct transfer of acquired skills to sensor-based
manipulation. Through extensive experiments, we show BLAZER to significantly
improve zero-shot manipulation in both simulated and real environments.
Moreover, BLAZER improves on tasks outside of its training pool and enables
downscaling of LLM models. Our code and data will be made publicly available on
the project page.

</details>


### [31] [Scalable Offline Metrics for Autonomous Driving](https://arxiv.org/abs/2510.08571v1)
*Animikh Aich,Adwait Kulkarni,Eshed Ohn-Bar*

Main category: cs.RO

TL;DR: 该论文评估了自动驾驶系统中感知规划模型的离线与在线性能相关性，发现两者相关性比先前研究更差，并提出基于认知不确定性的离线指标来弥合这一差距。


<details>
  <summary>Details</summary>
Motivation: 当前自动驾驶系统的离线评估难以准确预测在线性能，看似微小的错误在闭环设置中可能导致事故，这种关系在复杂城市场景中研究不足。

Method: 通过大量实验分析离线与在线评估的相关性，研究基于认知不确定性的离线指标来捕捉可能导致闭环错误的场景。

Result: 发现离线与在线性能相关性比先前报告更差，提出的新离线指标相关性提升了13%以上，在真实世界环境中表现更好。

Conclusion: 当前自动驾驶策略评估实践存在有效性疑问，基于认知不确定性的离线指标能更好地弥合离线与在线评估的差距。

Abstract: Real-World evaluation of perception-based planning models for robotic
systems, such as autonomous vehicles, can be safely and inexpensively conducted
offline, i.e., by computing model prediction error over a pre-collected
validation dataset with ground-truth annotations. However, extrapolating from
offline model performance to online settings remains a challenge. In these
settings, seemingly minor errors can compound and result in test-time
infractions or collisions. This relationship is understudied, particularly
across diverse closed-loop metrics and complex urban maneuvers. In this work,
we revisit this undervalued question in policy evaluation through an extensive
set of experiments across diverse conditions and metrics. Based on analysis in
simulation, we find an even worse correlation between offline and online
settings than reported by prior studies, casting doubts on the validity of
current evaluation practices and metrics for driving policies. Next, we bridge
the gap between offline and online evaluation. We investigate an offline metric
based on epistemic uncertainty, which aims to capture events that are likely to
cause errors in closed-loop settings. The resulting metric achieves over 13%
improvement in correlation compared to previous offline metrics. We further
validate the generalization of our findings beyond the simulation environment
in real-world settings, where even greater gains are observed.

</details>


### [32] [NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos](https://arxiv.org/abs/2510.08568v1)
*Hongyu Li,Lingfeng Sun,Yafei Hu,Duy Ta,Jennifer Barry,George Konidaris,Jiahui Fu*

Main category: cs.RO

TL;DR: NovaFlow是一个零样本自主操作框架，能够将任务描述转换为可执行计划，无需演示或特定机器人训练，支持刚性、关节和可变形物体的操作任务。


<details>
  <summary>Details</summary>
Motivation: 现有方法通常假设任务分布内或依赖特定机器人的微调数据，限制了跨平台迁移能力。目标是实现机器人零样本执行新操作任务。

Method: 使用视频生成模型将任务描述合成为视频，通过感知模块提取3D对象流，对刚性物体计算相对位姿并通过抓取提议和轨迹优化实现，对可变形物体使用基于粒子的动力学模型进行跟踪规划。

Result: 在桌面Franka机械臂和Spot四足移动机器人上验证了刚性、关节和可变形物体操作任务的有效零样本执行，无需演示或特定机器人训练。

Conclusion: 通过将任务理解与底层控制解耦，NovaFlow实现了跨机器人平台的零样本操作能力，为机器人自主操作提供了新途径。

Abstract: Enabling robots to execute novel manipulation tasks zero-shot is a central
goal in robotics. Most existing methods assume in-distribution tasks or rely on
fine-tuning with embodiment-matched data, limiting transfer across platforms.
We present NovaFlow, an autonomous manipulation framework that converts a task
description into an actionable plan for a target robot without any
demonstrations. Given a task description, NovaFlow synthesizes a video using a
video generation model and distills it into 3D actionable object flow using
off-the-shelf perception modules. From the object flow, it computes relative
poses for rigid objects and realizes them as robot actions via grasp proposals
and trajectory optimization. For deformable objects, this flow serves as a
tracking objective for model-based planning with a particle-based dynamics
model. By decoupling task understanding from low-level control, NovaFlow
naturally transfers across embodiments. We validate on rigid, articulated, and
deformable object manipulation tasks using a table-top Franka arm and a Spot
quadrupedal mobile robot, and achieve effective zero-shot execution without
demonstrations or embodiment-specific training. Project website:
https://novaflow.lhy.xyz/.

</details>


### [33] [DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model](https://arxiv.org/abs/2510.08556v1)
*Xueyi Liu,He Wang,Li Yi*

Main category: cs.RO

TL;DR: 提出一种新颖的框架，通过联合动力学模型解决机器人手部物体旋转任务中的仿真到现实转换问题，实现了单一策略在真实世界中广泛泛化。


<details>
  <summary>Details</summary>
Motivation: 解决机器人手部物体旋转任务中仿真到现实的转换挑战，克服现有方法在物体几何形状、尺寸、手腕姿态等方面的限制。

Method: 使用联合动力学模型学习现实差距，通过因子化关节动态、压缩系统影响为低维变量，结合自主数据收集策略获取多样化真实交互数据。

Result: 单一策略成功旋转复杂形状物体（如动物模型）、高宽比物体（达5.33）和小尺寸物体，处理多样化手腕方向和旋转轴。

Conclusion: 该方法在真实世界评估和复杂任务遥操作应用中验证了有效性和鲁棒性，展示了前所未有的泛化能力。

Abstract: Achieving generalized in-hand object rotation remains a significant challenge
in robotics, largely due to the difficulty of transferring policies from
simulation to the real world. The complex, contact-rich dynamics of dexterous
manipulation create a "reality gap" that has limited prior work to constrained
scenarios involving simple geometries, limited object sizes and aspect ratios,
constrained wrist poses, or customized hands. We address this sim-to-real
challenge with a novel framework that enables a single policy, trained in
simulation, to generalize to a wide variety of objects and conditions in the
real world. The core of our method is a joint-wise dynamics model that learns
to bridge the reality gap by effectively fitting limited amount of real-world
collected data and then adapting the sim policy's actions accordingly. The
model is highly data-efficient and generalizable across different whole-hand
interaction distributions by factorizing dynamics across joints, compressing
system-wide influences into low-dimensional variables, and learning each
joint's evolution from its own dynamic profile, implicitly capturing these net
effects. We pair this with a fully autonomous data collection strategy that
gathers diverse, real-world interaction data with minimal human intervention.
Our complete pipeline demonstrates unprecedented generality: a single policy
successfully rotates challenging objects with complex shapes (e.g., animals),
high aspect ratios (up to 5.33), and small sizes, all while handling diverse
wrist orientations and rotation axes. Comprehensive real-world evaluations and
a teleoperation application for complex tasks validate the effectiveness and
robustness of our approach. Website: https://meowuu7.github.io/DexNDM/

</details>


### [34] [R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation](https://arxiv.org/abs/2510.08547v1)
*Xiuwei Xu,Angyuan Ma,Hankun Li,Bingyao Yu,Zheng Zhu,Jie Zhou,Jiwen Lu*

Main category: cs.RO

TL;DR: 提出R2RGen框架，通过点云观测-动作对的真实数据增强来实现机器人操作的空间泛化，无需模拟器和渲染，直接从单源演示生成多样化的真实世界数据。


<details>
  <summary>Details</summary>
Motivation: 解决机器人操作中空间泛化问题，传统方法需要大量人工演示覆盖不同空间配置，现有数据生成方法存在显著的模拟到真实差距，且受限于固定基座场景和预定义相机视角。

Method: 提出真实到真实的3D数据生成框架R2RGen，包括细粒度场景和轨迹解析的标注机制、处理复杂多对象组合和任务约束的分组增强策略，以及对齐真实世界3D传感器分布的相机感知处理。

Result: R2RGen在广泛实验中显著提高了数据效率，并展示了在移动操作中扩展和应用的强大潜力。

Conclusion: R2RGen框架通过直接增强真实点云数据，有效解决了机器人操作的空间泛化问题，提供了一种高效、即插即用的数据生成方案。

Abstract: Towards the aim of generalized robotic manipulation, spatial generalization
is the most fundamental capability that requires the policy to work robustly
under different spatial distribution of objects, environment and agent itself.
To achieve this, substantial human demonstrations need to be collected to cover
different spatial configurations for training a generalized visuomotor policy
via imitation learning. Prior works explore a promising direction that
leverages data generation to acquire abundant spatially diverse data from
minimal source demonstrations. However, most approaches face significant
sim-to-real gap and are often limited to constrained settings, such as
fixed-base scenarios and predefined camera viewpoints. In this paper, we
propose a real-to-real 3D data generation framework (R2RGen) that directly
augments the pointcloud observation-action pairs to generate real-world data.
R2RGen is simulator- and rendering-free, thus being efficient and
plug-and-play. Specifically, given a single source demonstration, we introduce
an annotation mechanism for fine-grained parsing of scene and trajectory. A
group-wise augmentation strategy is proposed to handle complex multi-object
compositions and diverse task constraints. We further present camera-aware
processing to align the distribution of generated data with real-world 3D
sensor. Empirically, R2RGen substantially enhances data efficiency on extensive
experiments and demonstrates strong potential for scaling and application on
mobile manipulation.

</details>


### [35] [DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos](https://arxiv.org/abs/2510.08475v1)
*Jhen Hsieh,Kuan-Hsun Tu,Kuo-Han Hung,Tsung-Wei Ke*

Main category: cs.RO

TL;DR: DexMan是一个自动化框架，可将人类视觉演示转换为仿真环境中人形机器人的双手灵巧操作技能，无需相机标定、深度传感器、3D对象资产或真实运动标注。


<details>
  <summary>Details</summary>
Motivation: 解决现有方法仅考虑简化浮动手部、需要大量手动数据收集和昂贵运动捕捉的问题，实现从野外视频直接学习灵巧操作技能。

Method: 直接在第三方人类操作视频上操作，利用基于接触的奖励改进从噪声手-物体姿态估计的策略学习，控制完整人形机器人。

Result: 在TACO基准测试中物体姿态估计达到最先进性能（ADD-S和VSD分别提升0.08和0.12），在OakInk-v2上成功率比先前方法提高19%。

Conclusion: DexMan能够从真实和合成视频生成技能，无需手动数据收集和昂贵运动捕捉，为训练通用灵巧操作创建大规模多样化数据集。

Abstract: We present DexMan, an automated framework that converts human visual
demonstrations into bimanual dexterous manipulation skills for humanoid robots
in simulation. Operating directly on third-person videos of humans manipulating
rigid objects, DexMan eliminates the need for camera calibration, depth
sensors, scanned 3D object assets, or ground-truth hand and object motion
annotations. Unlike prior approaches that consider only simplified floating
hands, it directly controls a humanoid robot and leverages novel contact-based
rewards to improve policy learning from noisy hand-object poses estimated from
in-the-wild videos.
  DexMan achieves state-of-the-art performance in object pose estimation on the
TACO benchmark, with absolute gains of 0.08 and 0.12 in ADD-S and VSD.
Meanwhile, its reinforcement learning policy surpasses previous methods by 19%
in success rate on OakInk-v2. Furthermore, DexMan can generate skills from both
real and synthetic videos, without the need for manual data collection and
costly motion capture, and enabling the creation of large-scale, diverse
datasets for training generalist dexterous manipulation.

</details>


### [36] [Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered](https://arxiv.org/abs/2510.08464v1)
*Jason Jabbour,Dong-Ki Kim,Max Smith,Jay Patrikar,Radhika Ghosal,Youhui Wang,Ali Agha,Vijay Janapa Reddi,Shayegan Omidshafiei*

Main category: cs.RO

TL;DR: GLUESTICK是一种后剪枝恢复方法，通过在权重空间对密集和剪枝模型进行一次性插值计算校正项，恢复VLA模型剪枝后丢失的功能，同时保持稀疏性优势。


<details>
  <summary>Details</summary>
Motivation: VLA模型在资源受限硬件上部署困难，剪枝方法在机器人领域研究不足，且剪枝会导致模型性能急剧下降和安全违规增加。

Method: 在权重空间对密集和剪枝模型进行一次性插值计算校正项，推理时每个剪枝层使用该校正项恢复丢失能力，无需额外训练且与剪枝算法无关。

Result: 在多种VLA架构和操作、导航任务中，GLUESTICK实现了竞争性的内存效率，同时显著恢复成功率并减少安全违规。

Conclusion: GLUESTICK提供了一种无需训练、剪枝算法无关的后剪枝恢复方法，有效平衡了效率与精度之间的权衡。

Abstract: Vision-Language-Action (VLA) models have advanced robotic capabilities but
remain challenging to deploy on resource-limited hardware. Pruning has enabled
efficient compression of large language models (LLMs), yet it is largely
understudied in robotics. Surprisingly, we observe that pruning VLA models
leads to drastic degradation and increased safety violations. We introduce
GLUESTICK, a post-pruning recovery method that restores much of the original
model's functionality while retaining sparsity benefits. Our method performs a
one-time interpolation between the dense and pruned models in weight-space to
compute a corrective term. This correction is used during inference by each
pruned layer to recover lost capabilities with minimal overhead. GLUESTICK
requires no additional training, is agnostic to the pruning algorithm, and
introduces a single hyperparameter that controls the tradeoff between
efficiency and accuracy. Across diverse VLA architectures and tasks in
manipulation and navigation, GLUESTICK achieves competitive memory efficiency
while substantially recovering success rates and reducing safety violations.
Additional material can be found at: https://gluestick-vla.github.io/.

</details>


### [37] [Airy: Reading Robot Intent through Height and Sky](https://arxiv.org/abs/2510.08381v1)
*Baoyang Chen,Xian Xu,Huamin Qu*

Main category: cs.RO

TL;DR: Airy是一个艺术装置，通过两个强化学习训练的机械臂竞争将床单抛向空中的表演，探索如何让复杂的多智能体AI变得直观可理解。基于三个设计原则：竞争作为清晰指标、具身熟悉性和传感器到感官映射，该项目将黑盒AI转化为公共界面。


<details>
  <summary>Details</summary>
Motivation: 随着工业机器人进入共享人类空间，其不透明的决策过程威胁着安全、信任和公共监督。该项目旨在探索复杂多智能体AI是否能变得直观可理解。

Method: 采用三个设计原则：1）竞争作为清晰指标（谁抛得更高）；2）具身熟悉性（观众能识别织物抛掷动作）；3）传感器到感官映射（通过森林和天气投影显示机器人合作或竞争）。在两个强化学习训练的机械臂之间设置床单抛掷竞赛。

Result: 在五个国际展览中的观察表明，观众能够实时解读机器人的策略、冲突和合作，情感反应与系统内部状态相呼应。

Conclusion: 该项目展示了感官隐喻如何将黑盒AI转化为公共界面，使复杂的多智能体系统变得直观可理解。

Abstract: As industrial robots move into shared human spaces, their opaque decision
making threatens safety, trust, and public oversight. This artwork, Airy, asks
whether complex multi agent AI can become intuitively understandable by staging
a competition between two reinforcement trained robot arms that snap a bedsheet
skyward. Building on three design principles, competition as a clear metric
(who lifts higher), embodied familiarity (audiences recognize fabric snapping),
and sensor to sense mapping (robot cooperation or rivalry shown through forest
and weather projections), the installation gives viewers a visceral way to read
machine intent. Observations from five international exhibitions indicate that
audiences consistently read the robots' strategies, conflict, and cooperation
in real time, with emotional reactions that mirror the system's internal state.
The project shows how sensory metaphors can turn a black box into a public
interface.

</details>


### [38] [Evaluation of a Robust Control System in Real-World Cable-Driven Parallel Robots](https://arxiv.org/abs/2510.08270v1)
*Damir Nurtdinov,Aliaksei Korshuk,Alexei Kornaev,Alexander Maloletov*

Main category: cs.RO

TL;DR: 该研究比较了经典PID控制器与现代强化学习算法（DDPG、PPO、TRPO）在欠约束电缆驱动并联机器人控制中的性能，发现TRPO在不同轨迹下均获得最低RMS误差，且对较大时间间隔具有鲁棒性。


<details>
  <summary>Details</summary>
Motivation: 评估经典和现代控制方法在现实世界欠约束电缆驱动并联机器人中的性能，重点关注有限时间离散化条件下的控制效果。

Method: 对经典PID控制器与现代强化学习算法（包括DDPG、PPO和TRPO）进行对比分析，在不同轨迹和不同控制更新时间间隔下测试性能。

Result: TRPO在所有方法中表现最佳，在各种轨迹下获得最低的均方根误差，并且对较大的控制更新时间间隔表现出鲁棒性。TRPO能够平衡探索与利用，在噪声环境中实现稳定控制。

Conclusion: TRPO作为复杂机器人控制任务的鲁棒解决方案具有潜力，特别适用于动态环境，并为传感器融合或混合控制策略的未来应用提供了启示。

Abstract: This study evaluates the performance of classical and modern control methods
for real-world Cable-Driven Parallel Robots (CDPRs), focusing on
underconstrained systems with limited time discretization. A comparative
analysis is conducted between classical PID controllers and modern
reinforcement learning algorithms, including Deep Deterministic Policy Gradient
(DDPG), Proximal Policy Optimization (PPO), and Trust Region Policy
Optimization (TRPO). The results demonstrate that TRPO outperforms other
methods, achieving the lowest root mean square (RMS) errors across various
trajectories and exhibiting robustness to larger time intervals between control
updates. TRPO's ability to balance exploration and exploitation enables stable
control in noisy, real-world environments, reducing reliance on high-frequency
sensor feedback and computational demands. These findings highlight TRPO's
potential as a robust solution for complex robotic control tasks, with
implications for dynamic environments and future applications in sensor fusion
or hybrid control strategies.

</details>


### [39] [Accurate and Noise-Tolerant Extraction of Routine Logs in Robotic Process Automation (Extended Version)](https://arxiv.org/abs/2510.08118v1)
*Massimiliano de Leoni,Faizan Ahmed Khan,Simone Agostinelli*

Main category: cs.RO

TL;DR: 本文提出了一种基于聚类的技术，用于从用户界面日志中提取常规日志，以支持机器人流程自动化。该技术在存在噪声（即人类执行中的自然变异和偶然错误）的情况下表现优于现有方法。


<details>
  <summary>Details</summary>
Motivation: 现有的大多数工作没有直接关注于实现模型发现，仅限于提取构成常规操作的动作集合。这些方法也未在存在不一致常规执行（即噪声）的场景中进行评估，这反映了人类表现中的自然变异和偶然错误。

Method: 采用基于聚类的技术来提取常规日志。在九个具有不同注入噪声水平的UI日志上进行了实验，并与现有技术进行了比较。

Result: 通过标准的最先进指标评估结果，表明我们的技术能够提取比现有技术更准确的常规日志，特别是在存在噪声的情况下。

Conclusion: 所提出的聚类技术能够有效处理噪声，提取更准确的常规日志，为机器人流程自动化提供了更好的支持。

Abstract: Robotic Process Mining focuses on the identification of the routine types
performed by human resources through a User Interface. The ultimate goal is to
discover routine-type models to enable robotic process automation. The
discovery of routine-type models requires the provision of a routine log.
Unfortunately, the vast majority of existing works do not directly focus on
enabling the model discovery, limiting themselves to extracting the set of
actions that are part of the routines. They were also not evaluated in
scenarios characterized by inconsistent routine execution, hereafter referred
to as noise, which reflects natural variability and occasional errors in human
performance. This paper presents a clustering-based technique that aims to
extract routine logs. Experiments were conducted on nine UI logs from the
literature with different levels of injected noise. Our technique was compared
with existing techniques, most of which are not meant to discover routine logs
but were adapted for the purpose. The results were evaluated through standard
state-of-the-art metrics, showing that we can extract more accurate routine
logs than what the state of the art could, especially in the presence of noise.

</details>


### [40] [Beyond hospital reach: Autonomous lightweight ultrasound robot for liver sonography](https://arxiv.org/abs/2510.08106v1)
*Zihan Li,Yixiao Xu,Lei Zhang,Taiyu Han,Xinshan Yang,Yingni Wang,Mingxuan Liu,Shenghai Xin,Linxun Liu,Hongen Liao,Guochen Ning*

Main category: cs.RO

TL;DR: 开发了一个轻量级自主超声机器人系统，通过AI智能体和6自由度电缆驱动机器人，能够自主获取专家级标准肝脏超声平面并在患者中检测病理，特别适用于医疗资源有限地区。


<details>
  <summary>Details</summary>
Motivation: 肝脏疾病是全球主要健康负担，超声是首选诊断工具，但肝脏超声检查需要定位多个非连续平面，需要专业知识。然而，在资源有限地区专家超声医师严重短缺。

Method: 开发了一个自主轻量级超声机器人系统，包括集成多模态感知与记忆注意力的AI智能体，用于定位未见目标结构，以及一个588克6自由度电缆驱动机器人。通过安装在腹部增强运动鲁棒性。

Result: 机器人能够自主获取专家级标准肝脏超声平面，并在患者中检测病理，包括来自海拔2261米医疗资源有限的西宁的两名患者。系统在快速运动个体和野外环境中表现有效。

Conclusion: 这项工作代表了在多个挑战性场景下自主超声检查的首个演示，可能改变服务不足地区获得专家级诊断的机会。

Abstract: Liver disease is a major global health burden. While ultrasound is the
first-line diagnostic tool, liver sonography requires locating multiple
non-continuous planes from positions where target structures are often not
visible, for biometric assessment and lesion detection, requiring significant
expertise. However, expert sonographers are severely scarce in resource-limited
regions. Here, we develop an autonomous lightweight ultrasound robot comprising
an AI agent that integrates multi-modal perception with memory attention for
localization of unseen target structures, and a 588-gram 6-degrees-of-freedom
cable-driven robot. By mounting on the abdomen, the system enhances robustness
against motion. Our robot can autonomously acquire expert-level standard liver
ultrasound planes and detect pathology in patients, including two from Xining,
a 2261-meter-altitude city with limited medical resources. Our system performs
effectively on rapid-motion individuals and in wilderness environments. This
work represents the first demonstration of autonomous sonography across
multiple challenging scenarios, potentially transforming access to expert-level
diagnostics in underserved regions.

</details>


<div id='econ.GN'></div>

# econ.GN [[Back]](#toc)

### [41] [A data fusion approach for mobility hub impact assessment and location selection: integrating hub usage data into a large-scale mode choice model](https://arxiv.org/abs/2510.08366v1)
*Xiyuan Ren,Joseph Y. J. Chow*

Main category: econ.GN

TL;DR: 本研究提出了一种新颖的数据融合方法，将观察到的移动枢纽使用情况整合到基于合成出行数据的模式选择模型中，用于评估移动枢纽对出行需求、模式转换、车辆行驶里程减少和消费者剩余增加的影响。


<details>
  <summary>Details</summary>
Motivation: 城市面临交通拥堵和服务不平等问题，移动枢纽提供了可扩展的解决方案，但评估其影响仍然具有挑战性，因为缺乏整合大规模出行模式与现实枢纽使用行为的行为模型。

Method: 采用数据融合方法，识别可能受移动枢纽影响的出行，构建多模式子选择集，然后使用现场调查数据和真实出行计数校准枢纽特定参数。

Result: 在纽约首都区的案例研究中，两个已实施的枢纽预计每天分别产生8.83和6.17次多模式出行，每年减少车辆行驶里程20.37和13.16千英里，每日增加消费者剩余4000美元和1742美元。

Conclusion: 位于城际走廊和城市边缘、支持停车换乘模式的枢纽候选点显示出最显著的行为影响。

Abstract: As cities grapple with traffic congestion and service inequities, mobility
hubs offer a scalable solution to align increasing travel demand with
sustainability goals. However, evaluating their impacts remains challenging due
to the lack of behavioral models that integrate large-scale travel patterns
with real-world hub usage. This study presents a novel data fusion approach
that incorporates observed mobility hub usage into a mode choice model
estimated with synthetic trip data. We identify trips potentially affected by
mobility hubs and construct a multimodal sub-choice set, then calibrate
hub-specific parameters using on-site survey data and ground truth trip counts.
The enhanced model is used to evaluate mobility hub impacts on potential
demand, mode shift, reduced vehicle miles traveled (VMT), and increased
consumer surplus (CS). We apply this method to a case study in the Capital
District, NY, using data from a survey conducted by the Capital District
Transportation Authority (CDTA) and a mode choice model estimated using Replica
Inc. synthetic data. The two implemented hubs located near UAlbany Downtown
Campus and in Downtown Cohoes are projected to generate 8.83 and 6.17
multimodal trips per day, reduce annual VMT by 20.37 and 13.16 thousand miles,
and increase daily CS by $4,000 and $1,742, respectively. An evaluation of
potential hub candidates in the Albany-Schenectady-Troy metropolitan area with
the estimated models demonstrates that hubs located along intercity corridors
and at urban peripheries, supporting park-and-ride P+R patterns, yield the most
significant behavioral impacts.

</details>


<div id='quant-ph'></div>

# quant-ph [[Back]](#toc)

### [42] [Quartic quantum speedups for community detection](https://arxiv.org/abs/2510.08494v1)
*Alexander Schmidhuber,Alexander Zlokapa*

Main category: quant-ph

TL;DR: 本文开发了一种用于超图社区检测的量子算法，相比最佳经典算法实现了四次方量子加速和超多项式空间节省。该算法基于Kikuchi方法，扩展到广义随机块模型，并证明了在低度框架下的（近似）最优性。


<details>
  <summary>Details</summary>
Motivation: 社区检测是数据科学的基础问题，扩展到超图可以捕捉超越成对交互的高阶相关性。现有方法在计算效率上存在限制，需要开发更高效的算法。

Method: 基于Kikuchi方法，开发了量化版本的Kikuchi算法，通过高效准备与底层社区结构相关的引导状态来实现量子加速。该方法扩展到广义随机块模型家族。

Result: 算法实现了相对于最佳经典算法的四次方量子加速和超多项式空间节省。在低度框架下证明了匹配下界，表明算法达到了平滑的统计-计算权衡。

Conclusion: Kikuchi方法在量子计算中的加速效果比之前认为的更稳健，能够涵盖更广泛的问题。边际阶可能是表征这些量子加速存在性的关键量。

Abstract: Community detection is a foundational problem in data science. Its natural
extension to hypergraphs captures higher-order correlations beyond pairwise
interactions. In this work, we develop a quantum algorithm for hypergraph
community detection that achieves a quartic quantum speedup over the best known
classical algorithm, along with superpolynomial savings in space. Our algorithm
is based on the Kikuchi method, which we extend beyond previously considered
problems such as Tensor PCA and $p$XORSAT to a broad family of generalized
stochastic block models. To demonstrate (near) optimality of this method, we
prove matching lower bounds (up to logarithmic factors) in the low-degree
framework, showing that the algorithm saturates a smooth
statistical-computational tradeoff. The quantum speedup arises from a quantized
version of the Kikuchi method and is based on the efficient preparation of a
guiding state correlated with the underlying community structure. Our work
suggests that prior quantum speedups using the Kikuchi method are sufficiently
robust to encompass a broader set of problems than previously believed; we
conjecture that a quantity known as marginal order characterizes the existence
of these quantum speedups.

</details>


### [43] [Continuous Variable Hamiltonian Learning at Heisenberg Limit via Displacement-Random Unitary Transformation](https://arxiv.org/abs/2510.08419v1)
*Xi Huang,Lixing Zhang,Di Luo*

Main category: quant-ph

TL;DR: 提出了一种名为位移-随机酉变换（D-RUT）的高效实验协议，用于学习任意有限阶玻色子哈密顿量的系数，总演化时间按O(1/ε)缩放，对SPAM误差具有鲁棒性。


<details>
  <summary>Details</summary>
Motivation: 连续变量量子系统的哈密顿量表征面临无限维希尔伯特空间和无界算子的困难，现有协议通常限于特定哈密顿结构或需要实验上具有挑战性的资源。

Method: 开发了位移-随机酉变换（D-RUT）协议，对于多模系统采用分层系数恢复策略，并将协议扩展到第一量子化以学习位置和动量算符表示的哈密顿量基本物理参数。

Result: 该协议能够以海森堡极限精度学习哈密顿量系数，总演化时间按O(1/ε)缩放，对SPAM误差具有鲁棒性，且具有优越的统计效率。

Conclusion: D-RUT协议为连续变量量子系统的哈密顿量表征提供了一种高效且实验上可实现的解决方案，能够达到海森堡极限精度。

Abstract: Characterizing the Hamiltonians of continuous-variable (CV) quantum systems
is a fundamental challenge laden with difficulties arising from
infinite-dimensional Hilbert spaces and unbounded operators. Existing protocols
for achieving the Heisenberg limit precision are often restricted to specific
Hamiltonian structures or demand experimentally challenging resources. In this
work, we introduce an efficient and experimentally accessible protocol, the
Displacement-Random Unitary Transformation (D-RUT), that learns the
coefficients of general, arbitrary finite-order bosonic Hamiltonians with a
total evolution time scaling as $O(1/\epsilon)$ for a target precision
$\epsilon$ robust to SPAM error. For multi-mode systems, we develop a
hierarchical coefficients recovering strategy with superior statistical
efficiency. Furthermore, we extend our protocol to first quantization, enabling
the learning of fundamental physical parameters from Hamiltonians expressed in
position and momentum operators at the Heisenberg limit.

</details>


<div id='math-ph'></div>

# math-ph [[Back]](#toc)

### [44] [Quantum variance and fluctuations for Walsh-quantized baker's maps](https://arxiv.org/abs/2510.08321v1)
*Laura Shou*

Main category: math-ph

TL;DR: 该论文研究了Walsh量子化的面包师映射在环面上的量子混沌模型。研究表明，除了D=4的特殊情况外，对于所有D≥2的面包师映射缩放因子，随机特征基的缩放矩阵元素波动在N→∞的半经典极限下通常呈渐近高斯分布，方差由经典面包师映射关联给出。这确定了这些特征基在量子遍历定理中的精确收敛速率，并获得了特征态热化假设(ETH)的一个版本。


<details>
  <summary>Details</summary>
Motivation: 研究量子混沌系统中特征态的统计特性，特别是矩阵元素的波动分布，以理解量子遍历定理的收敛速率和特征态热化假设在具体模型中的表现。

Method: 使用Walsh量子化的面包师映射作为量子混沌模型，分析随机特征基的缩放矩阵元素波动在N→∞的半经典极限下的分布特性，通过经典面包师映射关联计算方差。

Result: 对于所有D≥2（除D=4外），随机特征基的缩放矩阵元素波动通常呈渐近高斯分布，方差由经典关联决定。获得了特征态热化假设的版本，包括非对角矩阵元素的极限复高斯分布。D=4的情况依赖于经典可观测量在环面分形子集上的值。

Conclusion: Walsh量子化面包师映射的特征态虽然是随机的，但具有微观关联，使其区别于Haar随机向量。这些关联通过经典面包师映射关联体现在矩阵元素波动的方差中，为量子遍历定理提供了精确的收敛速率。

Abstract: The Walsh-quantized baker's maps are models for quantum chaos on the torus.
We show that for all baker's map scaling factors $D\ge2$ except for $D=4$,
typically (in the sense of Haar measure on the eigenspaces, which are
degenerate) the empirical distribution of the scaled matrix element
fluctuations $\sqrt{N}\{\langle
\varphi^{(j)}|\operatorname{Op}_{k,\ell}(a)|\varphi^{(j)}\rangle-\int_{\mathbb{T}^2}a\}_{j=1}^{N}$
for a random eigenbasis $\{\varphi^{(j)}\}_{j=1}^{N}$ is asymptotically
Gaussian in the semiclassical limit $N\to\infty$, with variance given in terms
of classical baker's map correlations. This determines the precise rate of
convergence in the quantum ergodic theorem for these eigenbases. We obtain a
version of the Eigenstate Thermalization Hypothesis (ETH) for these
eigenstates, including a limiting complex Gaussian distribution for the
off-diagonal matrix elements, with variances also given in terms of classical
correlations. The presence of the classical correlations highlights that these
eigenstates, while random, have microscopic correlations that differentiate
them from Haar random vectors. For the single value $D=4$, the Gaussianity of
the matrix element fluctuations depends on the values of the classical
observable on a fractal subset of the torus.

</details>


<div id='cond-mat.mtrl-sci'></div>

# cond-mat.mtrl-sci [[Back]](#toc)

### [45] [Multimodal Topological Textures Arising from Coupled Structural Orders in SrTiO$_3$](https://arxiv.org/abs/2510.08186v1)
*Fernando Gómez-Ortiz,Louis Bastogne,Philippe Ghosez*

Main category: cond-mat.mtrl-sci

TL;DR: 该研究将拓扑结构概念扩展到非极性原子自由度，在SrTiO3中发现并稳定了反铁畸变拓扑纹理，包括180°畴壁、涡旋和反涡旋结构，这些结构与局域极化涡旋和应变场共同形成三模态拓扑结构。


<details>
  <summary>Details</summary>
Motivation: 磁自旋拓扑纹理最近在极性拓扑中发现了电学对应物，本研究旨在将这一概念进一步扩展到其他非极性原子自由度，探索钙钛矿氧化物中非极性结构自由度的拓扑有序性。

Method: 采用第二性原理原子模拟方法，以SrTiO3为原型系统，研究氧八面体自然反铁畸变旋转相关的平衡畴结构和拓扑纹理，分析在压缩外延应变下的畴壁取向和构型。

Result: 识别出新的亚稳态180°畴壁，发现反铁畸变的Bloch型和Néel型构型，其中后者更稳定；稳定了反铁畸变涡旋和反涡旋结构，这些结构与共局域化的极化涡旋和复杂局域应变场模式相结合，形成三模态拓扑结构。

Conclusion: 研究结果将拓扑有序概念扩展到非极性结构自由度，并强调了晶格介导耦合在稳定钙钛矿氧化物中复杂纹理结构中的重要作用。

Abstract: Magnetic spin topological textures recently found their electrical
counterparts in polar topologies emerging from the condensation of
inhomogeneous polar atomic distortions. Here, we further extend the concept to
other non-polar atomic degrees of freedom. Taking SrTiO$_3$ as a prototypical
example, we investigate from second-principles atomistic simulations, the
equilibrium domain structures and topological textures associated with the
natural antiferrodistortive rotations of its oxygen octahedra. % Besides the
common 90$^\circ$ antiferrodistortive domain walls (twin boundaries), we
identify new metastable 180$^\circ$ domain walls oriented along the
$\lbrace100\rbrace_\mathrm{pc}$ direction, when compressive epitaxial strain is
applied. These domains exhibit complex antiferrodistortive Bloch- and
N\'eel-like configurations with the later being the most favorable. We also
stabilize antiferrodistortive vortex and antivortex structures which are
accompanied by co-localized polarization vortices and a complex pattern of the
local strain field, giving rise to a trimodal topological structures. Our
results extends the concept of topological ordering to non-polar structural
degrees of freedom and highlights the role of lattice-mediated couplings in
stabilizing complex textures in perovskite oxides.

</details>
