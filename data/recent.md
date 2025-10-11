<div id=toc></div>

# Table of Contents

- [cs.CV](#cs.CV) [Total: 13]
- [cs.CL](#cs.CL) [Total: 1]
- [cs.GR](#cs.GR) [Total: 1]
- [cs.AI](#cs.AI) [Total: 2]
- [quant-ph](#quant-ph) [Total: 1]
- [cs.RO](#cs.RO) [Total: 6]
- [cs.LG](#cs.LG) [Total: 1]


<div id='cs.CV'></div>

# cs.CV [[Back]](#toc)

### [1] [MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning](https://arxiv.org/abs/2510.08567v1)
*Tajamul Ashraf,Umair Nawaz,Abdelrahman M. Shaker,Rao Anwer,Philip Torr,Fahad Shahbaz Khan,Salman Khan*

Main category: cs.CV

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='cs.CL'></div>

# cs.CL [[Back]](#toc)

### [14] [Which Heads Matter for Reasoning? RL-Guided KV Cache Compression](https://arxiv.org/abs/2510.08525v1)
*Wenjie Du,Li Jiang,Keda Tao,Xue Liu,Huan Wang*

Main category: cs.CL

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='cs.GR'></div>

# cs.GR [[Back]](#toc)

### [15] [X2Video: Adapting Diffusion Models for Multimodal Controllable Neural Video Rendering](https://arxiv.org/abs/2510.08530v1)
*Zhitong Huang,Mohan Zhang,Renhan Wang,Rui Tang,Hao Zhu,Jing Liao*

Main category: cs.GR

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='cs.AI'></div>

# cs.AI [[Back]](#toc)

### [16] [How to Teach Large Multimodal Models New Skills](https://arxiv.org/abs/2510.08564v1)
*Zhen Zhu,Yiming Gong,Yao Xiao,Yaoyao Liu,Derek Hoiem*

Main category: cs.AI

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [17] [Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling](https://arxiv.org/abs/2510.08470v1)
*Bianca-Mihaela Ganescu,Suchir Salhan,Andrew Caines,Paula Buttery*

Main category: cs.AI

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='quant-ph'></div>

# quant-ph [[Back]](#toc)

### [18] [Quartic quantum speedups for community detection](https://arxiv.org/abs/2510.08494v1)
*Alexander Schmidhuber,Alexander Zlokapa*

Main category: quant-ph

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='cs.RO'></div>

# cs.RO [[Back]](#toc)

### [19] [BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation](https://arxiv.org/abs/2510.08572v1)
*Rocktim Jyoti Das,Harsh Singh,Diana Turmakhan,Muhammad Abdullah Sohail,Mingfei Han,Preslav Nakov,Fabio Pizzati,Ivan Laptev*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [20] [Scalable Offline Metrics for Autonomous Driving](https://arxiv.org/abs/2510.08571v1)
*Animikh Aich,Adwait Kulkarni,Eshed Ohn-Bar*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [21] [NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos](https://arxiv.org/abs/2510.08568v1)
*Hongyu Li,Lingfeng Sun,Yafei Hu,Duy Ta,Jennifer Barry,George Konidaris,Jiahui Fu*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [22] [DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model](https://arxiv.org/abs/2510.08556v1)
*Xueyi Liu,He Wang,Li Yi*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [23] [R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation](https://arxiv.org/abs/2510.08547v1)
*Xiuwei Xu,Angyuan Ma,Hankun Li,Bingyao Yu,Zheng Zhu,Jie Zhou,Jiwen Lu*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


### [24] [DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos](https://arxiv.org/abs/2510.08475v1)
*Jhen Hsieh,Kuan-Hsun Tu,Kuo-Han Hung,Tsung-Wei Ke*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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


<div id='cs.LG'></div>

# cs.LG [[Back]](#toc)

### [25] [Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models](https://arxiv.org/abs/2510.08492v1)
*Sharut Gupta,Shobhita Sundaram,Chenyu Wang,Stefanie Jegelka,Phillip Isola*

Main category: cs.LG

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

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
