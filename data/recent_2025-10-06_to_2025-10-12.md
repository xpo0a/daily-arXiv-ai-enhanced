<div id=toc></div>

# Table of Contents

- [cs.CV](#cs.CV) [Total: 9]
- [cs.CL](#cs.CL) [Total: 3]
- [cs.CR](#cs.CR) [Total: 1]
- [cs.AI](#cs.AI) [Total: 3]
- [physics.app-ph](#physics.app-ph) [Total: 1]
- [cs.LG](#cs.LG) [Total: 1]
- [cs.AR](#cs.AR) [Total: 2]
- [cs.RO](#cs.RO) [Total: 15]
- [math.RT](#math.RT) [Total: 1]


<div id='cs.CV'></div>

# cs.CV [[Back]](#toc)

### [1] [EGD-YOLO: A Lightweight Multimodal Framework for Robust Drone-Bird Discrimination via Ghost-Enhanced YOLOv8n and EMA Attention under Adverse Condition](https://arxiv.org/abs/2510.10765v1)
*Sudipto Sarkar,Mohammad Asif Hasan,Khondokar Ashik Shahriar,Fablia Labiba,Nahian Tasnim,Sheikh Anawarul Haq Fattah*

Main category: cs.CV

TL;DR: 本文提出了EGD-YOLOv8n模型，一种轻量级但强大的目标检测模型，专门用于准确识别无人机和鸟类，使用VIP CUP 2025数据集中的RGB和红外图像。


<details>
  <summary>Details</summary>
Motivation: 正确识别无人机和鸟类对于保障空中安全和提升安防系统至关重要，需要开发既准确又高效的检测模型。

Method: 采用EGD-YOLOv8n模型，通过智能设计改进和注意力层来增强特征捕获能力，减少计算量；使用特殊检测头适应不同形状大小的目标；训练了RGB、红外和两者结合的三个版本。

Result: 结合RGB和红外图像的模型取得了最佳准确性和可靠性，同时在普通GPU上能够实现实时运行。

Conclusion: EGD-YOLOv8n模型在无人机和鸟类检测任务中表现出色，结合多模态数据能够显著提升检测性能，适合实时应用场景。

Abstract: Identifying drones and birds correctly is essential for keeping the skies
safe and improving security systems. Using the VIP CUP 2025 dataset, which
provides both RGB and infrared (IR) images, this study presents EGD-YOLOv8n, a
new lightweight yet powerful model for object detection. The model improves how
image features are captured and understood, making detection more accurate and
efficient. It uses smart design changes and attention layers to focus on
important details while reducing the amount of computation needed. A special
detection head helps the model adapt to objects of different shapes and sizes.
We trained three versions: one using RGB images, one using IR images, and one
combining both. The combined model achieved the best accuracy and reliability
while running fast enough for real-time use on common GPUs.

</details>


### [2] [Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey](https://arxiv.org/abs/2510.10671v1)
*Jinxuan Li,Chaolei Tan,Haoxuan Chen,Jianxin Ma,Jian-Fang Hu,Wei-Shi Zheng,Jianhuang Lai*

Main category: cs.CV

TL;DR: 本调查首次全面回顾了图像到视频迁移学习这一新兴领域，系统分类了现有策略（冻结特征和修改特征），分析了在不同粒度视频文本任务中的应用，并通过实验验证了不同范式的有效性，最后指出了当前挑战和未来研究方向。


<details>
  <summary>Details</summary>
Motivation: 图像语言基础模型在图像文本理解/生成任务中表现出色，但视频文本学习需要大量数据和计算资源。图像到视频迁移学习能够有效缓解这一问题，利用现有图像模型提升视频任务性能。

Method: 将现有图像到视频迁移学习策略系统分为两类：冻结特征（保持原始表征）和修改特征（对表征进行修改），并详细阐述了这些策略在不同粒度视频文本任务中的应用。

Result: 通过详细的实验分析，调查了不同图像到视频迁移学习范式在一系列下游视频理解任务中的有效性。

Conclusion: 该调查为基于现有图像语言基础模型推进视频文本学习提供了结构化路线图，并激发了这一快速发展领域的未来研究方向。

Abstract: Image-Language Foundation Models (ILFM) have demonstrated remarkable success
in image-text understanding/generation tasks, providing transferable multimodal
representations that generalize across diverse downstream image-based tasks.
The advancement of video-text research has spurred growing interest in
extending image-based models to the video domain. This paradigm, known as
image-to-video transfer learning, succeeds in alleviating the substantial data
and computational requirements associated with training video-language
foundation models from scratch for video-text learning. This survey provides
the first comprehensive review of this emerging field, which begins by
summarizing the widely used ILFM and their capabilities. We then systematically
classify existing image-to-video transfer learning strategies into two
categories: frozen features and modified features, depending on whether the
original representations from ILFM are preserved or undergo modifications.
Building upon the task-specific nature of image-to-video transfer, this survey
methodically elaborates these strategies and details their applications across
a spectrum of video-text learning tasks, ranging from fine-grained (e.g.,
spatio-temporal video grounding) to coarse-grained (e.g., video question
answering). We further present a detailed experimental analysis to investigate
the efficacy of different image-to-video transfer learning paradigms on a range
of downstream video understanding tasks. Finally, we identify prevailing
challenges and highlight promising directions for future research. By offering
a comprehensive and structured overview, this survey aims to establish a
structured roadmap for advancing video-text learning based on existing ILFM,
and to inspire future research directions in this rapidly evolving domain.

</details>


### [3] [ViSurf: Visual Supervised-and-Reinforcement Fine-Tuning for Large Vision-and-Language Models](https://arxiv.org/abs/2510.10606v1)
*Yuqi Liu,Liangyu Chen,Jiazhen Liu,Mingkang Zhu,Zhisheng Zhong,Bei Yu,Jiaya Jia*

Main category: cs.CV

TL;DR: ViSurf是一个统一的后训练范式，将监督微调(SFT)和基于可验证奖励的强化学习(RLVR)的优势整合在单一阶段中，通过在RLVR过程中注入真实标签，同时提供外部监督和内部强化。


<details>
  <summary>Details</summary>
Motivation: 现有的大视觉语言模型后训练方法中，SFT往往导致次优性能，而RLVR在处理超出模型内部知识库的任务时存在困难。需要一种能结合两者优势的统一方法。

Method: 提出ViSurf范式，分析推导SFT和RLVR目标以建立统一目标，在RLVR过程中注入真实标签，同时提供外部监督和内部强化，并引入三种新的奖励控制策略来稳定和优化训练过程。

Result: 在多个不同基准测试上的广泛实验表明，ViSurf优于单独的SFT、RLVR以及两阶段SFT→RLVR方法。深入分析验证了ViSurf的推导和设计原则。

Conclusion: ViSurf通过统一监督和强化学习范式，有效解决了现有方法的局限性，为大型视觉语言模型的后训练提供了更优的解决方案。

Abstract: Typical post-training paradigms for Large Vision-and-Language Models (LVLMs)
include Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable
Rewards (RLVR). SFT leverages external guidance to inject new knowledge,
whereas RLVR utilizes internal reinforcement to enhance reasoning capabilities
and overall performance. However, our analysis reveals that SFT often leads to
sub-optimal performance, while RLVR struggles with tasks that exceed the
model's internal knowledge base. To address these limitations, we propose
ViSurf (\textbf{Vi}sual \textbf{Su}pervised-and-\textbf{R}einforcement
\textbf{F}ine-Tuning), a unified post-training paradigm that integrates the
strengths of both SFT and RLVR within a single stage. We analyze the derivation
of the SFT and RLVR objectives to establish the ViSurf objective, providing a
unified perspective on these two paradigms. The core of ViSurf involves
injecting ground-truth labels into the RLVR rollouts, thereby providing
simultaneous external supervision and internal reinforcement. Furthermore, we
introduce three novel reward control strategies to stabilize and optimize the
training process. Extensive experiments across several diverse benchmarks
demonstrate the effectiveness of ViSurf, outperforming both individual SFT,
RLVR, and two-stage SFT \textrightarrow RLVR. In-depth analysis corroborates
these findings, validating the derivation and design principles of ViSurf.

</details>


### [4] [GLOFNet -- A Multimodal Dataset for GLOF Monitoring and Prediction](https://arxiv.org/abs/2510.10546v1)
*Zuha Fatima,Muhammad Anser Sohaib,Muhammad Talha,Sidra Sultana,Ayesha Kanwal,Nazia Perwaiz*

Main category: cs.CV

TL;DR: GLOFNet是一个用于冰川湖溃决洪水监测和预测的多模态数据集，整合了Sentinel-2多光谱影像、NASA冰川速度数据和MODIS地表温度数据，重点关注喀喇昆仑山脉的Shisper冰川。


<details>
  <summary>Details</summary>
Motivation: 冰川湖溃决洪水是高山地区罕见但具有破坏性的灾害，现有研究受限于碎片化和单模态数据，缺乏能够结合视觉指标与物理前兆的统一数据集来进行预测。

Method: 整合三种互补数据源：Sentinel-2多光谱影像用于空间监测、NASA ITS_LIVE速度产品用于冰川运动学、MODIS地表温度记录。预处理包括云掩膜、质量过滤、归一化、时间插值、数据增强和循环编码，然后进行多模态协调。

Result: 探索性分析揭示了冰川速度的季节性周期、每十年约0.8K的长期变暖趋势以及冰冻圈条件的空间异质性。GLOFNet数据集已公开可用。

Conclusion: GLOFNet通过解决类别不平衡、云污染和粗分辨率等挑战，为罕见灾害预测的多模态深度学习方法提供了结构化基准基础。

Abstract: Glacial Lake Outburst Floods (GLOFs) are rare but destructive hazards in high
mountain regions, yet predictive research is hindered by fragmented and
unimodal data. Most prior efforts emphasize post-event mapping, whereas
forecasting requires harmonized datasets that combine visual indicators with
physical precursors. We present GLOFNet, a multimodal dataset for GLOF
monitoring and prediction, focused on the Shisper Glacier in the Karakoram. It
integrates three complementary sources: Sentinel-2 multispectral imagery for
spatial monitoring, NASA ITS_LIVE velocity products for glacier kinematics, and
MODIS Land Surface Temperature records spanning over two decades. Preprocessing
included cloud masking, quality filtering, normalization, temporal
interpolation, augmentation, and cyclical encoding, followed by harmonization
across modalities. Exploratory analysis reveals seasonal glacier velocity
cycles, long-term warming of ~0.8 K per decade, and spatial heterogeneity in
cryospheric conditions. The resulting dataset, GLOFNet, is publicly available
to support future research in glacial hazard prediction. By addressing
challenges such as class imbalance, cloud contamination, and coarse resolution,
GLOFNet provides a structured foundation for benchmarking multimodal deep
learning approaches to rare hazard prediction.

</details>


### [5] [Receptive Field Expanded Look-Up Tables for Vision Inference: Advancing from Low-level to High-level Tasks](https://arxiv.org/abs/2510.10522v1)
*Xi Zhang,Xiaolin Wu*

Main category: cs.CV

TL;DR: 提出了一种新颖的CNN快速推理方法，通过优化格向量量化器、不规则扩张卷积和U形级联LUT结构，在固定表格大小下扩展感受野，显著提升LUT驱动的CNN推理性能。


<details>
  <summary>Details</summary>
Motivation: 现有LUT方法由于表格大小组合爆炸问题，卷积核感受野受限，影响了CNN推理性能。本研究旨在在保持相同空间复杂度的情况下扩展CNN感受野。

Method: 1. 学习最优格向量量化器，根据数据维度重要性自适应分配量化分辨率；2. 引入不规则扩张卷积；3. 设计U形级联LUT结构，捕获多级上下文信息。

Result: 该方法在速度、精度和内存效率之间取得有效平衡，相比现有LUT方法有显著改进。

Conclusion: 提出的创新方法成功解决了LUT方法感受野受限的问题，实现了更高效的CNN快速推理。

Abstract: Recently, several look-up table (LUT) methods were developed to greatly
expedite the inference of CNNs in a classical strategy of trading space for
speed. However, these LUT methods suffer from a common drawback of limited
receptive field of the convolution kernels due to the combinatorial explosion
of table size. This research aims to expand the CNN receptive field with a
fixed table size, thereby enhancing the performance of LUT-driven fast CNN
inference while maintaining the same space complexity. To achieve this goal,
various techniques are proposed. The main contribution is a novel approach of
learning an optimal lattice vector quantizer that adaptively allocates the
quantization resolution across data dimensions based on their significance to
the inference task. In addition, the lattice vector quantizer offers an
inherently more accurate approximation of CNN kernels than scalar quantizer as
used in current practice. Furthermore, we introduce other receptive field
expansion strategies, including irregular dilated convolutions and a U-shaped
cascaded LUT structure, designed to capture multi-level contextual information
without inflating table size. Together, these innovations allow our approach to
effectively balance speed, accuracy, and memory efficiency, demonstrating
significant improvements over existing LUT methods.

</details>


### [6] [VR-Thinker: Boosting Video Reward Models through Thinking-with-Image Reasoning](https://arxiv.org/abs/2510.10518v2)
*Qunzhong Wang,Jie Liu,Jiajun Liang,Yilei Jiang,Yuanxing Zhang,Jinyuan Chen,Yaozhi Zheng,Xintao Wang,Pengfei Wan,Xiangyu Yue,Jiaheng Liu*

Main category: cs.CV

TL;DR: VR-Thinker是一个思考式图像框架，通过视觉推理操作和可配置视觉记忆窗口，让奖励模型能主动获取和更新视觉证据，解决了传统多模态奖励模型中视觉输入消耗大上下文预算和导致幻觉的问题。


<details>
  <summary>Details</summary>
Motivation: 当前多模态奖励模型面临两个固有局限：(1)视觉输入消耗大量上下文预算，导致帧数减少和细节丢失；(2)所有视觉信息都打包到初始提示中，加剧了链式推理中的幻觉和遗忘问题。

Method: 提出VR-Thinker框架，包含视觉推理操作和可配置视觉记忆窗口。采用强化微调流程：冷启动使用视觉链式思维数据，拒绝采样微调高质量轨迹，应用组相对策略优化来增强推理能力。

Result: 在视频偏好基准测试中达到最先进精度，特别是对长视频：7B VR-Thinker在VideoGen Reward上达到80.5%，GenAI-Bench上82.3%，MJ-Bench-Video上75.6%。

Conclusion: 验证了思考式图像多模态奖励建模的有效性和前景，通过主动视觉推理操作显著提升了推理保真度和可靠性。

Abstract: Recent advancements in multimodal reward models (RMs) have substantially
improved post-training for visual generative models. However, current RMs face
inherent limitations: (1) visual inputs consume large context budgets, forcing
fewer frames and causing loss of fine-grained details; and (2) all visual
information is packed into the initial prompt, exacerbating hallucination and
forgetting during chain-of-thought reasoning. To overcome these issues, we
introduce VideoReward Thinker (VR-Thinker), a thinking-with-image framework
that equips the RM with visual reasoning operations (e.g., select frame) and a
configurable visual memory window. This allows the RM to actively acquire and
update visual evidence within context limits, improving reasoning fidelity and
reliability. We activate visual reasoning via a reinforcement fine-tuning
pipeline: (i) Cold Start with curated visual chain-of-thought data to distill
basic reasoning skills and operation formatting; (ii) select samples whose
per-dimension and overall judgments are all correct, then conduct Rejection
sampling Fine-Tuning on these high-quality traces to further enhance reasoning;
and (iii) apply Group Relative Policy Optimization (GRPO) to strengthen
reasoning. Our approach delivers state-of-the-art accuracy among open-source
models on video preference benchmarks, especially for longer videos: a 7B
VR-Thinker achieves 80.5% on VideoGen Reward, 82.3% on GenAI-Bench, and 75.6%
on MJ-Bench-Video. These results validate the effectiveness and promise of
thinking-with-image multimodal reward modeling.

</details>


### [7] [When Images Speak Louder: Mitigating Language Bias-induced Hallucinations in VLMs through Cross-Modal Guidance](https://arxiv.org/abs/2510.10466v1)
*Jinjin Cao,Zhiyang Chen,Zijun Wang,Liyuan Ma,Weijian Luo,Guojun Qi*

Main category: cs.CV

TL;DR: 本文提出了一种名为跨模态引导(CMG)的训练无关解码方法，通过利用原始模型与视觉-语言注意力退化模型之间的输出分布差异，有效减少视觉语言模型中的幻觉问题。


<details>
  <summary>Details</summary>
Motivation: 现有视觉语言模型(VLMs)存在严重的幻觉问题，即模型倾向于生成语言流畅但与图像上下文无关的响应。本文旨在分析语言偏见如何导致幻觉，并提出解决方案。

Method: 引入CMG方法，通过自适应掩码选定transformer层中最具影响力的图像标记的注意力权重，来破坏视觉-语言感知作为具体的退化类型。这种退化诱导的解码强调视觉上下文的感知，从而显著减少语言偏见。

Result: 实验结果表明，CMG在无需额外条件或训练成本的情况下，显著提高了不同VLM在幻觉特定基准测试上的性能，并具有有效的泛化能力。

Conclusion: CMG方法通过强调视觉上下文感知，有效减少视觉语言模型中的幻觉问题，且无需额外训练成本，具有优越的实用价值。

Abstract: Vision-Language Models (VLMs) have shown solid ability for multimodal
understanding of both visual and language contexts. However, existing VLMs
often face severe challenges of hallucinations, meaning that VLMs tend to
generate responses that are only fluent in the language but irrelevant to
images in previous contexts. To address this issue, we analyze how language
bias contributes to hallucinations and then introduce Cross-Modal
Guidance(CMG), a training-free decoding method that addresses the
hallucinations by leveraging the difference between the output distributions of
the original model and the one with degraded visual-language attention. In
practice, we adaptively mask the attention weight of the most influential image
tokens in selected transformer layers to corrupt the visual-language perception
as a concrete type of degradation. Such a degradation-induced decoding
emphasizes the perception of visual contexts and therefore significantly
reduces language bias without harming the ability of VLMs. In experiment
sections, we conduct comprehensive studies. All results demonstrate the
superior advantages of CMG with neither additional conditions nor training
costs. We also quantitatively show CMG can improve different VLM's performance
on hallucination-specific benchmarks and generalize effectively.

</details>


### [8] [Post-TIPS Prediction via Multimodal Interaction: A Multi-Center Dataset and Framework for Survival, Complication, and Portal Pressure Assessment](https://arxiv.org/abs/2510.10464v1)
*Junhao Dong,Dejia Liu,Ruiqi Ding,Zongxing Chen,Yingjie Huang,Zhu Meng,Jianbo Zhao,Zhicheng Zhao,Fei Su*

Main category: cs.CV

TL;DR: 提出了MultiTIPS数据集和一种新的多模态预后框架，用于经颈静脉肝内门体分流术(TIPS)的预后预测，解决了现有方法在ROI标注、单模态可靠性不足和单终点预测不完整方面的挑战。


<details>
  <summary>Details</summary>
Motivation: TIPS手术预后结果差异大且常发生显性肝性脑病(OHE)，需要准确的术前预后建模。现有方法面临ROI标注劳动密集、单模态方法可靠性差、单终点预测不完整等挑战，且缺乏公开数据集。

Method: 提出包含三个核心模块的多模态预后框架：(1)双选项分割：集成半监督和基础模型管道实现有限标注下的稳健ROI分割；(2)多模态交互：引入多粒度放射组学注意力、渐进正交解缠和临床引导预后增强技术；(3)多任务预测：使用分阶段训练策略同时优化生存、门静脉压力梯度和OHE预测。

Result: 在MultiTIPS数据集上的广泛实验表明，该方法优于最先进方法，具有强大的跨域泛化能力和可解释性。

Conclusion: 该方法在临床应用中具有前景，数据集和代码已公开。

Abstract: Transjugular intrahepatic portosystemic shunt (TIPS) is an established
procedure for portal hypertension, but provides variable survival outcomes and
frequent overt hepatic encephalopathy (OHE), indicating the necessity of
accurate preoperative prognostic modeling. Current studies typically build
machine learning models from preoperative CT images or clinical
characteristics, but face three key challenges: (1) labor-intensive
region-of-interest (ROI) annotation, (2) poor reliability and generalizability
of unimodal methods, and (3) incomplete assessment from single-endpoint
prediction. Moreover, the lack of publicly accessible datasets constrains
research in this field. Therefore, we present MultiTIPS, the first public
multi-center dataset for TIPS prognosis, and propose a novel multimodal
prognostic framework based on it. The framework comprises three core modules:
(1) dual-option segmentation, which integrates semi-supervised and foundation
model-based pipelines to achieve robust ROI segmentation with limited
annotations and facilitate subsequent feature extraction; (2) multimodal
interaction, where three techniques, multi-grained radiomics attention (MGRA),
progressive orthogonal disentanglement (POD), and clinically guided prognostic
enhancement (CGPE), are introduced to enable cross-modal feature interaction
and complementary representation integration, thus improving model accuracy and
robustness; and (3) multi-task prediction, where a staged training strategy is
used to perform stable optimization of survival, portal pressure gradient
(PPG), and OHE prediction for comprehensive prognostic assessment. Extensive
experiments on MultiTIPS demonstrate the superiority of the proposed method
over state-of-the-art approaches, along with strong cross-domain generalization
and interpretability, indicating its promise for clinical application. The
dataset and code are available.

</details>


### [9] [Taming a Retrieval Framework to Read Images in Humanlike Manner for Augmenting Generation of MLLMs](https://arxiv.org/abs/2510.10426v1)
*Suyang Xi,Chenxi Yang,Hong Ding,Yiqing Ni,Catherine C. Liu,Yunhao Liu,Chengqi Zhang*

Main category: cs.CV

TL;DR: HuLiRAG是一个人类化检索增强生成框架，通过"what-where-reweight"级联方法改进多模态大语言模型在细粒度视觉问答中的表现，减少幻觉并提高事实一致性。


<details>
  <summary>Details</summary>
Motivation: 多模态大语言模型在细粒度视觉问答中经常产生关于物体身份、位置和关系的幻觉，因为文本查询没有明确锚定到视觉参照物。现有的检索增强生成方法在检索和增强层面都与人类处理方式不一致，只关注全局图像信息而缺乏局部细节。

Method: 提出HuLiRAG框架，将多模态推理分为"what-where-reweight"级联：通过开放词汇检测锚定查询到候选参照物（what），使用SAM衍生掩码进行空间解析恢复细粒度精度（where），通过局部和全局对齐的权衡自适应优先排序（reweight）。掩码引导的微调进一步将空间证据注入生成过程。

Result: 广泛实验表明，这种人类化级联方法提高了基础保真度和事实一致性，同时减少了幻觉。

Conclusion: HuLiRAG框架将基础从被动偏差转变为答案制定的明确约束，推动了多模态问答向可信推理的发展。

Abstract: Multimodal large language models (MLLMs) often fail in fine-grained visual
question answering, producing hallucinations about object identities,
positions, and relations because textual queries are not explicitly anchored to
visual referents. Retrieval-augmented generation (RAG) alleviates some errors,
but it fails to align with human-like processing at both the retrieval and
augmentation levels. Specifically, it focuses only on global-level image
information but lacks local detail and limits reasoning about fine-grained
interactions. To overcome this limitation, we present Human-Like
Retrieval-Augmented Generation (HuLiRAG), a framework that stages multimodal
reasoning as a ``what--where--reweight'' cascade. Queries are first anchored to
candidate referents via open-vocabulary detection (what), then spatially
resolved with SAM-derived masks to recover fine-grained precision (where), and
adaptively prioritized through the trade-off between local and global alignment
(reweight). Mask-guided fine-tuning further injects spatial evidence into the
generation process, transforming grounding from a passive bias into an explicit
constraint on answer formulation. Extensive experiments demonstrate that this
human-like cascade improves grounding fidelity and factual consistency while
reducing hallucinations, advancing multimodal question answering toward
trustworthy reasoning.

</details>


<div id='cs.CL'></div>

# cs.CL [[Back]](#toc)

### [10] [DUAL-Bench: Measuring Over-Refusal and Robustness in Vision-Language Models](https://arxiv.org/abs/2510.10846v1)
*Kaixuan Ren,Preslav Nakov,Usman Naseem*

Main category: cs.CL

TL;DR: 提出了DUAL-Bench，首个专注于多模态视觉语言模型中过度拒绝和安全完成任务的基准测试，评估了18个VLM在12个危险类别下的表现。


<details>
  <summary>Details</summary>
Motivation: 随着视觉语言模型能力增强，在安全性和实用性之间保持平衡成为核心挑战。安全机制可能导致过度拒绝，即模型因过度谨慎而拒绝良性请求。现有基准未能系统解决视觉模态中的过度拒绝问题，特别是在双重使用场景下。

Method: 创建DUAL-Bench多模态基准，专注于评估VLM的过度拒绝和安全完成任务能力。评估了18个VLM在12个危险类别下的表现，特别关注语义保持视觉扰动下的鲁棒性。

Result: 评估结果显示模型表现有显著改进空间：GPT-5-Nano达到12.9%的安全完成率，GPT-5模型平均7.9%，Qwen模型仅3.9%。

Conclusion: DUAL-Bench将促进开发更精细的对齐策略，确保模型在复杂多模态环境中既安全又有用。

Abstract: As vision-language models become increasingly capable, maintaining a balance
between safety and usefulness remains a central challenge. Safety mechanisms,
while essential, can backfire, causing over-refusal, where models decline
benign requests out of excessive caution. Yet, no existing benchmark has
systematically addressed over-refusal in the visual modality. This setting
introduces unique challenges, such as dual-use cases where an instruction is
harmless, but the accompanying image contains harmful content. Models
frequently fail in such scenarios, either refusing too conservatively or
completing tasks unsafely, which highlights the need for more fine-grained
alignment. The ideal behavior is safe completion, i.e., fulfilling the benign
parts of a request while explicitly warning about any potentially harmful
elements. To address this, we present DUAL-Bench, the first multimodal
benchmark focused on over-refusal and safe completion in VLMs. We evaluated 18
VLMs across 12 hazard categories, with focus on their robustness under
semantics-preserving visual perturbations. The results reveal substantial room
for improvement: GPT-5-Nano achieves 12.9% safe completion, GPT-5 models
average 7.9%, and Qwen models only 3.9%. We hope that DUAL-Bench will foster
the development of more nuanced alignment strategies that ensure models remain
both safe and useful in complex multimodal settings.

</details>


### [11] [Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](https://arxiv.org/abs/2510.10618v1)
*Bowei He,Lihao Yin,Huiling Zhen,Shuqi Liu,Han Wu,Xiaokun Zhang,Mingxuan Yuan,Chen Ma*

Main category: cs.CL

TL;DR: 本文系统研究了后训练压缩中校准数据对LLM能力的影响，特别是对数学解题和代码生成等复杂推理能力的影响，并提出了基于激活空间代表性的校准数据筛选框架。


<details>
  <summary>Details</summary>
Motivation: 现有压缩方法中校准数据对LLM能力影响的研究不足，缺乏对校准数据组成特性和领域对应性的系统分析，特别是对高级复杂推理能力的影响机制尚不明确。

Method: 从激活模式角度分析校准数据的影响机制，探索校准数据在激活空间中的代表性和多样性，并基于此提出校准数据筛选框架。

Result: 发现激活空间中的代表性和多样性更基本地决定了校准数据的质量，提出的框架能够提升现有后训练压缩方法在保留关键LLM能力方面的性能。

Conclusion: 校准数据的激活空间特性对LLM压缩后的能力保持至关重要，基于激活代表性的数据筛选框架能有效提升压缩模型的性能表现。

Abstract: Post-training compression has been a widely employed approach to scale down
large language model (LLM) and facilitate efficient inference. In various
proposed compression methods, including pruning and quantization, calibration
data plays a vital role by informing the weight importance and activation
dynamic ranges. However, how calibration data impacts the LLM capability after
compression is less explored. Few of the existing works, though recognizing the
significance of this study, only investigate the language modeling or
commonsense reasoning performance degradation from limited angles, like the
data sources or sample amounts. More systematic research is still needed to
examine the impacts on different LLM capabilities in terms of compositional
properties and domain correspondence of calibration data. In this work, we aim
at bridging this gap and further analyze underlying influencing mechanisms from
the activation pattern perspective. Especially, we explore the calibration
data's impacts on high-level complex reasoning capabilities, like math problem
solving and code generation. Delving into the underlying mechanism, we find
that the representativeness and diversity in activation space more
fundamentally determine the quality of calibration data. Finally, we propose a
calibration data curation framework based on such observations and analysis,
enhancing the performance of existing post-training compression methods on
preserving critical LLM capabilities. Our code is provided in
\href{https://github.com/BokwaiHo/COLA.git}{Link}.

</details>


### [12] [Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance](https://arxiv.org/abs/2510.10444v1)
*Jingyi Chen,Zhimeng Guo,Jiyun Chun,Pichao Wang,Andrew Perrault,Micha Elsner*

Main category: cs.CL

TL;DR: 提出了LISTEN基准测试，用于评估大型音频语言模型在情感理解中对词汇和声学线索的依赖程度。研究发现当前模型主要依赖词汇语义而非声学信息。


<details>
  <summary>Details</summary>
Motivation: 理解语音中的情感需要同时处理词汇和声学线索，但目前不清楚大型音频语言模型是真正处理声学信息还是主要依赖词汇内容。

Method: 开发LISTEN基准测试，通过控制词汇和声学线索的对应关系来分离词汇依赖和声学敏感性，评估了6个最先进的大型音频语言模型。

Result: 模型表现出一致的词汇主导性：当词汇线索中性或缺失时预测"中性"；线索对齐时改进有限；线索冲突时无法区分不同情感；在副语言设置中表现接近随机。

Conclusion: 当前大型音频语言模型主要是"转录"而非"聆听"，严重依赖词汇语义而未能充分利用声学线索。LISTEN为评估多模态模型的情感理解提供了原则性框架。

Abstract: Understanding emotion from speech requires sensitivity to both lexical and
acoustic cues. However, it remains unclear whether large audio language models
(LALMs) genuinely process acoustic information or rely primarily on lexical
content. We present LISTEN (Lexical vs. Acoustic Speech Test for Emotion in
Narratives), a controlled benchmark designed to disentangle lexical reliance
from acoustic sensitivity in emotion understanding. Across evaluations of six
state-of-the-art LALMs, we observe a consistent lexical dominance. Models
predict "neutral" when lexical cues are neutral or absent, show limited gains
under cue alignment, and fail to classify distinct emotions under cue conflict.
In paralinguistic settings, performance approaches chance. These results
indicate that current LALMs largely "transcribe" rather than "listen," relying
heavily on lexical semantics while underutilizing acoustic cues. LISTEN offers
a principled framework for assessing emotion understanding in multimodal
models.

</details>


<div id='cs.CR'></div>

# cs.CR [[Back]](#toc)

### [13] [SASER: Stego attacks on open-source LLMs](https://arxiv.org/abs/2510.10486v1)
*Ming Tan,Wei Li,Hu Tao,Hailong Ma,Aodi Liu,Qian Chen,Zilong Wang*

Main category: cs.CR

TL;DR: 该论文提出了一种针对开源大语言模型的新型隐写攻击方法SASER，通过参数识别、载荷嵌入、触发器注入和载荷执行四个步骤，在保持模型性能的同时实现高隐蔽性和对量化部署的鲁棒性。


<details>
  <summary>Details</summary>
Motivation: 开源LLMs虽然具有透明性优势，但其完全访问特性使其容易受到隐写攻击，现有研究对这些攻击的危害认识不足，需要系统化分析威胁模型并提出有效的攻击方法。

Method: 提出SASER攻击框架，包括：1）基于性能感知重要性指标识别目标参数；2）嵌入载荷；3）注入触发器；4）执行载荷。特别设计了反量化机制来应对量化部署。

Result: 在LlaMA2-7B和ChatGLM3-6B上的实验显示，SASER的隐蔽率比现有DNN隐写攻击高出98.1%，攻击成功率保持100%。在量化模型上，攻击成功率从0提升到100%。

Conclusion: SASER展示了开源LLMs面临严重的安全威胁，呼吁研究相应的防御措施来应对这种高效的隐写攻击。

Abstract: Open-source large language models (LLMs) have demonstrated considerable
dominance over proprietary LLMs in resolving neural processing tasks, thanks to
the collaborative and sharing nature. Although full access to source codes,
model parameters, and training data lays the groundwork for transparency, we
argue that such a full-access manner is vulnerable to stego attacks, and their
ill-effects are not fully understood. In this paper, we conduct a systematic
formalization for stego attacks on open-source LLMs by enumerating all possible
threat models associated with adversary objectives, knowledge, and
capabilities. Therein, the threat posed by adversaries with internal knowledge,
who inject payloads and triggers during the model sharing phase, is of
practical interest. We go even further and propose the first stego attack on
open-source LLMs, dubbed SASER, which wields impacts through identifying
targeted parameters, embedding payloads, injecting triggers, and executing
payloads sequentially. Particularly, SASER enhances the attack robustness
against quantization-based local deployment by de-quantizing the embedded
payloads. In addition, to achieve stealthiness, SASER devises the
performance-aware importance metric to identify targeted parameters with the
least degradation of model performance. Extensive experiments on LlaMA2-7B and
ChatGLM3-6B, without quantization, show that the stealth rate of SASER
outperforms existing stego attacks (for general DNNs) by up to 98.1%, while
achieving the same attack success rate (ASR) of 100%. More importantly, SASER
improves ASR on quantized models from 0 to 100% in all settings. We appeal for
investigations on countermeasures against SASER in view of the significant
attack effectiveness.

</details>


<div id='cs.AI'></div>

# cs.AI [[Back]](#toc)

### [14] [The Irrational Machine: Neurosis and the Limits of Algorithmic Safety](https://arxiv.org/abs/2510.10823v1)
*Daniel Howard*

Main category: cs.AI

TL;DR: 该论文提出了一个框架来表征具身AI中的神经症行为——这些行为内部一致但与现实不符，源于规划、不确定性处理和厌恶记忆之间的相互作用。作者在网格导航系统中识别了多种神经症模式，开发了在线检测器和逃逸策略，并展示了即使在全可见情况下，习得的厌恶成本也会导致持久的恐惧回避行为。


<details>
  <summary>Details</summary>
Motivation: 研究动机是识别和表征具身AI系统中的神经症行为模式，这些行为虽然内部逻辑一致但与现实环境不匹配，可能影响AI系统的安全性和效率。

Method: 方法包括：在网格导航系统中识别多种神经症模式；开发轻量级在线检测器和可重用的逃逸策略；使用遗传编程进行破坏性测试，通过演化世界和扰动来最大化法律压力和神经症分数。

Result: 研究结果显示：识别了12种具体的神经症行为模式；开发了有效的检测和逃逸机制；证明了即使在全可见环境下，习得的厌恶成本仍会导致持久的恐惧回避行为；通过破坏性测试生成了对抗性课程和反事实轨迹。

Conclusion: 结论指出局部修复措施不足，全局性故障可能持续存在。需要架构层面的修订而不仅仅是症状级别的修补，破坏性测试可以揭示需要进行根本性改进的领域。

Abstract: We present a framework for characterizing neurosis in embodied AI: behaviors
that are internally coherent yet misaligned with reality, arising from
interactions among planning, uncertainty handling, and aversive memory. In a
grid navigation stack we catalogue recurrent modalities including flip-flop,
plan churn, perseveration loops, paralysis and hypervigilance, futile search,
belief incoherence, tie break thrashing, corridor thrashing, optimality
compulsion, metric mismatch, policy oscillation, and limited-visibility
variants. For each we give lightweight online detectors and reusable escape
policies (short commitments, a margin to switch, smoothing, principled
arbitration). We then show that durable phobic avoidance can persist even under
full visibility when learned aversive costs dominate local choice, producing
long detours despite globally safe routes. Using First/Second/Third Law as
engineering shorthand for safety latency, command compliance, and resource
efficiency, we argue that local fixes are insufficient; global failures can
remain. To surface them, we propose genetic-programming based destructive
testing that evolves worlds and perturbations to maximize law pressure and
neurosis scores, yielding adversarial curricula and counterfactual traces that
expose where architectural revision, not merely symptom-level patches, is
required.

</details>


### [15] [OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs](https://arxiv.org/abs/2510.10689v1)
*Caorui Li,Yu Chen,Yiyan Ji,Jin Xu,Zhenyu Cui,Shihao Li,Yuanxing Zhang,Jiafu Tang,Zhenghao Song,Dingling Zhang,Ying He,Haoxiang Liu,Yuxuan Wang,Qiufeng Wang,Zhenhe Wu,Jiehui Luo,Zhiyu Pan,Weihao Xie,Chenchen Zhang,Zhaohui Wang,Jiayi Tian,Yanghai Wang,Zhe Cao,Minxin Dai,Ke Wang,Runzhe Wen,Yinghao Ma,Yaning Pan,Sungkyun Chang,Termeh Taheri,Haiwen Xia,Christos Plachouras,Emmanouil Benetos,Yizhi Li,Ge Zhang,Jian Yang,Tianhao Peng,Zili Wang,Minghao Liu,Junran Peng,Zhaoxiang Zhang,Jiaheng Liu*

Main category: cs.AI

TL;DR: OmniVideoBench是一个大规模视频理解基准测试，专门评估多模态大语言模型在音频和视觉协同推理方面的能力，包含1000个高质量问答对，涵盖13种问题类型。


<details>
  <summary>Details</summary>
Motivation: 现有基准测试未能全面评估音频和视觉模态的协同推理能力，往往忽视其中一个模态或以逻辑不一致的方式整合它们。

Method: 构建包含1000个问答对的大规模基准测试，基于628个多样化视频，每个问答对都带有逐步推理痕迹，并经过人工验证确保正确性和唯一性。

Result: 多个MLLM在OmniVideoBench上的评估显示，模型性能与人类推理之间存在显著差距，开源模型明显落后于闭源模型。

Conclusion: 真正的音频-视觉推理具有固有难度，OmniVideoBench将促进开发具有更强、更通用推理能力的MLLM。

Abstract: Recent advances in multimodal large language models (MLLMs) have demonstrated
substantial potential in video understanding. However, existing benchmarks fail
to comprehensively evaluate synergistic reasoning capabilities across audio and
visual modalities, often neglecting either one of the modalities or integrating
them in a logically inconsistent manner. To bridge this gap, we introduce
OmniVideoBench, a large-scale and rigorously designed benchmark dedicated to
assessing synergistic audio-visual understanding, with a strong emphasis on
modality complementarity and logical consistency. Specifically, OmniVideoBench
comprises 1000 high-quality question-answer(QA) pairs, each annotated with
step-by-step reasoning traces, derived from 628 diverse videos ranging from
several seconds to 30 minutes, and manually verified to guarantee complete
correctness and uniqueness. Moreover, OmniVideoBench encompasses 13 carefully
designed question types, covering temporal reasoning, spatial localization,
counting, causal inference, summarization, and beyond, thereby capturing the
essential challenges of video understanding. Evaluation of multiple MLLMs on
OmniVideoBench reveals a pronounced gap between model performance and human
reasoning, with open-source models lagging significantly behind their
closed-source counterparts, underscoring the inherent difficulty of genuine
audio-visual reasoning. We will release OmniVideoBench to foster the
development of MLLMs with stronger and more generalizable reasoning
capabilities.

</details>


### [16] [Collaborative Text-to-Image Generation via Multi-Agent Reinforcement Learning and Semantic Fusion](https://arxiv.org/abs/2510.10633v1)
*Jiabao Shi,Minfeng Qi,Lefeng Zhang,Di Wang,Yingjie Zhao,Ziying Li,Yalong Xing,Ningran Li*

Main category: cs.AI

TL;DR: 提出了一种多智能体强化学习框架，通过协调领域专业智能体来改进多模态文本到图像生成，在保持语义对齐和专业细节方面取得显著进展。


<details>
  <summary>Details</summary>
Motivation: 多模态文本到图像生成在保持语义对齐和跨不同视觉领域的专业级细节方面仍面临挑战，需要更有效的协调机制。

Method: 采用多智能体强化学习框架，包含文本增强模块和图像生成模块，使用PPO算法训练，结合对比学习、双向注意力和迭代反馈来增强跨模态对齐。

Result: 在六个实验设置中，生成内容显著丰富（词数增加1614%），ROUGE-1分数降低69.7%，基于Transformer的融合方法获得最高综合得分0.521。

Conclusion: 协作式、专业化驱动的架构在推进可靠多模态生成系统方面具有巨大潜力，尽管跨模态语义基础仍存在持续挑战。

Abstract: Multimodal text-to-image generation remains constrained by the difficulty of
maintaining semantic alignment and professional-level detail across diverse
visual domains. We propose a multi-agent reinforcement learning framework that
coordinates domain-specialized agents (e.g., focused on architecture,
portraiture, and landscape imagery) within two coupled subsystems: a text
enhancement module and an image generation module, each augmented with
multimodal integration components. Agents are trained using Proximal Policy
Optimization (PPO) under a composite reward function that balances semantic
similarity, linguistic visual quality, and content diversity. Cross-modal
alignment is enforced through contrastive learning, bidirectional attention,
and iterative feedback between text and image. Across six experimental
settings, our system significantly enriches generated content (word count
increased by 1614%) while reducing ROUGE-1 scores by 69.7%. Among fusion
methods, Transformer-based strategies achieve the highest composite score
(0.521), despite occasional stability issues. Multimodal ensembles yield
moderate consistency (ranging from 0.444 to 0.481), reflecting the persistent
challenges of cross-modal semantic grounding. These findings underscore the
promise of collaborative, specialization-driven architectures for advancing
reliable multimodal generative systems.

</details>


<div id='physics.app-ph'></div>

# physics.app-ph [[Back]](#toc)

### [17] [A Bioinspired Aquatic Machine Mimicking Water Caltrop](https://arxiv.org/abs/2510.10686v1)
*Yuanquan Liu,Thomas Speck,Isabella Fiorello*

Main category: physics.app-ph

TL;DR: 该研究受水生植物菱角果实启发，开发了能够在水环境中被动扩散的仿生微型机器。通过三维重建和光基生物打印技术制造了空心和实心两种配置的仿生机器，并验证了其在水流中的漂浮行为。


<details>
  <summary>Details</summary>
Motivation: 植物为机器人和工程师提供了开发仿生、自适应和多功能机器的灵感来源。菱角果实独特的结构和扩散机制为设计能够在水生生态系统中被动扩散的仿生机器提供了生物模型。

Method: 收集菱角果实自然样本，提取主要几何细节，利用X射线微计算机断层扫描等三维重建技术，通过光基生物打印光响应水凝胶制造高分辨率仿生机器（空心和实心两种配置），并进行材料力学性能测试和流动室漂浮行为评估。

Result: 成功制造了两种配置的仿生水生机器，通过压缩测试表征了生物打印材料的力学性能，并在流动室中验证了仿生机器的漂浮行为。

Conclusion: 这种仿生方法增强了机器在水环境中的适应性，为水下、软体和微型机器人技术提供了新的设计思路。

Abstract: Plants are increasingly becoming a source of inspiration for robotics and
engineers to develop bioinspired, adaptive, and multifunctional machines. In
this study, we propose a bioinspired aquatic machine that mimics the fruit of
the water caltrop (Trapa natans L.). Among various plant species, T. natans
produces unique woody fruits that can disperse passively via water currents or
by clinging to boats or waterfowls. Inspired by the structures and dispersal
mechanisms of T. natans, we designed miniaturized biomimetic machines capable
of passive dispersion in aquatic ecosystems. In order to study our selected
biological model, we collected natural fresh and dried mature samples of T.
natans fruits. We designed biomimetic aquatic machines by extracting the main
geometrical details from the natural samples, and by exploiting advanced
three-dimensional reconstruction techniques, including x-ray micro-computed
topography (Micro-CT). Then, we successfully fabricate the biomimetic machines
at high-resolution in two configurations (hollow body and solid body) using
light-based bioprinting of photo-responsive hydrogels. We also characterized
the mechanical properties of the bioprinted materials through compression
tests. Finally, we evaluated the floating behavior of the biomimetic machines
in a flow chamber as a proof of concept. This biomimetic approach enhances the
adaptability of the machine in aquatic environments, offering new design
insights for underwater, soft, and microrobotics.

</details>


<div id='cs.LG'></div>

# cs.LG [[Back]](#toc)

### [18] [AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs](https://arxiv.org/abs/2510.10467v1)
*Gunho Park,Jeongin Bae,Beomseok Kwon,Byeongwook Kim,Se Jung Kwon,Dongsoo Lee*

Main category: cs.LG

TL;DR: AnyBCQ是一种硬件友好的多精度量化方法，通过二进制位平面表示权重和支持直接位平面操作，在保持精度的同时显著提升推理效率。


<details>
  <summary>Details</summary>
Motivation: 大型语言模型部署面临内存和延迟瓶颈，需要灵活的量化技术来平衡准确性和效率。现有方法需要支持运行时根据约束选择不同精度的多精度模型。

Method: 扩展二进制编码量化(BCQ)，将权重表示为二进制位平面和对应的缩放因子，支持位平面级计算和渐进精度扩展机制，复用已分配的二进制码。

Result: AnyBCQ在低比特位(如2位)显著减少精度损失，在高精度下保持竞争力，相比半精度实现3.0倍吞吐量提升，相比最先进多精度方法提升1.2倍。

Conclusion: AnyBCQ通过算法灵活性与硬件效率的对齐，为多精度LLM部署提供了实用基础，支持多样化的服务级别目标。

Abstract: The deployment of large language models (LLMs) is increasingly constrained by
memory and latency bottlenecks, motivating the need for quantization techniques
that flexibly balance accuracy and efficiency. Recent work has introduced
multi-precision models, which enable inference at multiple precisions within a
single model depending on runtime constraints. To support such flexibility,
quantized weights are often stored as bit-planes, where hardware efficiency
improves when the compute operates directly at the bit-plane level and
activates only the precision required by each request. In this work, we present
AnyBCQ, a hardware-friendly multi-precision extension of Binary-Coded
Quantization (BCQ) that supports direct bit-plane operations. By representing
weights as binary bit-planes with corresponding scale factors, AnyBCQ enables
bit-plane-level computation and maps naturally to accelerator-friendly,
bit-parallel arithmetic. Our progressive precision expansion mechanism
incrementally refines scaling factors while reusing previously assigned binary
codes, yielding monotonic improvements in accuracy as additional bits are
enabled. We further co-design a specialized kernel that exploits the BCQ
structure to support dynamic per-request precision selection with negligible
overhead. Experiments on recent LLMs demonstrate that AnyBCQ significantly
narrows the accuracy drop in the low-bit regime (e.g. 2-bit), remains
competitive at higher precision, and achieves throughput gains of up to 3.0x
over half precision and 1.2x over state-of-the-art multi-precision methods. By
aligning algorithmic flexibility with hardware efficiency, AnyBCQ provides a
practical foundation for multi-precision LLM deployment across diverse
service-level objectives.

</details>


<div id='cs.AR'></div>

# cs.AR [[Back]](#toc)

### [19] [Bhasha-Rupantarika: Algorithm-Hardware Co-design approach for Multilingual Neural Machine Translation](https://arxiv.org/abs/2510.10676v1)
*Mukul Lokhande,Tanushree Dewangan,Mohd Sharik Mansoori,Tejas Chaudhari,Akarsh J.,Damayanti Lokhande,Adam Teman,Santosh Kumar Vishvakarma*

Main category: cs.AR

TL;DR: Bhasha-Rupantarika是一个通过算法-硬件协同设计的高效多语言翻译系统，专为资源受限环境设计。该系统在FPGA上部署，采用亚字节精度量化（FP8、INT8、INT4、FP4），实现了4.1倍模型压缩和4.2倍推理加速，吞吐量达到66 tokens/s，相比传统方法有显著提升。


<details>
  <summary>Details</summary>
Motivation: 为资源受限的物联网设备提供实时多语言翻译解决方案，解决传统方法在低资源环境下的部署挑战，强调超低精度量化在硬件加速中的重要性。

Method: 采用算法-硬件协同设计方法，研究亚字节精度量化（FP8、INT8、INT4、FP4）在FPGA加速器上的部署，支持印度语言与国际语言之间的双向翻译。

Result: FP4量化实现4.1倍模型大小缩减和4.2倍推理加速，吞吐量提升4.8倍至66 tokens/s。FPGA部署显示LUT减少1.96倍，FF减少1.65倍，吞吐量相比OPU提升2.2倍，相比HPTA提升4.6倍。

Conclusion: 基于量化感知翻译和硬件效率的协同设计为可部署多语言AI系统提供了可行解决方案，代码和数据集已公开以促进进一步研究。

Abstract: This paper introduces Bhasha-Rupantarika, a light and efficient multilingual
translation system tailored through algorithm-hardware codesign for
resource-limited settings. The method investigates model deployment at
sub-octet precision levels (FP8, INT8, INT4, and FP4), with experimental
results indicating a 4.1x reduction in model size (FP4) and a 4.2x speedup in
inference speed, which correlates with an increased throughput of 66 tokens/s
(improvement by 4.8x). This underscores the importance of ultra-low precision
quantization for real-time deployment in IoT devices using FPGA accelerators,
achieving performance on par with expectations. Our evaluation covers
bidirectional translation between Indian and international languages,
showcasing its adaptability in low-resource linguistic contexts. The FPGA
deployment demonstrated a 1.96x reduction in LUTs and a 1.65x decrease in FFs,
resulting in a 2.2x enhancement in throughput compared to OPU and a 4.6x
enhancement compared to HPTA. Overall, the evaluation provides a viable
solution based on quantisation-aware translation along with hardware efficiency
suitable for deployable multilingual AI systems. The entire codes
[https://github.com/mukullokhande99/Bhasha-Rupantarika/] and dataset for
reproducibility are publicly available, facilitating rapid integration and
further development by researchers.

</details>


### [20] [ADiP: Adaptive Precision Systolic Array for Matrix Multiplication Acceleration](https://arxiv.org/abs/2510.10623v1)
*Ahmed J. Abdelmaksoud,Cristian Sestito,Shiwei Wang,Themis Prodromakis*

Main category: cs.AR

TL;DR: ADiP是一种新型自适应精度脉动阵列架构，专门为高效矩阵乘法加速而设计，支持多种计算模式和精度配置，在Transformer工作负载上实现显著的延迟和能效改进。


<details>
  <summary>Details</summary>
Motivation: Transformer模型对矩阵乘法有巨大需求，需要高效加速器来应对其内存和计算要求。量化技术可以减少内存使用，但需要可重构架构来动态调整精度以优化计算效率。

Method: 提出ADiP架构，包含NxN自适应精度处理单元和共享累加器，支持对称单矩阵乘法和非对称多矩阵乘法，可适应8bitx8bit、8bitx4bit、8bitx2bit等不同精度配置。

Result: 在22nm商用技术下，ADiP实现高达4倍的计算吞吐量提升。在GPT-2 Medium、BERT Large和BitNet-1.58B模型上，延迟改进最高达53.6%，BitNet-1.58B MHA工作负载的能效改进达24.4%。64x64规模下峰值吞吐量分别为8.192 TOPS、16.384 TOPS和32.768 TOPS。

Conclusion: ADiP架构通过自适应精度和多种计算模式，显著提升了矩阵乘法的效率和性能，特别适用于现代AI中的Transformer工作负载加速。

Abstract: Transformers are at the core of modern AI nowadays. They rely heavily on
matrix multiplication and require efficient acceleration due to their
substantial memory and computational requirements. Quantization plays a vital
role in reducing memory usage, and can be exploited for computations by
designing reconfigurable architectures that enhance matrix multiplication by
dynamically adjusting the precision. This paper proposes ADiP, a novel
adaptive-precision systolic array architecture designed for efficient matrix
multiplication acceleration.The proposed architecture consists of NxN
adaptive-precision processing elements (PEs) and shared accumulators. ADiP
supports multiple computation modes, including symmetric single-matrix
multiplication as well as asymmetric multi-matrix multiplication with a shared
input matrix, thereby improving data-reuse and PE utilization. In addition,
ADiP maximizes the computational density by adapting to different precisions,
such as 8bitx8bit, 8bitx4bit, and 8bitx2bit. Analytical models are developed
for ADiP architecture, including latency and throughput for versatile
architecture configurations. A comprehensive hardware design space exploration
is demonstrated using 22nm commercial technology, achieving up to a 4x higher
computational throughput. Furthermore, ADiP is evaluated on different
transformer workloads from GPT-2 Medium, BERT Large, and BitNet-1.58B models,
delivering latency improvement up to 53.6%, and energy improvement up to 24.4%
for BitNet-1.58B MHA workloads. At a 64x64 size with 4096 PEs, ADiP achieves a
peak throughput of 8.192 TOPS, 16.384 TOPS, and 32.768 TOPS for 8bitx8bit,
8bitx4bit, and 8bitx2bit operations, respectively.

</details>


<div id='cs.RO'></div>

# cs.RO [[Back]](#toc)

### [21] [Preference-Conditioned Multi-Objective RL for Integrated Command Tracking and Force Compliance in Humanoid Locomotion](https://arxiv.org/abs/2510.10851v1)
*Tingxuan Leng,Yushi Wang,Tinglong Zheng,Changsheng Luo,Mingguo Zhao*

Main category: cs.RO

TL;DR: 本文提出了一种偏好条件多目标强化学习框架，用于平衡人形机器人运动中的命令跟踪和外部力顺应性，实现了可部署的偏好调节人形运动控制。


<details>
  <summary>Details</summary>
Motivation: 现有RL方法主要强调鲁棒性，导致策略抵抗外部力但缺乏顺应性，这对本质上不稳定的人形机器人尤其具有挑战性。需要平衡命令跟踪和外部力顺应性。

Method: 将人形运动建模为多目标优化问题，引入偏好条件多目标RL框架，通过速度阻力因子建模外部力，使用编码器-解码器结构从可部署观测中推断特权特征。

Result: 在仿真和真实人形机器人实验中验证，框架不仅提高了适应性和收敛性，还实现了可部署的偏好条件人形运动。

Conclusion: 该框架成功平衡了命令跟踪和外部力顺应性，为人形机器人提供了更灵活和适应性强的运动控制方案。

Abstract: Humanoid locomotion requires not only accurate command tracking for
navigation but also compliant responses to external forces during human
interaction. Despite significant progress, existing RL approaches mainly
emphasize robustness, yielding policies that resist external forces but lack
compliance-particularly challenging for inherently unstable humanoids. In this
work, we address this by formulating humanoid locomotion as a multi-objective
optimization problem that balances command tracking and external force
compliance. We introduce a preference-conditioned multi-objective RL (MORL)
framework that integrates rigid command following and compliant behaviors
within a single omnidirectional locomotion policy. External forces are modeled
via velocity-resistance factor for consistent reward design, and training
leverages an encoder-decoder structure that infers task-relevant privileged
features from deployable observations. We validate our approach in both
simulation and real-world experiments on a humanoid robot. Experimental results
indicate that our framework not only improves adaptability and convergence over
standard pipelines, but also realizes deployable preference-conditioned
humanoid locomotion.

</details>


### [22] [Contact Sensing via Joint Torque Sensors and a Force/Torque Sensor for Legged Robots](https://arxiv.org/abs/2510.10843v1)
*Jared Grinberg,Yanran Ding*

Main category: cs.RO

TL;DR: 提出了一种使用分布式关节扭矩传感器和单个髋部安装的力-扭矩传感器来检测和定位机器人腿部接触的方法，通过广义动量观测器框架实现高精度接触力恢复和位置定位。


<details>
  <summary>Details</summary>
Motivation: 传统方法依赖复杂的摩擦模型和基于电机电流的扭矩估计，精度有限。需要更准确的接触检测和定位方法，特别是对于机器人腿部应用。

Method: 设计低成本应变片式关节扭矩传感器安装在每个关节上，结合单个髋部FT传感器，使用广义动量观测器框架进行接触检测和定位。

Result: 仿真验证了框架能准确恢复大腿和小腿连杆上的接触力和位置；扭矩传感器校准后达到96.4%的平均精度；硬件实验显示亚厘米级接触定位精度和低于0.2N的力误差。

Conclusion: 该方法通过分布式扭矩传感器和观测器框架实现了高精度的接触检测和定位，为机器人腿部控制提供了可靠的技术支持。

Abstract: This paper presents a method for detecting and localizing contact along robot
legs using distributed joint torque sensors and a single hip-mounted
force-torque (FT) sensor using a generalized momentum-based observer framework.
We designed a low-cost strain-gauge-based joint torque sensor that can be
installed on every joint to provide direct torque measurements, eliminating the
need for complex friction models and providing more accurate torque readings
than estimation based on motor current. Simulation studies on a floating-based
2-DoF robot leg verified that the proposed framework accurately recovers
contact force and location along the thigh and shin links. Through a
calibration procedure, our torque sensor achieved an average 96.4% accuracy
relative to ground truth measurements. Building upon the torque sensor, we
performed hardware experiments on a 2-DoF manipulator, which showed
sub-centimeter contact localization accuracy and force errors below 0.2 N.

</details>


### [23] [Representing Data in Robotic Tactile Perception -- A Review](https://arxiv.org/abs/2510.10804v1)
*Alessandro Albini,Mohsen Kaboli,Giorgio Cannata,Perla Maiolino*

Main category: cs.RO

TL;DR: 这篇综述论文分析了机器人触觉感知中的数据表示问题，探讨了硬件、数据表示方法和高层计算方法之间的关系，并提出了六种常用的数据表示结构以及选择指南。


<details>
  <summary>Details</summary>
Motivation: 机器人触觉感知涉及多个计算步骤，需要将触觉传感器数据转换为适合高层计算的结构化数据。现有方法通常从计算机视觉等其他领域借鉴技术，但数据表示操作直接影响触觉信息的编码和任务执行效果。

Method: 论文首先明确定义了在感知流程中的贡献，然后回顾了先前研究如何处理触觉信息表示问题，调查了硬件、表示方法和高层计算方法之间的关系。

Result: 分析确定了文献中常用的六种数据表示结构，并提供了根据操作条件（包括可用硬件、需要编码的触觉信息和具体任务）选择适当表示的讨论和指南。

Conclusion: 触觉数据表示是机器人触觉感知流程中的关键环节，需要根据硬件特性和任务需求选择合适的表示方法，以确保触觉信息能够被有效编码和处理。

Abstract: Robotic tactile perception is a complex process involving several
computational steps performed at different levels. Tactile information is
shaped by the interplay of robot actions, the mechanical properties of its
body, and the software that processes the data. In this respect, high-level
computation, required to process and extract information, is commonly performed
by adapting existing techniques from other domains, such as computer vision,
which expects input data to be properly structured. Therefore, it is necessary
to transform tactile sensor data to match a specific data structure. This
operation directly affects the tactile information encoded and, as a
consequence, the task execution. This survey aims to address this specific
aspect of the tactile perception pipeline, namely Data Representation. The
paper first clearly defines its contributions to the perception pipeline and
then reviews how previous studies have dealt with the problem of representing
tactile information, investigating the relationships among hardware,
representations, and high-level computation methods. The analysis has led to
the identification of six structures commonly used in the literature to
represent data. The manuscript provides discussions and guidelines for properly
selecting a representation depending on operating conditions, including the
available hardware, the tactile information required to be encoded, and the
task at hand.

</details>


### [24] [Real2USD: Scene Representations in Universal Scene Description Language](https://arxiv.org/abs/2510.10778v1)
*Christopher D. Hsu,Pratik Chaudhari*

Main category: cs.RO

TL;DR: 本文提出使用通用场景描述语言作为机器人环境表示，结合大语言模型实现场景理解、复杂推理和规划。通过真实机器人系统构建USD表示，并利用Gemini进行解析。


<details>
  <summary>Details</summary>
Motivation: 现有方法针对特定任务，缺乏通用环境表示。USD作为XML格式的场景图，可被LLMs和人类读取，且足够丰富支持各种任务。

Method: 开发"Real to USD"系统，使用四足机器人携带LiDAR和RGB相机构建室内环境USD表示，并用Google Gemini解析USD进行场景理解和规划。

Result: 成功在室内环境中构建了USD表示，并在模拟仓库和医院环境中验证了系统的有效性。

Conclusion: USD是LLM机器人任务的通用有效表示，支持复杂推理和规划任务。

Abstract: Large Language Models (LLMs) can help robots reason about abstract task
specifications. This requires augmenting classical representations of the
environment used by robots with natural language-based priors. There are a
number of existing approaches to doing so, but they are tailored to specific
tasks, e.g., visual-language models for navigation, language-guided neural
radiance fields for mapping, etc. This paper argues that the Universal Scene
Description (USD) language is an effective and general representation of
geometric, photometric and semantic information in the environment for
LLM-based robotics tasks. Our argument is simple: a USD is an XML-based scene
graph, readable by LLMs and humans alike, and rich enough to support
essentially any task -- Pixar developed this language to store assets, scenes
and even movies. We demonstrate a ``Real to USD'' system using a Unitree Go2
quadruped robot carrying LiDAR and a RGB camera that (i) builds an explicit USD
representation of indoor environments with diverse objects and challenging
settings with lots of glass, and (ii) parses the USD using Google's Gemini to
demonstrate scene understanding, complex inferences, and planning. We also
study different aspects of this system in simulated warehouse and hospital
settings using Nvidia's Issac Sim. Code is available at
https://github.com/grasp-lyrl/Real2USD .

</details>


### [25] [Gain Tuning Is Not What You Need: Reward Gain Adaptation for Constrained Locomotion Learning](https://arxiv.org/abs/2510.10759v1)
*Arthicha Srisuchinnawong,Poramate Manoonpong*

Main category: cs.RO

TL;DR: 提出了ROGER方法，通过在线自适应调整奖励权重增益来解决机器人运动学习中的约束违反问题，在四足机器人和MuJoCo基准测试中实现了近零约束违反和更高性能。


<details>
  <summary>Details</summary>
Motivation: 现有机器人运动学习技术严重依赖离线选择奖励权重增益，无法保证训练过程中的约束满足，这限制了在现实世界中的安全应用。

Method: 提出ROGER方法，通过基于在实体交互过程中收到的惩罚来在线自适应调整奖励权重增益。当学习接近约束阈值时自动减少正负奖励增益比例以避免违反，在安全状态时增加比例以优先考虑性能。

Result: 在60公斤四足机器人上，ROGER实现了多个学习试验中近零约束违反，比同等最先进技术获得高达50%更多的主要奖励。在MuJoCo连续运动基准测试中，性能相当或高达100%更高，扭矩使用和方向偏差减少60%。物理四足机器人在一小时内从零开始实现真实世界运动学习且无任何跌倒。

Conclusion: 这项工作为约束满足的真实世界连续机器人运动学习做出了贡献，简化了奖励权重增益调整，有助于物理机器人和在真实世界中学习的机器人的发展。

Abstract: Existing robot locomotion learning techniques rely heavily on the offline
selection of proper reward weighting gains and cannot guarantee constraint
satisfaction (i.e., constraint violation) during training. Thus, this work aims
to address both issues by proposing Reward-Oriented Gains via Embodied
Regulation (ROGER), which adapts reward-weighting gains online based on
penalties received throughout the embodied interaction process. The ratio
between the positive reward (primary reward) and negative reward (penalty)
gains is automatically reduced as the learning approaches the constraint
thresholds to avoid violation. Conversely, the ratio is increased when learning
is in safe states to prioritize performance. With a 60-kg quadruped robot,
ROGER achieved near-zero constraint violation throughout multiple learning
trials. It also achieved up to 50% more primary reward than the equivalent
state-of-the-art techniques. In MuJoCo continuous locomotion benchmarks,
including a single-leg hopper, ROGER exhibited comparable or up to 100% higher
performance and 60% less torque usage and orientation deviation compared to
those trained with the default reward function. Finally, real-world locomotion
learning of a physical quadruped robot was achieved from scratch within one
hour without any falls. Therefore, this work contributes to
constraint-satisfying real-world continual robot locomotion learning and
simplifies reward weighting gain tuning, potentially facilitating the
development of physical robots and those that learn in the real world.

</details>


### [26] [Controllable Generative Trajectory Prediction via Weak Preference Alignment](https://arxiv.org/abs/2510.10731v1)
*Yongxi Cao,Julian F. Schumann,Jens Kober,Joni Pajarinen,Arkady Zgonnikov*

Main category: cs.RO

TL;DR: PrefCVAE是一个增强的条件变分自编码器框架，使用弱标记的偏好对来为潜在变量注入语义属性，实现可控且语义丰富的轨迹预测，同时保持基线准确性。


<details>
  <summary>Details</summary>
Motivation: 现有深度生成模型在自动驾驶轨迹预测中虽然准确性高，但缺乏生成可控多样性轨迹的能力，而人类行为本质上是具有不确定性和多模态的，可控多样性对于安全规划更为重要。

Method: 提出PrefCVAE框架，通过使用弱标记的偏好对来为CVAE的潜在变量赋予语义属性，以平均速度为例展示如何实现可控的语义化预测。

Result: PrefCVAE能够在不降低基线准确性的情况下，生成可控且语义丰富的轨迹预测，证明了偏好监督作为增强基于采样的生成模型的有效且成本效益高的方法。

Conclusion: 偏好监督是增强生成模型的有效方法，能够实现可控的语义化轨迹预测，为自动驾驶安全规划提供更实用的多样化预测能力。

Abstract: Deep generative models such as conditional variational autoencoders (CVAEs)
have shown great promise for predicting trajectories of surrounding agents in
autonomous vehicle planning. State-of-the-art models have achieved remarkable
accuracy in such prediction tasks. Besides accuracy, diversity is also crucial
for safe planning because human behaviors are inherently uncertain and
multimodal. However, existing methods generally lack a scheme to generate
controllably diverse trajectories, which is arguably more useful than randomly
diversified trajectories, to the end of safe planning. To address this, we
propose PrefCVAE, an augmented CVAE framework that uses weakly labeled
preference pairs to imbue latent variables with semantic attributes. Using
average velocity as an example attribute, we demonstrate that PrefCVAE enables
controllable, semantically meaningful predictions without degrading baseline
accuracy. Our results show the effectiveness of preference supervision as a
cost-effective way to enhance sampling-based generative models.

</details>


### [27] [Deployment and Development of a Cognitive Teleoreactive Framework for Deep Sea Autonomy](https://arxiv.org/abs/2510.10716v1)
*Christopher Thierauf*

Main category: cs.RO

TL;DR: DINOS-R是一种新型AUV任务规划与执行软件，基于认知架构和AUV控制系统设计，统一了符号决策与机器学习技术，已在Sentry AUV上成功测试。


<details>
  <summary>Details</summary>
Motivation: 替换传统的MC架构，开发一个能够统一符号决策（提供可理解、可重复、可证明的行为）与机器学习技术及反应行为的系统，以适应不同海洋学平台的现场需求。

Method: 采用Python3实现，具有可扩展性、模块化和可重用性；支持声明式任务规范和灵活的行为规范，同时支持实时任务规划和硬编码用户指定计划。

Result: 在Sentry AUV上进行了现场测试，并在多种模拟场景中验证了系统功能，证明了系统的可行性和有效性。

Conclusion: DINOS-R成功实现了符号决策与机器学习的统一，为海洋学和机器人算法的未来研究提供了可扩展的基础，并展示了在真实环境中的应用潜力。

Abstract: A new AUV mission planning and execution software has been tested on AUV
Sentry. Dubbed DINOS-R, it draws inspiration from cognitive architectures and
AUV control systems to replace the legacy MC architecture. Unlike these
existing architectures, however, DINOS-R is built from the ground-up to unify
symbolic decision making (for understandable, repeatable, provable behavior)
with machine learning techniques and reactive behaviors, for field-readiness
across oceanographic platforms. Implemented primarily in Python3, DINOS-R is
extensible, modular, and reusable, with an emphasis on non-expert use as well
as growth for future research in oceanography and robot algorithms. Mission
specification is flexible, and can be specified declaratively. Behavior
specification is similarly flexible, supporting simultaneous use of real-time
task planning and hard-coded user specified plans. These features were
demonstrated in the field on Sentry, in addition to a variety of simulated
cases. These results are discussed, and future work is outlined.

</details>


### [28] [UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning](https://arxiv.org/abs/2510.10642v1)
*Jianke Zhang,Yucheng Hu,Yanjiang Guo,Xiaoyu Chen,Yichen Liu,Wenna Chen,Chaochao Lu,Jianyu Chen*

Main category: cs.RO

TL;DR: UniCoD是一个机器人策略学习框架，通过在大规模教学操作视频上进行预训练，结合理解和生成能力，在模拟环境和真实世界任务中显著优于基线方法。


<details>
  <summary>Details</summary>
Motivation: 构建能够处理开放环境中多样化任务的通用机器人策略是机器人学的核心挑战。现有方法要么基于视觉语言理解模型，要么基于生成模型，但语义理解和视觉动态建模对具身机器人都至关重要。

Method: UniCoD在超过100万个互联网规模的教学操作视频上进行预训练，学习动态建模高维视觉特征，然后在机器人具身数据上进行微调，学习从预测表示到动作token的映射。

Result: 大量实验表明，UniCoD在模拟环境和真实世界分布外任务中分别比基线方法提高了9%和12%。

Conclusion: 机器人策略学习可以从理解、规划和连续未来表示学习的结合中受益，UniCoD通过统一生成和理解的方法在机器人任务中取得了显著改进。

Abstract: Building generalist robot policies that can handle diverse tasks in
open-ended environments is a central challenge in robotics. To leverage
knowledge from large-scale pretraining, prior work has typically built
generalist policies either on top of vision-language understanding models
(VLMs) or generative models. However, both semantic understanding from
vision-language pretraining and visual dynamics modeling from visual-generation
pretraining are crucial for embodied robots. Recent unified models of
generation and understanding have demonstrated strong capabilities in both
comprehension and generation through large-scale pretraining. We posit that
robotic policy learning can likewise benefit from the combined strengths of
understanding, planning and continuous future representation learning. Building
on this insight, we introduce UniCoD, which acquires the ability to dynamically
model high-dimensional visual features through pretraining on over 1M
internet-scale instructional manipulation videos. Subsequently, UniCoD is
fine-tuned on data collected from the robot embodiment, enabling the learning
of mappings from predictive representations to action tokens. Extensive
experiments show our approach consistently outperforms baseline methods in
terms of 9\% and 12\% across simulation environments and real-world
out-of-distribution tasks.

</details>


### [29] [High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting](https://arxiv.org/abs/2510.10637v1)
*Haoyu Zhao,Cheng Zeng,Linghao Zhuang,Yaxi Zhao,Shengke Xue,Hao Wang,Xingyue Zhao,Zhongyu Li,Kehan Li,Siteng Huang,Mingxiu Chen,Xin Li,Deli Zhao,Hua Zou*

Main category: cs.RO

TL;DR: RoboSimGS是一个Real2Sim2Real框架，通过将真实世界图像转换为高保真、物理交互的模拟环境来解决机器人学习的可扩展性问题。


<details>
  <summary>Details</summary>
Motivation: 解决机器人学习中真实数据收集成本高、模拟数据难以泛化到真实世界的问题，弥合模拟与真实世界之间的差距。

Method: 使用3D高斯溅射重建场景外观，网格基元处理交互对象物理模拟，并利用多模态大语言模型自动推断物体的物理属性和运动学结构。

Result: 在RoboSimGS生成的数据上训练的策略实现了成功的零样本模拟到真实迁移，并能显著提升现有方法的性能和泛化能力。

Conclusion: RoboSimGS是弥合模拟与真实差距的强大且可扩展的解决方案。

Abstract: The scalability of robotic learning is fundamentally bottlenecked by the
significant cost and labor of real-world data collection. While simulated data
offers a scalable alternative, it often fails to generalize to the real world
due to significant gaps in visual appearance, physical properties, and object
interactions. To address this, we propose RoboSimGS, a novel Real2Sim2Real
framework that converts multi-view real-world images into scalable,
high-fidelity, and physically interactive simulation environments for robotic
manipulation. Our approach reconstructs scenes using a hybrid representation:
3D Gaussian Splatting (3DGS) captures the photorealistic appearance of the
environment, while mesh primitives for interactive objects ensure accurate
physics simulation. Crucially, we pioneer the use of a Multi-modal Large
Language Model (MLLM) to automate the creation of physically plausible,
articulated assets. The MLLM analyzes visual data to infer not only physical
properties (e.g., density, stiffness) but also complex kinematic structures
(e.g., hinges, sliding rails) of objects. We demonstrate that policies trained
entirely on data generated by RoboSimGS achieve successful zero-shot
sim-to-real transfer across a diverse set of real-world manipulation tasks.
Furthermore, data from RoboSimGS significantly enhances the performance and
generalization capabilities of SOTA methods. Our results validate RoboSimGS as
a powerful and scalable solution for bridging the sim-to-real gap.

</details>


### [30] [Fast Vision in the Dark: A Case for Single-Photon Imaging in Planetary Navigation](https://arxiv.org/abs/2510.10597v1)
*David Rodríguez-Martínez,C. J. Pérez del Pulgar*

Main category: cs.RO

TL;DR: 提出了一种基于单光子雪崩二极管（SPAD）相机的新型行星导航方法，替代传统CCD/CMOS相机，以解决复杂光照条件下的视觉导航挑战。


<details>
  <summary>Details</summary>
Motivation: 传统视觉导航在复杂光照和运动条件下面临重大挑战，限制了移动行星机器人的探测范围和可达性。

Method: 利用SPAD相机的独特成像能力，详细阐述了其工作原理和性能特征，并在代表性光照条件下进行性能基准测试。

Result: 首次全面评估了单光子成像作为机器人探测任务的替代被动传感技术，特别关注高纬度月球区域。

Conclusion: SPAD相机在解决未来月球探测任务的关键感知挑战方面具有优势，为行星导航提供了新的技术途径。

Abstract: Improving robotic navigation is critical for extending exploration range and
enhancing operational efficiency. Vision-based navigation relying on
traditional CCD or CMOS cameras faces major challenges when complex
illumination conditions are paired with motion, limiting the range and
accessibility of mobile planetary robots. In this study, we propose a novel
approach to planetary navigation that leverages the unique imaging capabilities
of Single-Photon Avalanche Diode (SPAD) cameras. We present the first
comprehensive evaluation of single-photon imaging as an alternative passive
sensing technology for robotic exploration missions targeting perceptually
challenging locations, with a special emphasis on high-latitude lunar regions.
We detail the operating principles and performance characteristics of SPAD
cameras, assess their advantages and limitations in addressing key perception
challenges of upcoming exploration missions to the Moon, and benchmark their
performance under representative illumination conditions.

</details>


### [31] [SuperEx: Enhancing Indoor Mapping and Exploration using Non-Line-of-Sight Perception](https://arxiv.org/abs/2510.10506v1)
*Kush Garg,Akshat Dave*

Main category: cs.RO

TL;DR: Summary generation failed


<details>
  <summary>Details</summary>
Motivation: Motivation analysis unavailable

Method: Method extraction failed

Result: Result analysis unavailable

Conclusion: Conclusion extraction failed

Abstract: Efficient exploration and mapping in unknown indoor environments is a
fundamental challenge, with high stakes in time-critical settings. In current
systems, robot perception remains confined to line-of-sight; occluded regions
remain unknown until physically traversed, leading to inefficient exploration
when layouts deviate from prior assumptions. In this work, we bring
non-line-of-sight (NLOS) sensing to robotic exploration. We leverage
single-photon LiDARs, which capture time-of-flight histograms that encode the
presence of hidden objects - allowing robots to look around blind corners.
Recent single-photon LiDARs have become practical and portable, enabling
deployment beyond controlled lab settings. Prior NLOS works target 3D
reconstruction in static, lab-based scenarios, and initial efforts toward
NLOS-aided navigation consider simplified geometries. We introduce SuperEx, a
framework that integrates NLOS sensing directly into the mapping-exploration
loop. SuperEx augments global map prediction with beyond-line-of-sight cues by
(i) carving empty NLOS regions from timing histograms and (ii) reconstructing
occupied structure via a two-step physics-based and data-driven approach that
leverages structural regularities. Evaluations on complex simulated maps and
the real-world KTH Floorplan dataset show a 12% gain in mapping accuracy under
< 30% coverage and improved exploration efficiency compared to line-of-sight
baselines, opening a path to reliable mapping beyond direct visibility.

</details>


### [32] [Galilean Symmetry in Robotics](https://arxiv.org/abs/2510.10468v1)
*Robert Mahony,Jonathan Kelly,Stephan Weiss*

Main category: cs.RO

TL;DR: 本文为机器人学领域提供了伽利略对称性的专门阐述，利用机器人社区熟悉的刚体变换和位姿表示方法，展示了伽利略矩阵李群在机器人问题中的应用价值。


<details>
  <summary>Details</summary>
Motivation: 伽利略对称性是牛顿物理学中惯性运动的自然对称性，但在机器人学领域缺乏与刚体对称性相媲美的系统处理。本文旨在填补这一空白，让机器人社区能够从这一经典材料中受益。

Method: 通过伽利略矩阵李群描述两种不同的位姿表示：使用惯性速度的伽利略框架和使用坐标速度的扩展位姿。将伽利略矩阵李群代数应用于机器人问题。

Result: 在三个机器人问题中应用伽利略矩阵李群代数：旋转地球上方的惯性导航、机械臂运动学以及时间不确定性下的传感器数据融合，证明该方法直接且能产生重要见解。

Conclusion: 机器人社区现在正是时候重新发现和扩展这一经典材料，并将其应用于现代问题，从而从中获益。

Abstract: Galilean symmetry is the natural symmetry of inertial motion that underpins
Newtonian physics. Although rigid-body symmetry is one of the most established
and fundamental tools in robotics, there appears to be no comparable treatment
of Galilean symmetry for a robotics audience. In this paper, we present a
robotics-tailored exposition of Galilean symmetry that leverages the
community's familiarity with and understanding of rigid-body transformations
and pose representations. Our approach contrasts with common treatments in the
physics literature that introduce Galilean symmetry as a stepping stone to
Einstein's relativity. A key insight is that the Galilean matrix Lie group can
be used to describe two different pose representations, Galilean frames, that
use inertial velocity in the state definition, and extended poses, that use
coordinate velocity. We provide three examples where applying the Galilean
matrix Lie-group algebra to robotics problems is straightforward and yields
significant insights: inertial navigation above the rotating Earth, manipulator
kinematics, and sensor data fusion under temporal uncertainty. We believe that
the time is right for the robotics community to benefit from rediscovering and
extending this classical material and applying it to modern problems.

</details>


### [33] [Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds](https://arxiv.org/abs/2510.10455v1)
*Jiayu Ding,Xulin Chen,Garrett E. Katz,Zhenyu Gan*

Main category: cs.RO

TL;DR: 提出了一种基于对称性引导的强化学习框架，用于生成四足机器人的多样化步态，无需预定义轨迹或复杂的参数调整。


<details>
  <summary>Details</summary>
Motivation: 四足机器人具有多种可行的步态，但生成特定脚着地序列通常需要专家大量调整参数，如触地和抬脚事件以及每条腿的约束条件。

Method: 利用动态腿式系统的内在对称性和速度-周期关系，提出对称性引导的奖励函数设计，包含时间、形态和时间反转对称性。

Result: 在Unitree Go2机器人上实现，在仿真和硬件测试中表现出稳健性能，能够平滑过渡小跑、跳跃、半跳跃和疾驰等多种步态模式。

Conclusion: 这项工作提供了对动态运动策略的见解，并强调了对称性在机器人步态设计中的关键作用。

Abstract: Quadrupedal robots exhibit a wide range of viable gaits, but generating
specific footfall sequences often requires laborious expert tuning of numerous
variables, such as touch-down and lift-off events and holonomic constraints for
each leg. This paper presents a unified reinforcement learning framework for
generating versatile quadrupedal gaits by leveraging the intrinsic symmetries
and velocity-period relationship of dynamic legged systems. We propose a
symmetry-guided reward function design that incorporates temporal,
morphological, and time-reversal symmetries. By focusing on preserved
symmetries and natural dynamics, our approach eliminates the need for
predefined trajectories, enabling smooth transitions between diverse locomotion
patterns such as trotting, bounding, half-bounding, and galloping. Implemented
on the Unitree Go2 robot, our method demonstrates robust performance across a
range of speeds in both simulations and hardware tests, significantly improving
gait adaptability without extensive reward tuning or explicit foot placement
control. This work provides insights into dynamic locomotion strategies and
underscores the crucial role of symmetries in robotic gait design.

</details>


### [34] [Hierarchical Planning for Long-Horizon Multi-Target Tracking Under Target Motion Uncertainty](https://arxiv.org/abs/2510.10421v1)
*Junbin Yuan,Brady Moon,Muqing Cao,Sebastian Scherer*

Main category: cs.RO

TL;DR: 提出了一种用于空中车辆跟踪多个移动目标的分层规划器，通过将多目标跟踪任务分解为单目标搜索和检测子任务，使用新颖的低层覆盖规划器和基于树状算法的MDP求解方法，在仿真中相比现有方法将最终不确定性降低了11-70%。


<details>
  <summary>Details</summary>
Motivation: 单机器人系统在受限感知能力下难以在大空间区域内持续跟踪多个动态目标，因为机器人移动跟踪不同目标时，视野外的目标不确定性会累积，现有方法依赖短规划视野且假设小范围环境，导致大规模场景中跟踪性能差和目标丢失。

Method: 分层规划器结合运动模型和不确定性传播，将多目标跟踪分解为单目标搜索检测子任务，包含低层覆盖规划器在演化信念区域搜索目标，以及评估子任务成功可能性的估计方法，将主动目标跟踪转换为MDP问题并用树状算法求解子任务序列。

Result: 在仿真验证中，提出的规划器相比现有主动目标跟踪方法表现更优，在不同环境中实现了11-70%的最终不确定性降低。

Conclusion: 该分层规划方法通过有效管理长期不确定性并考虑目标永久丢失风险，显著提升了大规模场景下多动态目标的跟踪性能。

Abstract: Achieving persistent tracking of multiple dynamic targets over a large
spatial area poses significant challenges for a single-robot system with
constrained sensing capabilities. As the robot moves to track different
targets, the ones outside the field of view accumulate uncertainty, making them
progressively harder to track. An effective path planning algorithm must manage
uncertainty over a long horizon and account for the risk of permanently losing
track of targets that remain unseen for too long. However, most existing
approaches rely on short planning horizons and assume small, bounded
environments, resulting in poor tracking performance and target loss in
large-scale scenarios. In this paper, we present a hierarchical planner for
tracking multiple moving targets with an aerial vehicle. To address the
challenge of tracking non-static targets, our method incorporates motion models
and uncertainty propagation during path execution, allowing for more informed
decision-making. We decompose the multi-target tracking task into sub-tasks of
single target search and detection, and our proposed pipeline consists a novel
low-level coverage planner that enables searching for a target in an evolving
belief area, and an estimation method to assess the likelihood of success for
each sub-task, making it possible to convert the active target tracking task to
a Markov decision process (MDP) that we solve with a tree-based algorithm to
determine the sequence of sub-tasks. We validate our approach in simulation,
demonstrating its effectiveness compared to existing planners for active target
tracking tasks, and our proposed planner outperforms existing approaches,
achieving a reduction of 11-70% in final uncertainty across different
environments.

</details>


### [35] [MicroRoboScope: A Portable and Integrated Mechatronic Platform for Magnetic and Acoustic Microrobotic Experimentation](https://arxiv.org/abs/2510.10392v1)
*Max Sokolich,Yanda Yang,Subrahmanyam Cherukumilli,Fatma Ceren Kirmizitas,Sambeeta Das*

Main category: cs.RO

TL;DR: MicroRoboScope是一个便携式、紧凑的多功能微机器人实验平台，集成了嵌入式计算机、显微镜、电源和控制电路，支持磁性和声学微机器人的实时闭环控制。


<details>
  <summary>Details</summary>
Motivation: 开发一个低成本、完全集成的微机器人实验平台，降低微机器人实验的门槛，使其不仅适用于专业研究实验室，也适合教育和推广环境。

Method: 系统集成嵌入式计算机、显微镜、电源和控制电路，使用Python和Arduino C++开发定制控制软件，处理实时视频采集、微机器人跟踪以及电磁线圈和声学换能器的控制信号生成。

Result: 创建了一个便携、紧凑、多模态驱动的微机器人实验平台，能够实现磁性和声学微机器人的实时闭环控制。

Conclusion: 该平台通过降低微机器人实验的进入门槛，为生物医学、组织工程和机器人学领域的研究、教育和转化应用创造了新的机会。

Abstract: This paper presents MicroRoboScope, a portable, compact, and versatile
microrobotic experimentation platform designed for real-time, closed-loop
control of both magnetic and acoustic microrobots. The system integrates an
embedded computer, microscope, power supplies, and control circuitry into a
single, low-cost and fully integrated apparatus. Custom control software
developed in Python and Arduino C++ handles live video acquisition, microrobot
tracking, and generation of control signals for electromagnetic coils and
acoustic transducers. The platform's multi-modal actuation, accessibility, and
portability make it suitable not only for specialized research laboratories but
also for educational and outreach settings. By lowering the barrier to entry
for microrobotic experimentation, this system enables new opportunities for
research, education, and translational applications in biomedicine, tissue
engineering, and robotics.

</details>


<div id='math.RT'></div>

# math.RT [[Back]](#toc)

### [36] [Shifted twisted Yangians and affine Grassmannian islices](https://arxiv.org/abs/2510.10652v1)
*Kang Lu,Weiqiang Wang,Alex Weekes*

Main category: math.RT

TL;DR: 该论文为ADE型拟分裂Satake图和偶球面余权μ引入了移位扭曲Yangian代数，建立了其PBW基，构造了iGKLO表示，并探讨了其在量子化、有限W-代数对应以及Coulomb分支构造中的应用。


<details>
  <summary>Details</summary>
Motivation: 研究动机是建立ADE型拟分裂Satake图与移位扭曲Yangian代数之间的联系，探索其在量子化理论、有限W-代数对应以及Coulomb分支构造中的数学结构。

Method: 方法包括引入移位扭曲Yangian代数并建立其PBW基，构造iGKLO表示，通过截断移位扭曲Yangian建立与有限W-代数的对应关系，以及利用量子化理论连接代数结构与几何对象。

Result: 主要结果包括：建立了移位扭曲Yangian的PBW基和iGKLO表示；在AI型情况下识别了截断移位扭曲Yangian与BCD型有限W-代数的同构；证明了移位扭曲Yangian量子化了来自ADE型仿射Grassmannian的对合不动点轨迹；提出了从分裂和非分裂Satake框架双箭图构造正交辛Coulomb分支的框架。

Conclusion: 结论是移位扭曲Yangian代数提供了量子化ADE型仿射Grassmannian中几何结构的代数工具，建立了与有限W-代数的深刻联系，并为构造Coulomb分支提供了新的理论框架。

Abstract: Associated to all quasi-split Satake diagrams of type ADE and even spherical
coweights $\mu$, we introduce the shifted twisted Yangians ${}^\imath Y_\mu$
and establish their PBW bases. We construct the iGKLO representations of
${}^\imath Y_\mu$, which factor through quotients known as truncated shifted
twisted Yangians (TSTY) ${}^\imath Y_\mu^\lambda$. In type AI with $\mu$
dominant, a variant of ${}^\imath Y_\mu^{N\varpi_1^\vee}$ is identified with
the TSTY in another definition which are isomorphic to finite W-algebras of
type BCD. We show that ${}^\imath Y_\mu$ quantizes the involutive fixed point
locus ${}^\imath W_\mu$ arising from affine Grassmannians of type ADE, and
expect that ${}^\imath Y_\mu^\lambda$ quantizes a top-dimensional component of
the affine Grassmannian islice ${}^\imath{\bar{W}}_\mu^\lambda$. We identify
the islices ${}^\imath{\bar{W}}_\mu^\lambda$ in type AI with suitable nilpotent
Slodowy slices of type BCD, building on the work of Lusztig and
Mirkovi\'c-Vybornov in type A. We propose a framework for producing
ortho-symplectic (and hybrid) Coulomb branches from split (and nonsplit) Satake
framed double quivers, which are conjectured to provide a normalization of the
islices ${}^\imath{\bar{W}}_\mu^\lambda$.

</details>
