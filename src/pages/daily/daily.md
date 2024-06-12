## 2024-06-12
### BAKU: An Efficient Transformer for Multi-Task Policy Learning

- **Authors**: Siddhant Haldar, Zhuoran Peng, Lerrel Pinto
- **Main Affiliations**: New York University
- **Tags**: `Policy Learning`

#### Abstract

Training generalist agents capable of solving diverse tasks is challenging, often requiring large datasets of expert demonstrations. This is particularly problematic in robotics, where each data point requires physical execution of actions in the real world. Thus, there is a pressing need for architectures that can effectively leverage the available training data. In this work, we present BAKU, a simple transformer architecture that enables efficient learning of multi-task robot policies. BAKU builds upon recent advancements in offline imitation learning and meticulously combines observation trunks, action chunking, multi-sensory observations, and action heads to substantially improve upon prior work. Our experiments on 129 simulated tasks across LIBERO, Meta-World suite, and the Deepmind Control suite exhibit an overall 18% absolute improvement over RT-1 and MT-ACT, with a 36% improvement on the harder LIBERO benchmark. On 30 real-world manipulation tasks, given an average of just 17 demonstrations per task, BAKU achieves a 91% success rate. Videos of the robot are best viewed at this https URL.

[Paper Link](https://arxiv.org/abs/2406.07539)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-47.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---



### A3VLM: Actionable Articulation-Aware Vision Language Model

- **Authors**: Siyuan Huang, Haonan Chang, Yuhan Liu, Yimeng Zhu, Hao Dong, Peng Gao, Abdeslam Boularias, Hongsheng Li
- **Main Affiliations**: SJTU, Shanghai AI Lab, Rutgers University, Yuandao AI, PKU, CUHK MMLab
- **Tags**: `LLM`

#### Abstract

Vision Language Models (VLMs) have received significant attention in recent years in the robotics community. VLMs are shown to be able to perform complex visual reasoning and scene understanding tasks, which makes them regarded as a potential universal solution for general robotics problems such as manipulation and navigation. However, previous VLMs for robotics such as RT-1, RT-2, and ManipLLM~ have focused on directly learning robot-centric actions. Such approaches require collecting a significant amount of robot interaction data, which is extremely costly in the real world. Thus, we propose A3VLM, an object-centric, actionable, articulation-aware vision language model. A3VLM focuses on the articulation structure and action affordances of objects. Its representation is robot-agnostic and can be translated into robot actions using simple action primitives. Extensive experiments in both simulation benchmarks and real-world settings demonstrate the effectiveness and stability of A3VLM. We release our code and other materials at this https URL.

[Paper Link](https://arxiv.org/abs/2406.07549)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-45.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---


## 2024-06-11

### Demonstrating HumanTHOR: A Simulation Platform and Benchmark for Human-Robot Collaboration in a Shared Workspace

- **Authors**: Chenxu Wang, Boyuan Du, Jiaxin Xu, Peiyan Li, Di Guo, Huaping Liu
- **Main Affiliations**: Tsinghua University
- **Tags**: `Simulation`

#### Abstract

Human-robot collaboration (HRC) in a shared workspace has become a common pattern in real-world robot applications and has garnered significant research interest. However, most existing studies for human-in-the-loop (HITL) collaboration with robots in a shared workspace evaluate in either simplified game environments or physical platforms, falling short in limited realistic significance or limited scalability. To support future studies, we build an embodied framework named HumanTHOR, which enables humans to act in the simulation environment through VR devices to support HITL collaborations in a shared workspace. To validate our system, we build a benchmark of everyday tasks and conduct a preliminary user study with two baseline algorithms. The results show that the robot can effectively assist humans in collaboration, demonstrating the significance of HRC. The comparison among different levels of baselines affirms that our system can adequately evaluate robot capabilities and serve as a benchmark for different robot algorithms. The experimental results also indicate that there is still much room in the area and our system can provide a preliminary foundation for future HRC research in a shared workspace. More information about the simulation environment, experiment videos, benchmark descriptions, and additional supplementary materials can be found on the website: this https URL.

[Paper Link](https://arxiv.org/abs/2406.06498)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-41.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-06-5

### DrEureka: Language Model Guided Sim-To-Real Transfer

- **Authors**: Yecheng Jason Ma, William Liang, Hung-Ju Wang, Sam Wang, Yuke Zhu, Linxi Fan, Osbert Bastani, Dinesh Jayaraman
- **Main Affiliations**: University of Pennsylvania, NVIDIA, University of Texas, Austin
- **Tags**: `Simulation`

#### Abstract

Transferring policies learned in simulation to the real world is a promising strategy for acquiring robot skills at scale. However, sim-to-real approaches typically rely on manual design and tuning of the task reward function as well as the simulation physics parameters, rendering the process slow and human-labor intensive. In this paper, we investigate using Large Language Models (LLMs) to automate and accelerate sim-to-real design. Our LLM-guided sim-to-real approach, DrEureka, requires only the physics simulation for the target task and automatically constructs suitable reward functions and domain randomization distributions to support real-world transfer. We first demonstrate that our approach can discover sim-to-real configurations that are competitive with existing human-designed ones on quadruped locomotion and dexterous manipulation tasks. Then, we showcase that our approach is capable of solving novel robot tasks, such as quadruped balancing and walking atop a yoga ball, without iterative manual design.

[Paper Link](https://arxiv.org/abs/2406.01967)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-27.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Query-based Semantic Gaussian Field for Scene Representation in Reinforcement Learning

- **Authors**: Jiaxu Wang, Ziyi Zhang, Qiang Zhang, Jia Li, Jingkai Sun, Mingyuan Sun, Junhao He, Renjing Xu
- **Main Affiliations**: HKUST (GZ), HKU, NEU
- **Tags**: `RL, 3DGS`

#### Abstract

Latent scene representation plays a significant role in training reinforcement learning (RL) agents. To obtain good latent vectors describing the scenes, recent works incorporate the 3D-aware latent-conditioned NeRF pipeline into scene representation learning. However, these NeRF-related methods struggle to perceive 3D structural information due to the inefficient dense sampling in volumetric rendering. Moreover, they lack fine-grained semantic information included in their scene representation vectors because they evenly consider free and occupied spaces. Both of them can destroy the performance of downstream RL tasks. To address the above challenges, we propose a novel framework that adopts the efficient 3D Gaussian Splatting (3DGS) to learn 3D scene representation for the first time. In brief, we present the Query-based Generalizable 3DGS to bridge the 3DGS technique and scene representations with more geometrical awareness than those in NeRFs. Moreover, we present the Hierarchical Semantics Encoding to ground the fine-grained semantic features to 3D Gaussians and further distilled to the scene representation vectors. We conduct extensive experiments on two RL platforms including Maniskill2 and Robomimic across 10 different tasks. The results show that our method outperforms the other 5 baselines by a large margin. We achieve the best success rates on 8 tasks and the second-best on the other two tasks.

[Paper Link](https://arxiv.org/abs/2406.02370)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-22.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots

- **Authors**: Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, Yuke Zhu
- **Main Affiliations**: The University of Texas at Austin, NVIDIA Research
- **Tags**: `Simulation`

#### Abstract

Recent advancements in Artificial Intelligence (AI) have largely been propelled by scaling. In Robotics, scaling is hindered by the lack of access to massive robot datasets. We advocate using realistic physical simulation as a means to scale environments, tasks, and datasets for robot learning methods. We present RoboCasa, a large-scale simulation framework for training generalist robots in everyday environments. RoboCasa features realistic and diverse scenes focusing on kitchen environments. We provide thousands of 3D assets across over 150 object categories and dozens of interactable furniture and appliances. We enrich the realism and diversity of our simulation with generative AI tools, such as object assets from text-to-3D models and environment textures from text-to-image models. We design a set of 100 tasks for systematic evaluation, including composite tasks generated by the guidance of large language models. To facilitate learning, we provide high-quality human demonstrations and integrate automated trajectory generation methods to substantially enlarge our datasets with minimal human burden. Our experiments show a clear scaling trend in using synthetically generated robot data for large-scale imitation learning and show great promise in harnessing simulation data in real-world tasks. Videos and open-source code are available at this https URL

[Paper Link](https://arxiv.org/abs/2406.02523)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-19.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-06-4

### PDP: Physics-Based Character Animation via Diffusion Policy

- **Authors**: Takara E. Truong, Michael Piseno, Zhaoming Xie, C. Karen Liu
- **Main Affiliations**: Stanford University
- **Tags**: `Diffusion Policy`

#### Abstract

Generating diverse and realistic human motion that can physically interact with an environment remains a challenging research area in character animation. Meanwhile, diffusion-based methods, as proposed by the robotics community, have demonstrated the ability to capture highly diverse and multi-modal skills. However, naively training a diffusion policy often results in unstable motions for high-frequency, under-actuated control tasks like bipedal locomotion due to rapidly accumulating compounding errors, pushing the agent away from optimal training trajectories. The key idea lies in using RL policies not just for providing optimal trajectories but for providing corrective actions in sub-optimal states, giving the policy a chance to correct for errors caused by environmental stimulus, model errors, or numerical errors in simulation. Our method, Physics-Based Character Animation via Diffusion Policy (PDP), combines reinforcement learning (RL) and behavior cloning (BC) to create a robust diffusion policy for physics-based character animation. We demonstrate PDP on perturbation recovery, universal motion tracking, and physics-based text-to-motion synthesis.

[Paper Link](https://arxiv.org/abs/2406.00960)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-15.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Learning Manipulation by Predicting Interaction

- **Authors**: Jia Zeng, Qingwen Bu, Bangjun Wang, Wenke Xia, Li Chen, Hao Dong, Haoming Song, Dong Wang, Di Hu, Ping Luo, Heming Cui, Bin Zhao, Xuelong Li, Yu Qiao, Hongyang Li
- **Main Affiliations**: Shanghai AI Lab, Shanghai Jiao Tong University, Renmin University of China, Peking University, Northwestern Polytechnical University, TeleAI, China Telecom Corp Ltd
- **Tags**: `Manipulation`

#### Abstract

Representation learning approaches for robotic manipulation have boomed in recent years. Due to the scarcity of in-domain robot data, prevailing methodologies tend to leverage large-scale human video datasets to extract generalizable features for visuomotor policy learning. Despite the progress achieved, prior endeavors disregard the interactive dynamics that capture behavior patterns and physical interaction during the manipulation process, resulting in an inadequate understanding of the relationship between objects and the environment. To this end, we propose a general pre-training pipeline that learns Manipulation by Predicting the Interaction (MPI) and enhances the visual representation.Given a pair of keyframes representing the initial and final states, along with language instructions, our algorithm predicts the transition frame and detects the interaction object, respectively. These two learning objectives achieve superior comprehension towards "how-to-interact" and "where-to-interact". We conduct a comprehensive evaluation of several challenging robotic tasks.The experimental results demonstrate that MPI exhibits remarkable improvement by 10% to 64% compared with previous state-of-the-art in real-world robot platforms as well as simulation environments. Code and checkpoints are publicly shared at this https URL.

[Paper Link](https://arxiv.org/abs/2406.00439)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-07.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Unsupervised Neural Motion Retargeting for Humanoid Teleoperation

- **Authors**: Satoshi Yagi, Mitsunori Tada, Eiji Uchibe, Suguru Kanoga, Takamitsu Matsubara, Jun Morimoto
- **Main Affiliations**: Kyoto University, National Institute of Advanced Industrial Science and Technology (AIST), Advanced Telecommunications Research Institute International (ATR), Graduate School of Science Technology, Nara Institute of Science and Technology
- **Tags**: `Humanoid`

#### Abstract

This study proposes an approach to human-to-humanoid teleoperation using GAN-based online motion retargeting, which obviates the need for the construction of pairwise datasets to identify the relationship between the human and the humanoid kinematics. Consequently, it can be anticipated that our proposed teleoperation system will reduce the complexity and setup requirements typically associated with humanoid controllers, thereby facilitating the development of more accessible and intuitive teleoperation systems for users without robotics knowledge. The experiments demonstrated the efficacy of the proposed method in retargeting a range of upper-body human motions to humanoid, including a body jab motion and a basketball shoot motion. Moreover, the human-in-the-loop teleoperation performance was evaluated by measuring the end-effector position errors between the human and the retargeted humanoid motions. The results demonstrated that the error was comparable to those of conventional motion retargeting methods that require pairwise motion datasets. Finally, a box pick-and-place task was conducted to demonstrate the usability of the developed humanoid teleoperation system.

[Paper Link](https://arxiv.org/abs/2406.00727)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_23-01.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Learning-based legged locomotion; state of the art and future perspectives

- **Authors**: Sehoon Ha, Joonho Lee, Michiel van de Panne, Zhaoming Xie, Wenhao Yu, Majid Khadiv
- **Main Affiliations**: Georgia Institute of Technology, Neuromeka, University of British Columbia, The AI Institute, 5Google DeepMind, Technical University of Munich
- **Tags**: `Locomotion`

#### Abstract

Legged locomotion holds the premise of universal mobility, a critical capability for many real-world robotic applications. Both model-based and learning-based approaches have advanced the field of legged locomotion in the past three decades. In recent years, however, a number of factors have dramatically accelerated progress in learning-based methods, including the rise of deep learning, rapid progress in simulating robotic systems, and the availability of high-performance and affordable hardware. This article aims to give a brief history of the field, to summarize recent efforts in learning locomotion skills for quadrupeds, and to provide researchers new to the area with an understanding of the key issues involved. With the recent proliferation of humanoid robots, we further outline the rapid rise of analogous methods for bipedal locomotion. We conclude with a discussion of open problems as well as related societal impact.

[Paper Link](https://arxiv.org/abs/2406.01152)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_22-53.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation

- **Authors**: Guanxing Lu, Zifeng Gao, Tianxing Chen, Wenxun Dai, Ziwei Wang, Yansong Tang
- **Main Affiliations**: Tsinghua University, Shanghai AI Laboratory, Carnegie Mellon University
- **Tags**: `Diffusion Policy, Consistency Model`

#### Abstract

Diffusion models have been verified to be effective in generating complex distributions from natural images to motion trajectories. Recent diffusion-based methods show impressive performance in 3D robotic manipulation tasks, whereas they suffer from severe runtime inefficiency due to multiple denoising steps, especially with high-dimensional observations. To this end, we propose a real-time robotic manipulation model named ManiCM that imposes the consistency constraint to the diffusion process, so that the model can generate robot actions in only one-step inference. Specifically, we formulate a consistent diffusion process in the robot action space conditioned on the point cloud input, where the original action is required to be directly denoised from any point along the ODE trajectory. To model this process, we design a consistency distillation technique to predict the action sample directly instead of predicting the noise within the vision community for fast convergence in the low-dimensional action manifold. We evaluate ManiCM on 31 robotic manipulation tasks from Adroit and Metaworld, and the results demonstrate that our approach accelerates the state-of-the-art method by 10 times in average inference speed while maintaining competitive average success rate.

[Paper Link](https://arxiv.org/abs/2406.01586)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-12_22-46.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-30

### Grasp as You Say: Language-guided Dexterous Grasp Generation

- **Authors**: Yi-Lin Wei, Jian-Jian Jiang, Chengyi Xing, Xiantuo Tan, Xiao-Ming Wu, Hao Li, Mark Cutkosky, Wei-Shi Zheng
- **Main Affiliations**: Sun Yat-sen University, Stanford University, Wuhan University
- **Tags**: `Simulation`

#### Abstract

This paper explores a novel task ""Dexterous Grasp as You Say"" (DexGYS), enabling robots to perform dexterous grasping based on human commands expressed in natural language. However, the development of this field is hindered by the lack of datasets with natural human guidance; thus, we propose a language-guided dexterous grasp dataset, named DexGYSNet, offering high-quality dexterous grasp annotations along with flexible and fine-grained human language guidance. Our dataset construction is cost-efficient, with the carefully-design hand-object interaction retargeting strategy, and the LLM-assisted language guidance annotation system. Equipped with this dataset, we introduce the DexGYSGrasp framework for generating dexterous grasps based on human language instructions, with the capability of producing grasps that are intent-aligned, high quality and diversity. To achieve this capability, our framework decomposes the complex learning process into two manageable progressive objectives and introduce two components to realize them. The first component learns the grasp distribution focusing on intention alignment and generation diversity. And the second component refines the grasp quality while maintaining intention consistency. Extensive experiments are conducted on DexGYSNet and real world environment for validation.

[Paper Link](https://arxiv.org/abs/2405.19291)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-06-01_18-02.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-21

### URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images

- **Authors**: Zoey Chen, Aaron Walsman, Marius Memmel, Kaichun Mo, Alex Fang, Karthikeya Vemuri, Alan Wu, Dieter Fox, Abhishek Gupta
- **Main Affiliations**: University of Washington, Nvidia
- **Tags**: `Simulation`

#### Abstract

Constructing simulation scenes that are both visually and physically realistic is a problem of practical interest in domains ranging from robotics to computer vision. This problem has become even more relevant as researchers wielding large data-hungry learning methods seek new sources of training data for physical decision-making systems. However, building simulation models is often still done by hand. A graphic designer and a simulation engineer work with predefined assets to construct rich scenes with realistic dynamic and kinematic properties. While this may scale to small numbers of scenes, to achieve the generalization properties that are required for data-driven robotic control, we require a pipeline that is able to synthesize large numbers of realistic scenes, complete with 'natural' kinematic and dynamic structures. To attack this problem, we develop models for inferring structure and generating simulation scenes from natural images, allowing for scalable scene generation from web-scale datasets. To train these image-to-simulation models, we show how controllable text-to-image generative models can be used in generating paired training data that allows for modeling of the inverse problem, mapping from realistic images back to complete scene models. We show how this paradigm allows us to build large datasets of scenes in simulation with semantic and physical realism. We present an integrated end-to-end pipeline that generates simulation scenes complete with articulated kinematic and dynamic structures from real-world images and use these for training robotic control policies. We then robustly deploy in the real world for tasks like articulated object manipulation. In doing so, our work provides both a pipeline for large-scale generation of simulation environments and an integrated system for training robust robotic control policies in the resulting environments.

[Paper Link](https://arxiv.org/abs/2405.11656)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-21_22-43.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Octo: An Open-Source Generalist Robot Policy

- **Authors**: Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, Sergey Levine
- **Main Affiliations**: University of California, Berkeley, Carnegie Mellon University, Google DeepMind
- **Tags**: `large policy`

#### Abstract

Large policies pretrained on diverse robot datasets have the potential to transform robotic learning: instead of training new policies from scratch, such generalist robot policies may be finetuned with only a little in-domain data, yet generalize broadly. However, to be widely applicable across a range of robotic learning scenarios, environments, and tasks, such policies need to handle diverse sensors and action spaces, accommodate a variety of commonly used robotic platforms, and finetune readily and efficiently to new domains. In this work, we aim to lay the groundwork for developing open-source, widely applicable, generalist policies for robotic manipulation. As a first step, we introduce Octo, a large transformer-based policy trained on 800k trajectories from the Open X-Embodiment dataset, the largest robot manipulation dataset to date. It can be instructed via language commands or goal images and can be effectively finetuned to robot setups with new sensory inputs and action spaces within a few hours on standard consumer GPUs. In experiments across 9 robotic platforms, we demonstrate that Octo serves as a versatile policy initialization that can be effectively finetuned to new observation and action spaces. We also perform detailed ablations of design decisions for the Octo model, from architecture to training data, to guide future research on building generalist robot models.

[Paper Link](https://arxiv.org/abs/2405.12213)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-21_21-57.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-17

### Natural Language Can Help Bridge the Sim2Real Gap

- **Authors**: Albert Yu, Adeline Foote, Raymond Mooney, Roberto Martín-Martín
- **Main Affiliations**: University of Texas at Austin
- **Tags**: `Simulation to Reality`

#### Abstract

The main challenge in learning image-conditioned robotic policies is acquiring a visual representation conducive to low-level control. Due to the high dimensionality of the image space, learning a good visual representation requires a considerable amount of visual data. However, when learning in the real world, data is expensive. Sim2Real is a promising paradigm for overcoming data scarcity in the real-world target domain by using a simulator to collect large amounts of cheap data closely related to the target task. However, it is difficult to transfer an image-conditioned policy from sim to real when the domains are very visually dissimilar. To bridge the sim2real visual gap, we propose using natural language descriptions of images as a unifying signal across domains that captures the underlying task-relevant semantics. Our key insight is that if two image observations from different domains are labeled with similar language, the policy should predict similar action distributions for both images. We demonstrate that training the image encoder to predict the language description or the distance between descriptions of a sim or real image serves as a useful, data-efficient pretraining step that helps learn a domain-invariant image representation. We can then use this image encoder as the backbone of an IL policy trained simultaneously on a large amount of simulated and a handful of real demonstrations. Our approach outperforms widely used prior sim2real methods and strong vision-language pretraining baselines like CLIP and R3M by 25 to 40%.

[Paper Link](https://arxiv.org/abs/2405.10020)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_22-04.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-14

### Scene Action Maps: Behavioural Maps for Navigation without Metric Information

- **Authors**: Joel Loo, David Hsu
- **Main Affiliations**: National University of Singapore
- **Tags**: `Navigation`

#### Abstract

Humans are remarkable in their ability to navigate without metric information. We can read abstract 2D maps, such as floor-plans or hand-drawn sketches, and use them to navigate in unseen rich 3D environments, without requiring prior traversals to map out these scenes in detail. We posit that this is enabled by the ability to represent the environment abstractly as interconnected navigational behaviours, e.g., "follow the corridor" or "turn right", while avoiding detailed, accurate spatial information at the metric level. We introduce the Scene Action Map (SAM), a behavioural topological graph, and propose a learnable map-reading method, which parses a variety of 2D maps into SAMs. Map-reading extracts salient information about navigational behaviours from the overlooked wealth of pre-existing, abstract and inaccurate maps, ranging from floor-plans to sketches. We evaluate the performance of SAMs for navigation, by building and deploying a behavioural navigation stack on a quadrupedal robot. Videos and more information is available at: this https URL.

[Paper Link](https://arxiv.org/abs/2405.07948)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_17-52.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### SPIN: Simultaneous Perception, Interaction and Navigation

- **Authors**: Shagun Uppal, Ananye Agarwal, Haoyu Xiong, Kenneth Shaw, Deepak Pathak
- **Main Affiliations**: Carnegie Mellon University
- **Tags**: `whole body navigation`

#### Abstract

While there has been remarkable progress recently in the fields of manipulation and locomotion, mobile manipulation remains a long-standing challenge. Compared to locomotion or static manipulation, a mobile system must make a diverse range of long-horizon tasks feasible in unstructured and dynamic environments. While the applications are broad and interesting, there are a plethora of challenges in developing these systems such as coordination between the base and arm, reliance on onboard perception for perceiving and interacting with the environment, and most importantly, simultaneously integrating all these parts together. Prior works approach the problem using disentangled modular skills for mobility and manipulation that are trivially tied together. This causes several limitations such as compounding errors, delays in decision-making, and no whole-body coordination. In this work, we present a reactive mobile manipulation framework that uses an active visual system to consciously perceive and react to its environment. Similar to how humans leverage whole-body and hand-eye coordination, we develop a mobile manipulator that exploits its ability to move and see, more specifically -- to move in order to see and to see in order to move. This allows it to not only move around and interact with its environment but also, choose "when" to perceive "what" using an active visual system. We observe that such an agent learns to navigate around complex cluttered scenarios while displaying agile whole-body coordination using only ego-vision without needing to create environment maps. Results visualizations and videos at this https URL

[Paper Link](https://arxiv.org/abs/2405.07991)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_17-36.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-10

### ASGrasp: Generalizable Transparent Object Reconstruction and Grasping from RGB-D Active Stereo Camera

- **Authors**: Jun Shi, Yong A, Yixiang Jin, Dingzhe Li, Haoyu Niu, Zhezhu Jin, He Wang
- **Main Affiliations**: Samsung R&D Institute China-Beijing, Peking University, Galbot, BAAI
- **Tags**: `Large Language Models`

#### Abstract

In this paper, we tackle the problem of grasping transparent and specular objects. This issue holds importance, yet it remains unsolved within the field of robotics due to failure of recover their accurate geometry by depth cameras. For the first time, we propose ASGrasp, a 6-DoF grasp detection network that uses an RGB-D active stereo camera. ASGrasp utilizes a two-layer learning-based stereo network for the purpose of transparent object reconstruction, enabling material-agnostic object grasping in cluttered environments. In contrast to existing RGB-D based grasp detection methods, which heavily depend on depth restoration networks and the quality of depth maps generated by depth cameras, our system distinguishes itself by its ability to directly utilize raw IR and RGB images for transparent object geometry reconstruction. We create an extensive synthetic dataset through domain randomization, which is based on GraspNet-1Billion. Our experiments demonstrate that ASGrasp can achieve over 90% success rate for generalizable transparent object grasping in both simulation and the real via seamless sim-to-real transfer. Our method significantly outperforms SOTA networks and even surpasses the performance upper bound set by perfect visible point cloud inputs.Project page: this https URL

[Paper Link](https://arxiv.org/abs/2405.05648)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_16-05.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### RoboHop: Segment-based Topological Map Representation for Open-World Visual Navigation

- **Authors**: Sourav Garg, Krishan Rana, Mehdi Hosseinzadeh, Lachlan Mares, Niko Sünderhauf, Feras Dayoub, Ian Reid
- **Main Affiliations**: The University of Adelaide
- **Tags**: `Large Language Models`

#### Abstract

Mapping is crucial for spatial reasoning, planning and robot navigation. Existing approaches range from metric, which require precise geometry-based optimization, to purely topological, where image-as-node based graphs lack explicit object-level reasoning and interconnectivity. In this paper, we propose a novel topological representation of an environment based on "image segments", which are semantically meaningful and open-vocabulary queryable, conferring several advantages over previous works based on pixel-level features. Unlike 3D scene graphs, we create a purely topological graph with segments as nodes, where edges are formed by a) associating segment-level descriptors between pairs of consecutive images and b) connecting neighboring segments within an image using their pixel centroids. This unveils a "continuous sense of a place", defined by inter-image persistence of segments along with their intra-image neighbours. It further enables us to represent and update segment-level descriptors through neighborhood aggregation using graph convolution layers, which improves robot localization based on segment-level retrieval. Using real-world data, we show how our proposed map representation can be used to i) generate navigation plans in the form of "hops over segments" and ii) search for target objects using natural language queries describing spatial relations of objects. Furthermore, we quantitatively analyze data association at the segment level, which underpins inter-image connectivity during mapping and segment-level localization when revisiting the same place. Finally, we show preliminary trials on segment-level `hopping' based zero-shot real-world navigation. Project page with supplementary details: this http URL

[Paper Link](https://arxiv.org/abs/2405.05792)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_16-02.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Robots Can Feel: LLM-based Framework for Robot Ethical Reasoning

- **Authors**: Artem Lykov, Miguel Altamirano Cabrera, Koffivi Fidèle Gbagbe, Dzmitry Tsetserukou
- **Main Affiliations**: Intelligent Space Robotics Laboratory, Center for Digital Engineering, Skolkovo Institute of Science and Technology, Moscow, Russia
- **Tags**: `Large Language Models`

#### Abstract

This paper presents the development of a novel ethical reasoning framework for robots. "Robots Can Feel" is the first system for robots that utilizes a combination of logic and human-like emotion simulation to make decisions in morally complex situations akin to humans. The key feature of the approach is the management of the Emotion Weight Coefficient - a customizable parameter to assign the role of emotions in robot decision-making. The system aims to serve as a tool that can equip robots of any form and purpose with ethical behavior close to human standards. Besides the platform, the system is independent of the choice of the base model. During the evaluation, the system was tested on 8 top up-to-date LLMs (Large Language Models). This list included both commercial and open-source models developed by various companies and countries. The research demonstrated that regardless of the model choice, the Emotions Weight Coefficient influences the robot's decision similarly. According to ANOVA analysis, the use of different Emotion Weight Coefficients influenced the final decision in a range of situations, such as in a request for a dietary violation F(4, 35) = 11.2, p = 0.0001 and in an animal compassion situation F(4, 35) = 8.5441, p = 0.0001. A demonstration code repository is provided at: this https URL

[Paper Link](https://arxiv.org/abs/2405.05824)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_15-56.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Evaluating Real-World Robot Manipulation Policies in Simulation

- **Authors**: Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, Ted Xiao
- **Main Affiliations**: UC San Diego, Stanford University, UC Berkeley, Google Deepmind
- **Tags**: `Simulation to Reality`

#### Abstract

The field of robotics has made significant advances towards generalist robot manipulation policies. However, real-world evaluation of such policies is not scalable and faces reproducibility challenges, which are likely to worsen as policies broaden the spectrum of tasks they can perform. We identify control and visual disparities between real and simulated environments as key challenges for reliable simulated evaluation and propose approaches for mitigating these gaps without needing to craft full-fidelity digital twins of real-world environments. We then employ these approaches to create SIMPLER, a collection of simulated environments for manipulation policy evaluation on common real robot setups. Through paired sim-and-real evaluations of manipulation policies, we demonstrate strong correlation between policy performance in SIMPLER environments and in the real world. Additionally, we find that SIMPLER evaluations accurately reflect real-world policy behavior modes such as sensitivity to various distribution shifts. We open-source all SIMPLER environments along with our workflow for creating new environments at this https URL to facilitate research on general-purpose manipulation policies and simulated evaluation frameworks.

[Paper Link](https://arxiv.org/abs/2405.05941)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_15-47.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## Probing Multimodal LLMs as World Models for Driving

- **Authors**: Shiva Sreeram, Tsun-Hsuan Wang, Alaa Maalouf, Guy Rosman, Sertac Karaman, Daniela Rus
- **Main Affiliations**: MIT
- **Tags**: `Large Language Models`

#### Abstract

We provide a sober look at the application of Multimodal Large Language Models (MLLMs) within the domain of autonomous driving and challenge/verify some common assumptions, focusing on their ability to reason and interpret dynamic driving scenarios through sequences of images/frames in a closed-loop control environment. Despite the significant advancements in MLLMs like GPT-4V, their performance in complex, dynamic driving environments remains largely untested and presents a wide area of exploration. We conduct a comprehensive experimental study to evaluate the capability of various MLLMs as world models for driving from the perspective of a fixed in-car camera. Our findings reveal that, while these models proficiently interpret individual images, they struggle significantly with synthesizing coherent narratives or logical sequences across frames depicting dynamic behavior. The experiments demonstrate considerable inaccuracies in predicting (i) basic vehicle dynamics (forward/backward, acceleration/deceleration, turning right or left), (ii) interactions with other road actors (e.g., identifying speeding cars or heavy traffic), (iii) trajectory planning, and (iv) open-set dynamic scene reasoning, suggesting biases in the models' training data. To enable this experimental study we introduce a specialized simulator, DriveSim, designed to generate diverse driving scenarios, providing a platform for evaluating MLLMs in the realms of driving. Additionally, we contribute the full open-source code and a new dataset, "Eval-LLM-Drive", for evaluating MLLMs in driving. Our results highlight a critical gap in the current capabilities of state-of-the-art MLLMs, underscoring the need for enhanced foundation models to improve their applicability in real-world dynamic environments.

[Paper Link](https://arxiv.org/abs/2405.05956)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-20_15-44.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-07

### DexSkills: Skill Segmentation Using Haptic Data for Learning Autonomous Long-Horizon Robotic Manipulation Tasks

- **Authors**: Xiaofeng Mao, Gabriele Giudici, Claudio Coppola, Kaspar Althoefer, Ildar Farkhatdinov, Zhibin Li, Lorenzo Jamone
- **Main Affiliations**: University of Edinburgh, ARQ (the Centre for Advanced Robotics @ Queen Mary)
- **Tags**: `teleoperation`

#### Abstract

Effective execution of long-horizon tasks with dexterous robotic hands remains a significant challenge in real-world problems. While learning from human demonstrations have shown encouraging results, they require extensive data collection for training. Hence, decomposing long-horizon tasks into reusable primitive skills is a more efficient approach. To achieve so, we developed DexSkills, a novel supervised learning framework that addresses long-horizon dexterous manipulation tasks using primitive skills. DexSkills is trained to recognize and replicate a select set of skills using human demonstration data, which can then segment a demonstrated long-horizon dexterous manipulation task into a sequence of primitive skills to achieve one-shot execution by the robot directly. Significantly, DexSkills operates solely on proprioceptive and tactile data, i.e., haptic data. Our real-world robotic experiments show that DexSkills can accurately segment skills, thereby enabling autonomous robot execution of a diverse range of tasks.

[Paper Link](https://arxiv.org/abs/2405.03476)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-09_17-11.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-05-06

### Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots

- **Authors**: Joonho Lee, Marko Bjelonic, Alexander Reske, Lorenz Wellhausen, Takahiro Miki, Marco Hutter
- **Main Affiliations**: Robotic Systems Lab-ETH Zurich
- **Tags**: `Wheeled-Legged Robots`

#### Abstract

Autonomous wheeled-legged robots have the potential to transform logistics systems, improving operational efficiency and adaptability in urban environments. Navigating urban environments, however, poses unique challenges for robots, necessitating innovative solutions for locomotion and navigation. These challenges include the need for adaptive locomotion across varied terrains and the ability to navigate efficiently around complex dynamic obstacles. This work introduces a fully integrated system comprising adaptive locomotion control, mobility-aware local navigation planning, and large-scale path planning within the city. Using model-free reinforcement learning (RL) techniques and privileged learning, we develop a versatile locomotion controller. This controller achieves efficient and robust locomotion over various rough terrains, facilitated by smooth transitions between walking and driving modes. It is tightly integrated with a learned navigation controller through a hierarchical RL framework, enabling effective navigation through challenging terrain and various obstacles at high speed. Our controllers are integrated into a large-scale urban navigation system and validated by autonomous, kilometer-scale navigation missions conducted in Zurich, Switzerland, and Seville, Spain. These missions demonstrate the system's robustness and adaptability, underscoring the importance of integrated control systems in achieving seamless navigation in complex environments. Our findings support the feasibility of wheeled-legged robots and hierarchical RL for autonomous navigation, with implications for last-mile delivery and beyond.

[Paper Link](https://arxiv.org/abs/2405.01792)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-09_16-57.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-30

### Dexterous Grasp Transformer

- **Authors**: Guo-Hao Xu, Yi-Lin Wei, Dian Zheng, Xiao-Ming Wu, Wei-Shi Zheng
- **Main Affiliations**: Sun Yat-sen University
- **Tags**: `Dexterous Grasp`

#### Abstract

In this work, we propose a novel discriminative framework for dexterous grasp generation, named Dexterous Grasp TRansformer (DGTR), capable of predicting a diverse set of feasible grasp poses by processing the object point cloud with only one forward pass. We formulate dexterous grasp generation as a set prediction task and design a transformer-based grasping model for it. However, we identify that this set prediction paradigm encounters several optimization challenges in the field of dexterous grasping and results in restricted performance. To address these issues, we propose progressive strategies for both the training and testing phases. First, the dynamic-static matching training (DSMT) strategy is presented to enhance the optimization stability during the training phase. Second, we introduce the adversarial-balanced test-time adaptation (AB-TTA) with a pair of adversarial losses to improve grasping quality during the testing phase. Experimental results on the DexGraspNet dataset demonstrate the capability of DGTR to predict dexterous grasp poses with both high quality and diversity. Notably, while keeping high quality, the diversity of grasp poses predicted by DGTR significantly outperforms previous works in multiple metrics without any data pre-processing. Codes are available at this https URL .

[Paper Link](https://arxiv.org/abs/2404.18135)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-03_16-56.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Quadruped robot traversing 3D complex environments with limited perception

- **Authors**: Yi Cheng, Hang Liu, Guoping Pan, Linqi Ye, Houde Liu, Bin Liang
- **Main Affiliations**: Tsinghua University
- **Tags**: `Simulation to Reality`

#### Abstract

Traversing 3-D complex environments has always been a significant challenge for legged locomotion. Existing methods typically rely on external sensors such as vision and lidar to preemptively react to obstacles by acquiring environmental information. However, in scenarios like nighttime or dense forests, external sensors often fail to function properly, necessitating robots to rely on proprioceptive sensors to perceive diverse obstacles in the environment and respond promptly. This task is undeniably challenging. Our research finds that methods based on collision detection can enhance a robot's perception of environmental obstacles. In this work, we propose an end-to-end learning-based quadruped robot motion controller that relies solely on proprioceptive sensing. This controller can accurately detect, localize, and agilely respond to collisions in unknown and complex 3D environments, thereby improving the robot's traversability in complex environments. We demonstrate in both simulation and real-world experiments that our method enables quadruped robots to successfully traverse challenging obstacles in various complex environments.

[Paper Link](https://arxiv.org/abs/2404.18225)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-03_16-52.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-29

### Part-Guided 3D RL for Sim2Real Articulated Object Manipulation

- **Authors**: Pengwei Xie, Rui Chen, Siang Chen, Yuzhe Qin, Fanbo Xiang, Tianyu Sun, Jing Xu, Guijin Wang, Hao Su
- **Main Affiliations**: Tsinghua University, Shanghai AI Laboratory, University of California, San Diego
- **Tags**: `Simulation to Reality`

#### Abstract

Manipulating unseen articulated objects through visual feedback is a critical but challenging task for real robots. Existing learning-based solutions mainly focus on visual affordance learning or other pre-trained visual models to guide manipulation policies, which face challenges for novel instances in real-world scenarios. In this paper, we propose a novel part-guided 3D RL framework, which can learn to manipulate articulated objects without demonstrations. We combine the strengths of 2D segmentation and 3D RL to improve the efficiency of RL policy training. To improve the stability of the policy on real robots, we design a Frame-consistent Uncertainty-aware Sampling (FUS) strategy to get a condensed and hierarchical 3D representation. In addition, a single versatile RL policy can be trained on multiple articulated object manipulation tasks simultaneously in simulation and shows great generalizability to novel categories and instances. Experimental results demonstrate the effectiveness of our framework in both simulation and real-world settings. Our code is available at this https URL.

[Paper Link](https://arxiv.org/abs/2404.17302)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-03_16-31.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Ag2Manip: Learning Novel Manipulation Skills with Agent-Agnostic Visual and Action Representations

- **Authors**: Puhao Li, Tengyu Liu, Yuyang Li, Muzhi Han, Haoran Geng, Shu Wang, Yixin Zhu, Song-Chun Zhu, Siyuan Huang
- **Main Affiliations**: Beijing Institute for General Artificial Intelligence, Tsinghua University, Peking University, University of California, Los Angeles
- **Tags**: `Agent-Agnostic Visual Representation`

#### Abstract

Autonomous robotic systems capable of learning novel manipulation tasks are poised to transform industries from manufacturing to service automation. However, modern methods (e.g., VIP and R3M) still face significant hurdles, notably the domain gap among robotic embodiments and the sparsity of successful task executions within specific action spaces, resulting in misaligned and ambiguous task representations. We introduce Ag2Manip (Agent-Agnostic representations for Manipulation), a framework aimed at surmounting these challenges through two key innovations: a novel agent-agnostic visual representation derived from human manipulation videos, with the specifics of embodiments obscured to enhance generalizability; and an agent-agnostic action representation abstracting a robot's kinematics to a universal agent proxy, emphasizing crucial interactions between end-effector and object. Ag2Manip's empirical validation across simulated benchmarks like FrankaKitchen, ManiSkill, and PartManip shows a 325% increase in performance, achieved without domain-specific demonstrations. Ablation studies underline the essential contributions of the visual and action representations to this success. Extending our evaluations to the real world, Ag2Manip significantly improves imitation learning success rates from 50% to 77.5%, demonstrating its effectiveness and generalizability across both simulated and physical environments.

[Paper Link](https://arxiv.org/abs/2404.17521)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-05-03_16-25.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-26

### Leveraging Pretrained Latent Representations for Few-Shot Imitation Learning on a Dexterous Robotic Hand

- **Authors**: Davide Liconti, Yasunori Toshimitsu, Robert Katzschmann
- **Main Affiliations**: ETH
- **Tags**: `Behavioral Cloning`

#### Abstract

In the context of imitation learning applied to dexterous robotic hands, the high complexity of the systems makes learning complex manipulation tasks challenging. However, the numerous datasets depicting human hands in various different tasks could provide us with better knowledge regarding human hand motion. We propose a method to leverage multiple large-scale task-agnostic datasets to obtain latent representations that effectively encode motion subtrajectories that we included in a transformer-based behavior cloning method. Our results demonstrate that employing latent representations yields enhanced performance compared to conventional behavior cloning methods, particularly regarding resilience to errors and noise in perception and proprioception. Furthermore, the proposed approach solely relies on human demonstrations, eliminating the need for teleoperation and, therefore, accelerating the data acquisition process. Accurate inverse kinematics for fingertip retargeting ensures precise transfer from human hand data to the robot, facilitating effective learning and deployment of manipulation policies. Finally, the trained policies have been successfully transferred to a real-world 23Dof robotic system.

[Paper Link](https://arxiv.org/abs/2404.16483)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-28_11-49.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Learning Visuotactile Skills with Two Multifingered Hands

- **Authors**: Toru Lin, Yu Zhang, Qiyang Li, Haozhi Qi, Brent Yi, Sergey Levine, Jitendra Malik
- **Main Affiliations**: University of California, Berkeley
- **Tags**: `visuotactile`

#### Abstract

Aiming to replicate human-like dexterity, perceptual experiences, and motion patterns, we explore learning from human demonstrations using a bimanual system with multifingered hands and visuotactile data. Two significant challenges exist: the lack of an affordable and accessible teleoperation system suitable for a dual-arm setup with multifingered hands, and the scarcity of multifingered hand hardware equipped with touch sensing. To tackle the first challenge, we develop HATO, a low-cost hands-arms teleoperation system that leverages off-the-shelf electronics, complemented with a software suite that enables efficient data collection; the comprehensive software suite also supports multimodal data processing, scalable policy learning, and smooth policy deployment. To tackle the latter challenge, we introduce a novel hardware adaptation by repurposing two prosthetic hands equipped with touch sensors for research. Using visuotactile data collected from our system, we learn skills to complete long-horizon, high-precision tasks which are difficult to achieve without multifingered dexterity and touch feedback. Furthermore, we empirically investigate the effects of dataset size, sensing modality, and visual input preprocessing on policy learning. Our results mark a promising step forward in bimanual multifingered manipulation from visuotactile data. Videos, code, and datasets can be found at this https URL .

[Paper Link](https://arxiv.org/abs/2404.16823)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-28_11-36.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-17

### Scaling Instructable Agents Across Many Simulated Worlds

- **Authors**: SIMA Team, Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, Stephanie C. Y. Chan, Jeff Clune, Adrian Collister, Vikki Copeman, Alex Cullum, Ishita Dasgupta, Dario de Cesare, Julia Di Trapani, Yani Donchev, Emma Dunleavy, Martin Engelcke, Ryan Faulkner, Frankie Garcia, Charles Gbadamosi, Zhitao Gong, Lucy Gonzales, Karol Gregor, Arne Olav Hallingstad, Tim Harley, Sam Haves, Felix Hill, Ed Hirst, Drew A. Hudson, Steph Hughes-Fitt, Danilo J. Rezende, Mimi Jasarevic, Laura Kampis, Rosemary Ke, Thomas Keck, Junkyung Kim, Oscar Knagg, Kavya Kopparapu, Andrew Lampinen, Shane Legg, Alexander Lerchner, Marjorie Limont, Yulan Liu, Maria Loks-Thompson, Joseph Marino, Kathryn Martin Cussons, Loic Matthey, Siobhan Mcloughlin, Piermaria Mendolicchio, Hamza Merzic, Anna Mitenkova, Alexandre Moufarek, Valeria Oliveira, Yanko Oliveira, Hannah Openshaw, Renke Pan, Aneesh Pappu, Alex Platonov, Ollie Purkiss, David Reichert, John Reid, Pierre Harvey Richemond, Tyson Roberts, Giles Ruscoe, Jaume Sanchez Elias, Tasha Sandars, Daniel P. Sawyer, Tim Scholtes, Guy Simmons, Daniel Slater, Hubert Soyer, Heiko Strathmann, Peter Stys, Allison C. Tam, Denis Teplyashin, Tayfun Terzi, Davide Vercelli, Bojan Vujatovic, Marcus Wainwright, Jane X. Wang, Zhengdong Wang, Daan Wierstra, Duncan Williams, Nathaniel Wong, Sarah York, Nick Young
- **Main Affiliations**: Google DeepMind
- **Tags**: `dataset`

#### Abstract

Building embodied AI systems that can follow arbitrary language instructions in any 3D environment is a key challenge for creating general AI. Accomplishing this goal requires learning to ground language in perception and embodied actions, in order to accomplish complex tasks. The Scalable, Instructable, Multiworld Agent (SIMA) project tackles this by training agents to follow free-form instructions across a diverse range of virtual 3D environments, including curated research environments as well as open-ended, commercial video games. Our goal is to develop an instructable agent that can accomplish anything a human can do in any simulated 3D environment. Our approach focuses on language-driven generality while imposing minimal assumptions. Our agents interact with environments in real-time using a generic, human-like interface: the inputs are image observations and language instructions and the outputs are keyboard-and-mouse actions. This general approach is challenging, but it allows agents to ground language across many visually complex and semantically rich environments while also allowing us to readily run agents in new environments. In this paper we describe our motivation and goal, the initial progress we have made, and promising preliminary results on several diverse research environments and a variety of commercial video games.

[Paper Link](https://arxiv.org/abs/2404.10179)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-17_21-52.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Closed-Loop Open-Vocabulary Mobile Manipulation with GPT-4V

- **Authors**: Peiyuan Zhi, Zhiyuan Zhang, Muzhi Han, Zeyu Zhang, Zhitian Li, Ziyuan Jiao, Baoxiong Jia, Siyuan Huang
- **Main Affiliations**: State Key Laboratory of General Artificial Intelligence-Beijing Institute for General Artificial Intelligence (BIGAI), Department of Automation-Tsinghua University, University of California Los Angeles
- **Tags**: `Large Language Models`

#### Abstract

Autonomous robot navigation and manipulation in open environments require reasoning and replanning with closed-loop feedback. We present COME-robot, the first closed-loop framework utilizing the GPT-4V vision-language foundation model for open-ended reasoning and adaptive planning in real-world scenarios. We meticulously construct a library of action primitives for robot exploration, navigation, and manipulation, serving as callable execution modules for GPT-4V in task planning. On top of these modules, GPT-4V serves as the brain that can accomplish multimodal reasoning, generate action policy with code, verify the task progress, and provide feedback for replanning. Such design enables COME-robot to (i) actively perceive the environments, (ii) perform situated reasoning, and (iii) recover from failures. Through comprehensive experiments involving 8 challenging real-world tabletop and manipulation tasks, COME-robot demonstrates a significant improvement in task success rate (~25%) compared to state-of-the-art baseline methods. We further conduct comprehensive analyses to elucidate how COME-robot's design facilitates failure recovery, free-form instruction following, and long-horizon task planning.

[Paper Link](https://arxiv.org/abs/2404.10220)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-17_21-47.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-16

### PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI

- **Authors**: Yandan Yang, Baoxiong Jia, Peiyuan Zhi, Siyuan Huang
- **Main Affiliations**: State Key Laboratory of General Artificial Intelligence,Beijing Institute for General Artificial Intelligence (BIGAI)
- **Tags**: `dataset`

#### Abstract

With recent developments in Embodied Artificial Intelligence (EAI) research, there has been a growing demand for high-quality, large-scale interactive scene generation. While prior methods in scene synthesis have prioritized the naturalness and realism of the generated scenes, the physical plausibility and interactivity of scenes have been largely left unexplored. To address this disparity, we introduce PhyScene, a novel method dedicated to generating interactive 3D scenes characterized by realistic layouts, articulated objects, and rich physical interactivity tailored for embodied agents. Based on a conditional diffusion model for capturing scene layouts, we devise novel physics- and interactivity-based guidance mechanisms that integrate constraints from object collision, room layout, and object reachability. Through extensive experiments, we demonstrate that PhyScene effectively leverages these guidance functions for physically interactable scene synthesis, outperforming existing state-of-the-art scene synthesis methods by a large margin. Our findings suggest that the scenes generated by PhyScene hold considerable potential for facilitating diverse skill acquisition among agents within interactive environments, thereby catalyzing further advancements in embodied AI research. Project website: this http URL.

[Paper Link](https://arxiv.org/abs/2404.09465)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-17_21-32.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-12

### QuasiSim: Parameterized Quasi-Physical Simulators for Dexterous Manipulations Transfer

- **Authors**: Xueyi Liu, Kangbo Lyu, Jieqiong Zhang, Tao Du, Li Yi
- **Main Affiliations**: Tsinghua University, Shanghai AI Laboratory, Shanghai Qi Zhi Institute
- **Tags**: `Simulation to Reality`

#### Abstract

We explore the dexterous manipulation transfer problem by designing simulators. The task wishes to transfer human manipulations to dexterous robot hand simulations and is inherently difficult due to its intricate, highly-constrained, and discontinuous dynamics and the need to control a dexterous hand with a DoF to accurately replicate human manipulations. Previous approaches that optimize in high-fidelity black-box simulators or a modified one with relaxed constraints only demonstrate limited capabilities or are restricted by insufficient simulation fidelity. We introduce parameterized quasi-physical simulators and a physics curriculum to overcome these limitations. The key ideas are 1) balancing between fidelity and optimizability of the simulation via a curriculum of parameterized simulators, and 2) solving the problem in each of the simulators from the curriculum, with properties ranging from high task optimizability to high fidelity. We successfully enable a dexterous hand to track complex and diverse manipulations in high-fidelity simulated environments, boosting the success rate by 11\%+ from the best-performed baseline. The project website is available at this https URL.

[Paper Link](https://arxiv.org/abs/2404.07988)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-12_11-19.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-11

### GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation

- **Authors**: Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi
- **Main Affiliations**: Georgia Institute of Technology, Carnegie Mellon University, University of Illinois Urbana-Champaign, Mistral AI, University of Washington
- **Tags**: `Large Language Models`

#### Abstract

The Embodied AI community has made significant strides in visual navigation tasks, exploring targets from 3D coordinates, objects, language descriptions, and images. However, these navigation models often handle only a single input modality as the target. With the progress achieved so far, it is time to move towards universal navigation models capable of handling various goal types, enabling more effective user interaction with robots. To facilitate this goal, we propose GOAT-Bench, a benchmark for the universal navigation task referred to as GO to AnyThing (GOAT). In this task, the agent is directed to navigate to a sequence of targets specified by the category name, language description, or image in an open-vocabulary fashion. We benchmark monolithic RL and modular methods on the GOAT task, analyzing their performance across modalities, the role of explicit and implicit scene memories, their robustness to noise in goal specifications, and the impact of memory in lifelong scenarios.

[Paper Link](https://arxiv.org/abs/2404.06609)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-11_16-29.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### GenCHiP: Generating Robot Policy Code for High-Precision and Contact-Rich Manipulation Tasks

- **Authors**: Kaylee Burns, Ajinkya Jain, Keegan Go, Fei Xia, Michael Stark, Stefan Schaal, Karol Hausman
- **Main Affiliations**: [Google] Intrinsic, Stanford University, Google DeepMind
- **Tags**: `Large Language Models`

#### Abstract

Large Language Models (LLMs) have been successful at generating robot policy code, but so far these results have been limited to high-level tasks that do not require precise movement. It is an open question how well such approaches work for tasks that require reasoning over contact forces and working within tight success tolerances. We find that, with the right action space, LLMs are capable of successfully generating policies for a variety of contact-rich and high-precision manipulation tasks, even under noisy conditions, such as perceptual errors or grasping inaccuracies. Specifically, we reparameterize the action space to include compliance with constraints on the interaction forces and stiffnesses involved in reaching a target pose. We validate this approach on subtasks derived from the Functional Manipulation Benchmark (FMB) and NIST Task Board Benchmarks. Exposing this action space alongside methods for estimating object poses improves policy generation with an LLM by greater than 3x and 4x when compared to non-compliant action spaces

[Paper Link](https://arxiv.org/abs/2404.06645)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-11_16-21.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models and Online Self-Supervision

- **Authors**: Matías Mattamala, Jonas Frey, Piotr Libera, Nived Chebrolu, Georg Martius, Cesar Cadena, Marco Hutter, Maurice Fallon
- **Main Affiliations**: University of Oxford, Robotic Systems Lab-ETH Zurich
- **Tags**: `Navigation`

#### Abstract

Natural environments such as forests and grasslands are challenging for robotic navigation because of the false perception of rigid obstacles from high grass, twigs, or bushes. In this work, we present Wild Visual Navigation (WVN), an online self-supervised learning system for visual traversability estimation. The system is able to continuously adapt from a short human demonstration in the field, only using onboard sensing and computing. One of the key ideas to achieve this is the use of high-dimensional features from pre-trained self-supervised models, which implicitly encode semantic information that massively simplifies the learning task. Further, the development of an online scheme for supervision generator enables concurrent training and inference of the learned model in the wild. We demonstrate our approach through diverse real-world deployments in forests, parks, and grasslands. Our system is able to bootstrap the traversable terrain segmentation in less than 5 min of in-field training time, enabling the robot to navigate in complex, previously unseen outdoor terrains. Code: this https URL - Project page:this https URL

[Paper Link](https://arxiv.org/abs/2404.07110)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-11_16-03.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-09

### Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models

- **Authors**: Yutao Ouyang, Jinhan Li, Yunfei Li, Zhongyu Li, Chao Yu, Koushil Sreenath, Yi Wu
- **Main Affiliations**: Shanghai Qizhi Institute, Tsinghua University, University of California, Berkeley
- **Tags**: `Large Language Models`

#### Abstract

We present a large language model (LLM) based system to empower quadrupedal robots with problem-solving abilities for long-horizon tasks beyond short-term motions. Long-horizon tasks for quadrupeds are challenging since they require both a high-level understanding of the semantics of the problem for task planning and a broad range of locomotion and manipulation skills to interact with the environment. Our system builds a high-level reasoning layer with large language models, which generates hybrid discrete-continuous plans as robot code from task descriptions. It comprises multiple LLM agents: a semantic planner for sketching a plan, a parameter calculator for predicting arguments in the plan, and a code generator to convert the plan into executable robot code. At the low level, we adopt reinforcement learning to train a set of motion planning and control skills to unleash the flexibility of quadrupeds for rich environment interactions. Our system is tested on long-horizon tasks that are infeasible to complete with one single skill. Simulation and real-world experiments show that it successfully figures out multi-step strategies and demonstrates non-trivial behaviors, including building tools or notifying a human for help.

[Paper Link](https://arxiv.org/abs/2404.05291)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-09_22-05.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer

- **Authors**: Xinyang Gu, Yen-Jen Wang, Jianyu Chen
- **Main Affiliations**: Shanghai Qizhi Institute, RobotEra, IIIS, Tsinghua University
- **Tags**: `Simulation to Reality`

#### Abstract

Humanoid-Gym is an easy-to-use reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment. Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujoco that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies. This framework is verified by RobotEra's XBot-S (1.2-meter tall humanoid robot) and XBot-L (1.65-meter tall humanoid robot) in a real-world environment with zero-shot sim-to-real transfer. The project website and source code can be found at: this https URL.

[Paper Link](https://arxiv.org/abs/2404.05695)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-09_21-00.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-05

### Self-supervised 6-DoF Robot Grasping by Demonstration via Augmented Reality Teleoperation System

- **Authors**: Xiwen Dengxiong, Xueting Wang, Shi Bai, Yunbo Zhang
- **Main Affiliations**: Rochester Institute of Technology
- **Tags**: `AR demonstrations`

#### Abstract

Most existing 6-DoF robot grasping solutions depend on strong supervision on grasp pose to ensure satisfactory performance, which could be laborious and impractical when the robot works in some restricted area. To this end, we propose a self-supervised 6-DoF grasp pose detection framework via an Augmented Reality (AR) teleoperation system that can efficiently learn human demonstrations and provide 6-DoF grasp poses without grasp pose annotations. Specifically, the system collects the human demonstration from the AR environment and contrastively learns the grasping strategy from the demonstration. For the real-world experiment, the proposed system leads to satisfactory grasping abilities and learning to grasp unknown objects within three demonstrations.

[Paper Link](https://arxiv.org/abs/2404.03067)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_14-08.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Embodied Neuromorphic Artificial Intelligence for Robotics: Perspectives, Challenges, and Research Development Stack

- **Authors**: Rachmad Vidya Wicaksana Putra, Alberto Marchisio, Fakhreddine Zayer, Jorge Dias, Muhammad Shafique
- **Main Affiliations**: New York University
- **Tags**: `Survey`

#### Abstract

Robotic technologies have been an indispensable part for improving human productivity since they have been helping humans in completing diverse, complex, and intensive tasks in a fast yet accurate and efficient way. Therefore, robotic technologies have been deployed in a wide range of applications, ranging from personal to industrial use-cases. However, current robotic technologies and their computing paradigm still lack embodied intelligence to efficiently interact with operational environments, respond with correct/expected actions, and adapt to changes in the environments. Toward this, recent advances in neuromorphic computing with Spiking Neural Networks (SNN) have demonstrated the potential to enable the embodied intelligence for robotics through bio-plausible computing paradigm that mimics how the biological brain works, known as "neuromorphic artificial intelligence (AI)". However, the field of neuromorphic AI-based robotics is still at an early stage, therefore its development and deployment for solving real-world problems expose new challenges in different design aspects, such as accuracy, adaptability, efficiency, reliability, and security. To address these challenges, this paper will discuss how we can enable embodied neuromorphic AI for robotic systems through our perspectives: (P1) Embodied intelligence based on effective learning rule, training mechanism, and adaptability; (P2) Cross-layer optimizations for energy-efficient neuromorphic computing; (P3) Representative and fair benchmarks; (P4) Low-cost reliability and safety enhancements; (P5) Security and privacy for neuromorphic computing; and (P6) A synergistic development for energy-efficient and robust neuromorphic-based robotics. Furthermore, this paper identifies research challenges and opportunities, as well as elaborates our vision for future research development toward embodied neuromorphic AI for robotics.

[Paper Link](https://arxiv.org/abs/2404.03325)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_14-00.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Robot Safety Monitoring using Programmable Light Curtains

- **Authors**: Karnik Ram, Shobhit Aggarwal, Robert Tamburo, Siddharth Ancha, Srinivasa Narasimhan
- **Main Affiliations**: Carnegie Mellon University, Technical University of Munich, MIT
- **Tags**: `Programmable Light Curtains`

#### Abstract

As factories continue to evolve into collaborative spaces with multiple robots working together with human supervisors in the loop, ensuring safety for all actors involved becomes critical. Currently, laser-based light curtain sensors are widely used in factories for safety monitoring. While these conventional safety sensors meet high accuracy standards, they are difficult to reconfigure and can only monitor a fixed user-defined region of space. Furthermore, they are typically expensive. Instead, we leverage a controllable depth sensor, programmable light curtains (PLC), to develop an inexpensive and flexible real-time safety monitoring system for collaborative robot workspaces. Our system projects virtual dynamic safety envelopes that tightly envelop the moving robot at all times and detect any objects that intrude the envelope. Furthermore, we develop an instrumentation algorithm that optimally places (multiple) PLCs in a workspace to maximize the visibility coverage of robots. Our work enables fence-less human-robot collaboration, while scaling to monitor multiple robots with few sensors. We analyze our system in a real manufacturing testbed with four robot arms and demonstrate its capabilities as a fast, accurate, and inexpensive safety monitoring solution.

[Paper Link](https://arxiv.org/abs/2404.03556)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-58.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity

- **Authors**: Jake Varley, Sumeet Singh, Deepali Jain, Krzysztof Choromanski, Andy Zeng, Somnath Basu Roy Chowdhury, Avinava Dubey, Vikas Sindhwani
- **Main Affiliations**: Google DeepMind, Google Research
- **Tags**: `Large Language Models`

#### Abstract

We present an embodied AI system which receives open-ended natural language instructions from a human, and controls two arms to collaboratively accomplish potentially long-horizon tasks over a large workspace. Our system is modular: it deploys state of the art Large Language Models for task planning,Vision-Language models for semantic perception, and Point Cloud transformers for grasping. With semantic and physical safety in mind, these modules are interfaced with a real-time trajectory optimizer and a compliant tracking controller to enable human-robot proximity. We demonstrate performance for the following tasks: bi-arm sorting, bottle opening, and trash disposal tasks. These are done zero-shot where the models used have not been trained with any real world data from this bi-arm robot, scenes or workspace.Composing both learning- and non-learning-based components in a modular fashion with interpretable inputs and outputs allows the user to easily debug points of failures and fragilities. One may also in-place swap modules to improve the robustness of the overall platform, for instance with imitation-learned policies.

[Paper Link](https://arxiv.org/abs/2404.03570)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-52.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-54.png" alt="Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Anticipate & Collab: Data-driven Task Anticipation and Knowledge-driven Planning for Human-robot Collaboration

- **Authors**: Shivam Singh, Karthik Swaminathan, Raghav Arora, Ramandeep Singh, Ahana Datta, Dipanjan Das, Snehasis Banerjee, Mohan Sridharan, Madhava Krishna
- **Main Affiliations**: Robotics Research Center, IIIT Hyderabad, India
- **Tags**: `Human-Robot Collaboration`, `Large Language Models`

#### Abstract

An agent assisting humans in daily living activities can collaborate more effectively by anticipating upcoming tasks. Data-driven methods represent the state of the art in task anticipation, planning, and related problems, but these methods are resource-hungry and opaque. Our prior work introduced a proof of concept framework that used an LLM to anticipate 3 high-level tasks that served as goals for a classical planning system that computed a sequence of low-level actions for the agent to achieve these goals. This paper describes DaTAPlan, our framework that significantly extends our prior work toward human-robot collaboration. Specifically, DaTAPlan planner computes actions for an agent and a human to collaboratively and jointly achieve the tasks anticipated by the LLM, and the agent automatically adapts to unexpected changes in human action outcomes and preferences. We evaluate DaTAPlan capabilities in a realistic simulation environment, demonstrating accurate task anticipation, effective human-robot collaboration, and the ability to adapt to unexpected changes. Project website: this https URL

[Paper Link](https://arxiv.org/abs/2404.03587)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-48_1.png" alt="Anticipate & Collab: Data-driven Task Anticipation and Knowledge-driven Planning for Human-robot Collaboration" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-48.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments

- **Authors**: Kairui Ding, Boyuan Chen, Ruihai Wu, Yuyang Li, Zongzheng Zhang, Huan-ang Gao, Siqi Li, Yixin Zhu, Guyue Zhou, Hao Dong, Hao Zhao
- **Main Affiliations**: Tsinghua University, Peking University, Zhejiang University
- **Tags**: `Affordance`

#### Abstract

Robotic manipulation of ungraspable objects with two-finger grippers presents significant challenges due to the paucity of graspable features, while traditional pre-grasping techniques, which rely on repositioning objects and leveraging external aids like table edges, lack the adaptability across object categories and scenes. Addressing this, we introduce PreAfford, a novel pre-grasping planning framework that utilizes a point-level affordance representation and a relay training approach to enhance adaptability across a broad range of environments and object types, including those previously unseen. Demonstrated on the ShapeNet-v2 dataset, PreAfford significantly improves grasping success rates by 69% and validates its practicality through real-world experiments. This work offers a robust and adaptable solution for manipulating ungraspable objects.

[Paper Link](https://arxiv.org/abs/2404.03634)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-05_13-32.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-04

### SliceIt! -- A Dual Simulator Framework for Learning Robot Food Slicing

- **Authors**: Cristian C. Beltran-Hernandez, Nicolas Erbetti, Masashi Hamaya
- **Main Affiliations**: OMRON SINIC X Corporation
- **Tags**: `Simulation to Reality`

#### Abstract

Cooking robots can enhance the home experience by reducing the burden of daily chores. However, these robots must perform their tasks dexterously and safely in shared human environments, especially when handling dangerous tools such as kitchen knives. This study focuses on enabling a robot to autonomously and safely learn food-cutting tasks. More specifically, our goal is to enable a collaborative robot or industrial robot arm to perform food-slicing tasks by adapting to varying material properties using compliance control. Our approach involves using Reinforcement Learning (RL) to train a robot to compliantly manipulate a knife, by reducing the contact forces exerted by the food items and by the cutting board. However, training the robot in the real world can be inefficient, and dangerous, and result in a lot of food waste. Therefore, we proposed SliceIt!, a framework for safely and efficiently learning robot food-slicing tasks in simulation. Following a real2sim2real approach, our framework consists of collecting a few real food slicing data, calibrating our dual simulation environment (a high-fidelity cutting simulator and a robotic simulator), learning compliant control policies on the calibrated simulation environment, and finally, deploying the policies on the real robot.

[Paper Link](https://arxiv.org/abs/2404.02569)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-04_17-52.png" alt="image" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### A Survey of Optimization-based Task and Motion Planning: From Classical To Learning Approaches

- **Authors**: Zhigen Zhao, Shuo Chen, Yan Ding, Ziyi Zhou, Shiqi Zhang, Danfei Xu, Ye Zhao
- **Main Affiliations**: Georgia Institute of Technology
- **Tags**: `Survey`, `TAMP`

#### Abstract

Task and Motion Planning (TAMP) integrates high-level task planning and low-level motion planning to equip robots with the autonomy to effectively reason over long-horizon, dynamic tasks. Optimization-based TAMP focuses on hybrid optimization approaches that define goal conditions via objective functions and are capable of handling open-ended goals, robotic dynamics, and physical interaction between the robot and the environment. Therefore, optimization-based TAMP is particularly suited to solve highly complex, contact-rich locomotion and manipulation problems. This survey provides a comprehensive review on optimization-based TAMP, covering (i) planning domain representations, including action description languages and temporal logic, (ii) individual solution strategies for components of TAMP, including AI planning and trajectory optimization (TO), and (iii) the dynamic interplay between logic-based task planning and model-based TO. A particular focus of this survey is to highlight the algorithm structures to efficiently solve TAMP, especially hierarchical and distributed approaches. Additionally, the survey emphasizes the synergy between the classical methods and contemporary learning-based innovations such as large language models. Furthermore, the future research directions for TAMP is discussed in this survey, highlighting both algorithmic and application-specific challenges.

[Paper Link](https://arxiv.org/abs/2404.02817)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-04_17-42.png" alt="image" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-04_17-42_1.png" alt="image" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Learning Quadrupedal Locomotion via Differentiable Simulation

- **Authors**: Clemens Schwarke, Victor Klemm, Jesus Tordesillas, Jean-Pierre Sleiman, Marco Hutter
- **Main Affiliations**: Robotic Systems Lab-ETH Zurich
- **Tags**: `Simulation`

#### Abstract

The emergence of differentiable simulators enabling analytic gradient computation has motivated a new wave of learning algorithms that hold the potential to significantly increase sample efficiency over traditional Reinforcement Learning (RL) methods. While recent research has demonstrated performance gains in scenarios with comparatively smooth dynamics and, thus, smooth optimization landscapes, research on leveraging differentiable simulators for contact-rich scenarios, such as legged locomotion, is scarce. This may be attributed to the discontinuous nature of contact, which introduces several challenges to optimizing with analytic gradients. The purpose of this paper is to determine if analytic gradients can be beneficial even in the face of contact. Our investigation focuses on the effects of different soft and hard contact models on the learning process, examining optimization challenges through the lens of contact simulation. We demonstrate the viability of employing analytic gradients to learn physically plausible locomotion skills with a quadrupedal robot using Short-Horizon Actor-Critic (SHAC), a learning algorithm leveraging analytic gradients, and draw a comparison to a state-of-the-art RL algorithm, Proximal Policy Optimization (PPO), to understand the benefits of analytic gradients.

[Paper Link](https://arxiv.org/abs/2404.02887)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-04_17-00.png" alt="image" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-03

### Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge

- **Authors**: Haoxiang Ma, Modi Shi, Boyang Gao, Di Huang
- **Main Affiliations**: Beihang University, Harbin Institute of Technology, Geometry Robotics
- **Tags**: `6-DoF`

#### Abstract

We focus on the generalization ability of the 6-DoF grasp detection method in this paper. While learning-based grasp detection methods can predict grasp poses for unseen objects using the grasp distribution learned from the training set, they often exhibit a significant performance drop when encountering objects with diverse shapes and structures. To enhance the grasp detection methods' generalization ability, we incorporate domain prior knowledge of robotic grasping, enabling better adaptation to objects with significant shape and structure differences. More specifically, we employ the physical constraint regularization during the training phase to guide the model towards predicting grasps that comply with the physical rule on grasping. For the unstable grasp poses predicted on novel objects, we design a contact-score joint optimization using the projection contact map to refine these poses in cluttered scenarios. Extensive experiments conducted on the GraspNet-1billion benchmark demonstrate a substantial performance gain on the novel object set and the real-world grasping experiments also demonstrate the effectiveness of our generalizing 6-DoF grasp detection method.

[Paper Link](https://arxiv.org/abs/2404.01727)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-08_17-45.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

## 2024-04-01

### Snap-it, Tap-it, Splat-it: Tactile-Informed 3D Gaussian Splatting for Reconstructing Challenging Surfaces

- **Authors**: Mauro Comi, Alessio Tonioni, Max Yang, Jonathan Tremblay, Valts Blukis, Yijiong Lin, Nathan F. Lepora, Laurence Aitchison
- **Main Affiliations**: University of Bristol, Google Zurich, NVIDIA
- **Tags**: `tactile`, `Gaussian Splatting`

#### Abstract

Touch and vision go hand in hand, mutually enhancing our ability to understand the world. From a research perspective, the problem of mixing touch and vision is underexplored and presents interesting challenges. To this end, we propose Tactile-Informed 3DGS, a novel approach that incorporates touch data (local depth maps) with multi-view vision data to achieve surface reconstruction and novel view synthesis. Our method optimises 3D Gaussian primitives to accurately model the object's geometry at points of contact. By creating a framework that decreases the transmittance at touch locations, we achieve a refined surface reconstruction, ensuring a uniformly smooth depth map. Touch is particularly useful when considering non-Lambertian objects (e.g. shiny or reflective surfaces) since contemporary methods tend to fail to reconstruct with fidelity specular highlights. By combining vision and tactile sensing, we achieve more accurate geometry reconstructions with fewer images than prior methods. We conduct evaluation on objects with glossy and reflective surfaces and demonstrate the effectiveness of our approach, offering significant improvements in reconstruction quality.

[Paper Link](https://arxiv.org/abs/2403.20275)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-08_17-31.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---

### Learning Visual Quadrupedal Loco-Manipulation from Demonstrations

- **Authors**: Zhengmao He, Kun Lei, Yanjie Ze, Koushil Sreenath, Zhongyu Li, Huazhe Xu
- **Main Affiliations**: Shanghai Qizhi Institute, Hong Kong University of Science and Technology, University of California, Berkeley, IIIS, Tsinghua University
- **Tags**: `Behavioral Cloning`, `Reinforcement Learning`

#### Abstract

Quadruped robots are progressively being integrated into human environments. Despite the growing locomotion capabilities of quadrupedal robots, their interaction with objects in realistic scenes is still limited. While additional robotic arms on quadrupedal robots enable manipulating objects, they are sometimes redundant given that a quadruped robot is essentially a mobile unit equipped with four limbs, each possessing 3 degrees of freedom (DoFs). Hence, we aim to empower a quadruped robot to execute real-world manipulation tasks using only its legs. We decompose the loco-manipulation process into a low-level reinforcement learning (RL)-based controller and a high-level Behavior Cloning (BC)-based planner. By parameterizing the manipulation trajectory, we synchronize the efforts of the upper and lower layers, thereby leveraging the advantages of both RL and BC. Our approach is validated through simulations and real-world experiments, demonstrating the robot's ability to perform tasks that demand mobility and high precision, such as lifting a basket from the ground while moving, closing a dishwasher, pressing a button, and pushing a door. Project website: this https URL

[Paper Link](https://arxiv.org/abs/2403.20328)

[Website Link](https://zhengmaohe.github.io/leg-manip)

<div style={{ display: 'flex', justifyContent: 'center' }}>
<div style={{ textAlign: 'center', marginRight: '10px' }}>
<img src="/img/daily/2024-04-08_17-20.png" alt="img" style={{ width: 'auto', maxHeight: '400px' }} />
</div>
</div>

---
