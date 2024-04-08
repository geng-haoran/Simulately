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
