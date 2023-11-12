/* eslint-disable global-require */

import { sortBy } from '@site/src/utils/jsUtils';
import { Paper } from '@site/src/shared/dto/Paper';

// prettier-ignore
const PAPERS: Paper[] = [
  {
    title: 'UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://pku-epic.github.io/UniDexGrasp/',
    tags: ['rl', 'rgbd_pc'],
  },
  {
    title: 'GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://github.com/PKU-EPIC/GAPartNet',
    tags: ['rl', 'articulated', 'rgbd_pc'],
  },
  {
    title: 'UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning',
    description: 'ICCV 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://pku-epic.github.io/UniDexGrasp++/',
    tags: ['rl', 'dexterousgrasping', 'rgbd_pc'],
  },
  {
    title: 'GenDexGrasp: Generalizable Dexterous Grasping',
    description: 'ICRA 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://sites.google.com/view/gendexgrasp/',
    tags: ['rl', 'dexterousgrasping', 'rgbd_pc'],
  },
  {
    title: 'ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes',
    description: 'ICCV 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://arnold-benchmark.github.io/',
    tags: ['manipulation', 'rgbd_pc'],
  },
  {
    title: 'RLAfford: End-to-End Affordance Learning for Robotic Manipulation',
    description: 'ICRA 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://sites.google.com/view/rlafford/',
    tags: ['rl', 'manipulation', 'articulated', 'rgbd_pc', 'affordance'],
  },
  {
    title: 'PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://github.com/PKU-EPIC/PartManip',
    tags: ['rl', 'dexterousgrasping', 'articulated'],
  },
  {
    title: 'Grasp Multiple Objects with One Hand',
    description: 'RGB-D point cloud.',
    preview: null,
    website: 'https://multigrasp.github.io/',
    tags: ['rgbd_pc', 'rl', 'dexterousgrasping'],
  },
  {
    title: 'Dynamic Handover: Throw and Catch with Bimanual Hands',
    description: 'CoRL2023.',
    preview: null,
    website: 'https://binghao-huang.github.io/dynamic_handover/',
    tags: ['rl'],
  },
  {
    title: 'Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation',
    description: 'CoRL2023.',
    preview: null,
    website: 'https://sequential-dexterity.github.io/',
    tags: ['rl'],
  },
  {
    title: 'Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks',
    description: 'CORL2023.',
    preview: null,
    website: 'https://openreview.net/forum?id=QG_ERxtDAP-&referrer=%5Bthe%20profile%20of%20Clemens%20Schwarke%5D(%2Fprofile%3Fid%3D~Clemens_Schwarke1',
    tags: ['rl'],
  },
  {
    title: 'General In-Hand Object Rotation with Vision and Touch',
    description: 'CoRL2023; RGB-D.',
    preview: null,
    website: 'https://haozhi.io/rotateit/',
    tags: ['rl'],
  },
  {
    title: 'DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training',
    description: 'RSS2023.',
    preview: null,
    website: 'https://sites.google.com/view/dexpbt',
    tags: ['rl'],
  },
  {
    title: 'Rotating without Seeing: Towards In-hand Dexterity through Touch',
    description: 'RSS2023.',
    preview: null,
    website: 'https://touchdexterity.github.io/',
    tags: ['rl'],
  },
  {
    title: 'In-Hand Object Rotation via Rapid Motor Adaptation',
    description: 'CoRL2022.',
    preview: null,
    website: 'https://haozhi.io/hora/',
    tags: ['rl'],
  },
  {
    title: 'Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learnin',
    description: 'NIPS2022.',
    preview: null,
    website: 'https://bi-dexhands.ai/',
    tags: ['rgbd_pc', 'rl'],
  },
  {
    title: 'Data-Driven Operational Space Control for Adaptative and Robust Robot Manipulation',
    description: 'ICRA2022.',
    preview: null,
    website: 'https://github.com/nvlabs/oscar',
    tags: ['rgbd_pc', 'rl', 'dexterousgrasping'],
  },
  {
    title: 'Factory: Fast contact for robotic assembly',
    description: 'RSS2022.',
    preview: null,
    website: 'https://sites.google.com/nvidia.com/factory',
    tags: ['rl'],
  },
  {
    title: 'ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters',
    description: 'SIGGRAPH2022.',
    preview: null,
    website: 'https://nv-tlabs.github.io/ASE/',
    tags: ['rl'],
  },
  {
    title: 'STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation',
    description: 'CoRL2021.',
    preview: null,
    website: 'https://github.com/NVlabs/storm',
    tags: ['rl'],
  },
  {
    title: 'Causal Reasoning in Simulationfor Structure and Transfer Learning of Robot Manipulation Policies',
    description: 'ICRA2021.',
    preview: null,
    website: 'https://sites.google.com/view/crest-causal-struct-xfer-manip',
    tags: ['rgbd_pc', 'rl', 'dexterousgrasping', 'rgbd_pc'],
  },
  {
    title: 'In-Hand Object Pose Tracking via Contact Feedback and GPU-Accelerated Robotic Simulation',
    description: 'ICRA2021.',
    preview: null,
    website: 'https://sites.google.com/view/in-hand-object-pose-tracking/',
    tags: ['rl'],
  },
  {
    title: 'Reactive Long Horizon Task Execution via Visual Skill and Precondition Models',
    description: 'IROS2021.',
    preview: null,
    website: 'https://arxiv.org/pdf/2011.08694.pdf',
    tags: ['rl'],
  },
  {
    title: 'Sim-to-Real for Robotic Tactile Sensing via Physics-Based Simulation and Learned Latent Projections',
    description: 'ICRA2021.',
    preview: null,
    website: 'https://arxiv.org/pdf/2103.16747.pdf',
    tags: ['rl'],
  },
  {
    title: 'A Simple Method for Complex In-Hand Manipulation',
    description: 'RSS2021_VLRR.',
    preview: null,
    website: 'https://sites.google.com/view/in-hand-reorientation',
    tags: ['rl'],
  },
  {
    title: 'Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning',
    description: 'CoRL2021.',
    preview: null,
    website: 'https://leggedrobotics.github.io/legged_gym/',
    tags: ['rl'],
  },
  {
    title: 'Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning',
    description: 'NIPS2021.',
    preview: null,
    website: 'https://sites.google.com/view/isaacgym-nvidia',
    tags: ['rl'],
  },
  {
    title: 'Dynamics Randomization Revisited:A Case Study for Quadrupedal Locomotion',
    description: 'ICRA2021.',
    preview: null,
    website: 'https://www.pair.toronto.edu/understanding-dr/',
    tags: ['rl'],
  },
  {
    title: 'Learning a State Representation and Navigation in Cluttered and Dynamic Environments',
    description: 'RAL2021.',
    preview: null,
    website: 'https://arxiv.org/pdf/2103.04351.pdf',
    tags: ['rl'],
  },
  {
    title: 'Learning to Compose Hierarchical Object-Centric Controllers for Robotic Manipulation',
    description: 'CoRL2020.',
    preview: null,
    website: 'https://sites.google.com/view/compositional-object-control/',
    tags: ['rl'],
  },
  {
    title: 'Learning a Contact-Adaptive Controller for Robust, Efficient Legged Locomotion',
    description: 'CoRL2020.',
    preview: null,
    website: 'https://sites.google.com/view/learn-contact-controller/home',
    tags: ['rl'],
  },
  {
    title: 'Learning Active Task-Oriented Exploration Policies for Bridging the Sim-to-Real Gap',
    description: 'RSS2020.',
    preview: null,
    website: 'https://sites.google.com/view/task-oriented-exploration/',
    tags: ['rl'],
  },
  {
    title: 'Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience',
    description: 'ICRA2019.',
    preview: null,
    website: 'https://sites.google.com/view/simopt',
    tags: ['rl'],
  },
  {
    title: 'GPU-Accelerated Robotics Simulation for Distributed Reinforcement Learning',
    description: 'CoRL2018.',
    preview: null,
    website: 'https://sites.google.com/view/accelerated-gpu-simulation/home',
    tags: ['rl'],
  },
  {
    title: 'On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline',
    description: 'ICML2021; RGB.',
    preview: null,
    website: 'https://arxiv.org/abs/2212.05749',
    tags: ['rl'],
  },
  {
    title: 'AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System',
    description: 'RSS2023.',
    preview: null,
    website: 'http://anyteleop.com/',
    tags: ['rl'],
  }
];

function sortPapers() {
  let result = PAPERS;
  // Sort by site name
  result = sortBy(result, (paper) => paper.title.toLowerCase());
  
  // Sort by favorite tag, favorites first
  result = sortBy(result, (paper) => !paper.tags.includes('rl'));
  return result;
}

export const sortedPapers = sortPapers();
