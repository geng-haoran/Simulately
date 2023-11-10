/* eslint-disable global-require */

import { sortBy } from '@site/src/utils/jsUtils';
import { Paper } from '@site/src/shared/dto/Paper';

// prettier-ignore
const PAPERS: Paper[] = [
  {
    title: 'UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: "https://arnold-benchmark.github.io/assets/teaser.png",
    website: 'https://pku-epic.github.io/UniDexGrasp/',
    source: 'https://pku-epic.github.io/UniDexGrasp/',
    tags: ['isaacgym'],
  },
  {
    title: 'GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: "https://pku-epic.github.io/GAPartNet/images/teaser.png",
    website: 'https://github.com/PKU-EPIC/GAPartNet',
    source: 'https://github.com/PKU-EPIC/GAPartNet',
    tags: ['sapien', 'articulated'],
  },
  {
    title: 'UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning',
    description: 'ICCV 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://pku-epic.github.io/UniDexGrasp++/',
    source: 'https://pku-epic.github.io/UniDexGrasp++/',
    tags: ['isaacgym', 'dexterousgrasping'],
  },
  {
    title: 'GenDexGrasp: Generalizable Dexterous Grasping',
    description: 'ICRA 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://sites.google.com/view/gendexgrasp/',
    source: 'https://sites.google.com/view/gendexgrasp/',
    tags: ['isaacgym', 'dexterousgrasping'],
  },
  {
    title: 'ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes',
    description: 'ICCV 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://arnold-benchmark.github.io/',
    source: 'https://arnold-benchmark.github.io/',
    tags: ['isaacsim', 'manipulation'],
  },
  {
    title: 'RLAfford: End-to-End Affordance Learning for Robotic Manipulation',
    description: 'ICRA 2023; RGB-D point cloud.',
    preview: null,
    website: 'https://sites.google.com/view/rlafford/',
    source: 'https://sites.google.com/view/rlafford/',
    tags: ['isaacgym', 'manipulation', 'articulated'],
  },
  {
    title: 'PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations',
    description: 'CVPR 2023; RGB-D point cloud.',
    preview: "https://pku-epic.github.io/PartManip/images/teaser_img.png",
    website: 'https://github.com/PKU-EPIC/PartManip',
    source: 'https://github.com/PKU-EPIC/PartManip',
    tags: ['isaacsim', 'dexterousgrasping', 'articulated'],
  },
  {
    title: 'Grasp Multiple Objects with One Hand',
    description: 'RGB-D point cloud.',
    preview: "https://multigrasp.github.io/static/images/teaser/Teaser.png",
    website: 'https://multigrasp.github.io/',
    source: 'https://multigrasp.github.io/',
    tags: ['isaacsim', 'dexterousgrasping'],
  }
];

function sortPapers() {
  let result = PAPERS;
  // Sort by site name
  result = sortBy(result, (paper) => paper.title.toLowerCase());
  
  // Sort by favorite tag, favorites first
  result = sortBy(result, (paper) => !paper.tags.includes('isaacgym'));
  return result;
}

export const sortedPapers = sortPapers();
