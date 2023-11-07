/* eslint-disable global-require */

import { sortBy } from '@site/src/utils/jsUtils';
import { Paper } from '@site/src/shared/dto/Paper';

// prettier-ignore
const PAPERS: Paper[] = [
  {
    title: 'UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy',
    description: 'CVPR 2023',
    preview: null,
    website: 'https://pku-epic.github.io/UniDexGrasp/',
    source: 'https://pku-epic.github.io/UniDexGrasp/',
    tags: ['isaacgym'],
  },
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
