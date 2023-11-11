import { translate } from '@docusaurus/Translate';
import { Tag } from '@site/src/shared/dto/Tag';

export type PaperTagType =
    | 'isaacgym'
    | 'isaacsim'
    | 'sapien'
    | 'pybullet'
    | 'mujoco'
    | 'articulated'
    | 'manipulation'
    | 'dexterousgrasping';

export const PaperTags: { [type in PaperTagType]: Tag } = {
    "isaacgym": {
        label: translate({ message: 'IsaacGym' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.isaacgym.description',
        }),
        color: '#e9669e',
    },
    "isaacsim": {
        label: translate({ message: 'IsaacSim' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.isaacsim.description',
        }),
        color: '#e9669e',
    },
    "sapien": {
        label: translate({ message: 'SAPIEN' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.sapien.description',
        }),
        color: '#e9669e',
    },
    "pybullet": {
        label: translate({ message: 'PyBullet' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.pybullet.description',
        }),
        color: '#e9669e',
    },
    "mujoco": {
        label: translate({ message: 'Isaac Gym' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.mujoco.description',
        }),
        color: '#e9669e',
    },
    "articulated": {
        label: translate({ message: 'Articulated Object' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.articulated.description',
        }),
        color: '#e9669e',
    },
    "manipulation": {
        label: translate({ message: 'Manipulation' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.manipulation.description',
        }),
        color: '#e9669e',
    },
    "dexterousgrasping": {
        label: translate({ message: 'Dexterous Grasping' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.dexterousgrasping.description',
        }),
        color: '#e9669e',
    }
};

export const PaperTagList = Object.keys(PaperTags) as PaperTagType[];
