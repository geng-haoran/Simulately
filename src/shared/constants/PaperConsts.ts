import { translate } from '@docusaurus/Translate';
import { Tag } from '@site/src/shared/dto/Tag';

export type PaperTagType =
| 'dataset'
    | 'rl'
    | 'affordance'
    | 'rgbd_pc'
    | 'articulated'
    | 'manipulation'
    | 'dexterousgrasping'
    | 'mobilemanip'
    | 'rgb';

    export const PaperTags: { [type in PaperTagType]: Tag } = {
    "dataset": {
        label: translate({ message: 'Dataset' }),
        description: translate({
            message: 
                'Releasing a dataset',
            id: 'showcase.tag.dataset.description',
        }),
        color: '#e9669e',
    },
    "rl": {
        label: translate({ message: 'Reinforcement Learning' }),
        description: translate({
            message: 
                'Learning with RL',
            id: 'showcase.tag.rl.description',
        }),
        color: '#e9669e',
    },
    "affordance": {
        label: translate({ message: 'Affordance' }),
        description: translate({
            message: 
                'The secret ingredient from J J Gibson',
            id: 'showcase.tag.affordance.description',
        }),
        color: '#e9669e',
    },
    "rgbd_pc": {
        label: translate({ message: 'RGB-D Point Cloud' }),
        description: translate({
            message: 
                'Perceiving RGB-D poing clouds',
            id: 'showcase.tag.rgbd_pc.description',
        }),
        color: '#e9669e',
    },
    "rgb": {
        label: translate({ message: 'RGB' }),
        description: translate({
            message: 
                'Perceiving RGB images',
            id: 'showcase.tag.rgb.description',
        }),
        color: '#e9669e',
    },
    "articulated": {
        label: translate({ message: 'Articulated Object' }),
        description: translate({
            message: 
                'Interacting with articulated objects',
            id: 'showcase.tag.articulated.description',
        }),
        color: '#e9669e',
    },
    "manipulation": {
        label: translate({ message: 'Manipulation' }),
        description: translate({
            message: 
                'Robot manipulation',
            id: 'showcase.tag.manipulation.description',
        }),
        color: '#e9669e',
    },
    "dexterousgrasping": {
        label: translate({ message: 'Dexterous Grasping' }),
        description: translate({
            message: 
                'Grasping with dexterity',
            id: 'showcase.tag.dexterousgrasping.description',
        }),
        color: '#e9669e',
    },
    "mobilemanip": {
        label: translate({ message: 'Mobile Manipulator' }),
        description: translate({
            message: 
                'Mobile manipulators',
            id: 'showcase.tag.mobilemanip.description',
        }),
        color: '#e9669e',
    }
};

export const PaperTagList = Object.keys(PaperTags) as PaperTagType[];
