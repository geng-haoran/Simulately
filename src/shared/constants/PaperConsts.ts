import { translate } from '@docusaurus/Translate';
import { Tag } from '@site/src/shared/dto/Tag';

export type PaperTagType =
    | 'isaacgym'
    // For open-source sites, a link to the source code is required.
    // The source should be the *website's* source, not the project's source!
    | 'isaacsim'
    | 'sapien'
    | 'pybullet'
    | 'mujoco';

export const PaperTags: { [type in PaperTagType]: Tag } = {
    isaacgym: {
        label: translate({ message: 'Isaac Gym' }),
        description: translate({
            message: 
                'Related Work',
            id: 'showcase.tag.isaacgym.description',
        }),
        color: '#e9669e',
    }
};

export const PaperTagList = Object.keys(PaperTags) as PaperTagType[];
