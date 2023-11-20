import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import Image from '@theme/IdealImage';
import FavoriteIcon from '@site/src/components/svgIcons/FavoriteIcon';
import { Paper } from '@site/src/shared/dto/Paper';
import { sortBy } from '@site/src/utils/jsUtils';
import Heading from '@theme/Heading';
import Tooltip from '../ShowcaseTooltip';
import styles from './styles.module.css';
import { PaperTagList, PaperTagType, PaperTags } from '@site/src/shared/constants/PaperConsts';
import { Tag } from '@site/src/shared/dto/Tag';
import { GetWebsiteScreenshot } from '@site/src/utils/common';

const SIMULATELY = require('@site/static/img/teaser.png');

const TagComp = React.forwardRef<HTMLLIElement, Tag>(
  ({ label, color, description }, ref) => (
    <li ref={ref} className={styles.tag} title={description}>
      <span className={styles.textLabel}>{label.toLowerCase()}</span>
      <span className={styles.colorLabel} style={{ backgroundColor: color }} />
    </li>
  ),
);

function ShowcaseCardTag({ tags }: { tags: PaperTagType[] }) {
  const tagObjects = tags.map((tag) => ({ tag, ...PaperTags[tag] }));

  // Keep same order for all tags
  const tagObjectsSorted = sortBy(tagObjects, (tagObject) =>
    PaperTagList.indexOf(tagObject.tag),
  );

  return (
    <>
      {tagObjectsSorted.map((tagObject, index) => {
        const id = `showcase_card_tag_${tagObject.tag}`;
        return (
          <Tooltip key={index} id={id} text={tagObject.description} anchorEl="#__docusaurus">
            <TagComp key={index} {...tagObject} />
          </Tooltip>
        );
      })}
    </>
  );
}

/**
 * Returns the image url for the card, either the user-provided one or a screenshot of the website
 * Ref: https://api-explorer.11ty.dev/
 * @param user user object
 * @returns image url
 */
function getCardImage(user: Paper): string {
  if (user.preview) {
    return user.preview;
  }
  if (user.website) {
    var img = user.website ?? 'https://github.com/Simulately'
    return user.preview ?? GetWebsiteScreenshot(img)
  }
  return SIMULATELY;
}

function ShowcaseCard({ user }: { user: Paper }) {
  const image = getCardImage(user);
  return (
    <li key={user.title} className="card shadow--md">
      <div className={clsx('card__image', styles.showcaseCardImage)}>
        {/* Vertically fill */}
        <Image img={image} alt={user.title} loading='lazy' about={user.title} decoding='async' style={{ objectFit: 'cover', height: '100%' }} />
      </div>

      <div className="card__body">
        <div className={clsx(styles.showcaseCardHeader)}>
          <Heading as="h4" className={styles.showcaseCardTitle}>
            <Link href={user.website} className={styles.showcaseCardLink}>
              {user.title}
            </Link>
          </Heading>
        </div>

        <p className={styles.showcaseCardBody}>{user.description}</p>
      </div>

      <ul className={clsx('card__footer', styles.cardFooter)}>
        <ShowcaseCardTag tags={user.tags} />
      </ul>
    </li>
  );
}

export default React.memo(ShowcaseCard);
