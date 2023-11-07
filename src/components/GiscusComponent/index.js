import React from 'react';
import Giscus from "@giscus/react";
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// Ref: https://giscus.app/vi

export default function GiscusComponent() {
  const { colorMode } = useColorMode();

  // Using for access process env variables
  const {
    siteConfig: { customFields },
  } = useDocusaurusContext();

  return (
    <Giscus
      repo={customFields.GITHUB_REPO}
      repoId={customFields.GITHUB_REPO_ID}
      category="General"
      categoryId={customFields.GITHUB_CATEGORY_ID}
      mapping="title"
      term="Discussion"
      strict="1"
      reactionsEnabled="1"
      emitMetadata="1"
      inputPosition="top"
      theme={colorMode}
      lang="vi"
      loading="lazy"
      id='giscus'
    />
  );
}