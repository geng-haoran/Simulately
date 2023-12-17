import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <span className={styles.indexCtasGitHubButtonWrapper}>
          <iframe
            className={styles.indexCtasGitHubButton}
            src="https://ghbtns.com/github-btn.html?user=geng-haoran&amp;repo=Simulately&amp;type=star&amp;count=true&amp;size=large"
            width={120}
            height={30}
            title="GitHub Stars"
          />
        </span>
        <h1 className="hero__title">Welcome to {siteConfig.title}</h1>
        <p className="hero__subtitle">🦾{siteConfig.tagline}</p>
        <div className={styles.buttons}>
        <Link
            className="button button--secondary button--lg"
            to="/docs">
            Simulate Like a Pro 👉
        </Link>
        </div>
        <br/>
        <div className={styles.buttons}>
        <Link
            className="button button--secondary button--lg"
            to="/gpt/gpt">
            SimulatelyGPT 🧠
        </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Simulately`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
