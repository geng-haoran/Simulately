// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Simulately',
  tagline: ' Handy information and resources for physics simulators for robot learning research.',
  favicon: 'img/logo.png',

  // Set the production url of your site here
  url: 'https://simulately.wiki',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'haoran-geng', // Usually your GitHub org/user name.
  projectName: 'Simulately', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X',
      crossorigin: 'anonymous',
    },
  ],
  
  themeConfig:
  /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    {
      // Searching
      algolia: {
        appId: '81A7HEMAGK',
        apiKey: 'abf40978712646a81b2212726a3ee6e7',
        indexName: 'simulately',
        contextualSearch: false,
        searchParameters: {},
        searchPagePath: 'search'
      },

      image: 'img/social.png',
      navbar: {
        title: 'Simulately',
        logo: {
          alt: 'Simulately Logo',
          src: 'img/logo.png',
        },
        items: [
          {type: 'docSidebar', sidebarId: 'tutorialSidebar', position: 'left', label: 'Wiki'},
          {to: '/blog', label: 'Blog', position: 'left'},
          {to: '/related', label: 'Related Work', position: 'left'},
          {to: '/gpt/gpt', label: 'Simulately GPT', position: 'left'},
          {href: 'https://github.com/geng-haoran/Simulately', label: 'GitHub', position: 'right'},
          {href: 'https://chat.openai.com/g/g-cjN7iYpRZ-simulately', label: 'Ask GPT', position: 'right'}
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Community',
            items: [
              {
                label: 'BIGAI',
                href: 'http://www.bigai.ai/',
              },
              {
                label: 'PKU EPIC Lab',
                href: 'https://pku-epic.github.io/',
              },
              {
                label: 'PKU CoRe Lab',
                href: 'https://pku.ai/',
              },
              {
                label: 'UCLA VCLA',
                href: 'https://vcla.stat.ucla.edu/',
              },
            ],
          },
          {
            title: 'Powered By',
            items: [
              {
                label: 'Docusaurus',
                href: 'https://docusaurus.io/',
              },
              {
                label: 'Cloudflare',
                href: 'https://www.cloudflare.com/',
              },
            ],
          },
          {
            title: 'About Us',
            items: [
              {
                label: 'X (Twitter)',
                href: 'https://x.com/simulately12492',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/geng-haoran/Simulately',
              },
              {
                label: 'E-Mail',
                href: 'mailto:ghr@stu.pku.edu.cn',
              },
            ],
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Simulately. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    },

  plugins: ['@docusaurus/plugin-ideal-image'],
};

module.exports = config;
