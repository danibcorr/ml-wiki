import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");
const organizationName = "danibcorr";
const projectName = "ml-wiki";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Daniel Bazo",
  url: `https://${organizationName}.github.io`,
  baseUrl: `/${projectName}/`,
  onBrokenLinks: "warn",
  favicon: "img/logo.ico",
  trailingSlash: true,
  organizationName,
  projectName,
  deploymentBranch: "gh-pages",
  i18n: {
    defaultLocale: "es",
    locales: ["es"],
    localeConfigs: {
      es: {
        label: "Español",
        direction: "ltr",
        htmlLang: "es",
      },
    },
  },

  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          breadcrumbs: false,
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      },
    ],
  ],

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
  ],

  themeConfig: {
    metadata: [
      {
        name: "keywords",
        content:
          "machine learning, deep learning, artificial intelligence, python programming, data science, neural networks, MLOps, software engineering, AI blog, tech tutorials, programming guides, model deployment, data engineering",
      },
      {
        name: "description",
        content:
          "Comprehensive Machine Learning Engineering wiki featuring tutorials, best practices, and insights on AI, deep learning, Python programming, and modern software development techniques.",
      },
      {
        name: "author",
        content: "Daniel Bazo Correa",
      },
    ],

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },

    colorMode: {
      disableSwitch: false,
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: "Daniel Bazo Correa",
      logo: {
        alt: "Logo",
        src: "img/logo motivo.svg",
        srcDark: "img/logo motivo dark.svg",
        href: "https://danibcorr.github.io/web/",
      },
      items: [
        {
          type: "doc",
          docId: "intro",
          label: "Wiki",
        },
        {
          type: "localeDropdown",
          position: "right",
        },
        {
          href: "https://github.com/danibcorr/ml-wiki",
          position: "right",
          className: "header-github-link",
          "aria-label": "GitHub repository",
        },
      ],
      hideOnScroll: true,
    },

    footer: {
      copyright: `Copyright © ${new Date().getFullYear()} Daniel Bazo Correa`,
    },

    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ["bash", "makefile"],
    },
  },
};

module.exports = config;
