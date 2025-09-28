import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3"
import footnote from "markdown-it-footnote"
import markdownItContainer from 'markdown-it-container'
import path from 'path'

// console.log(process.env)

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker',
  }
]

function capitalize(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1)
}

export default defineConfig({
  ignoreDeadLinks: true,
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', { src: `${getBaseRepository(baseTemp.base)}versions.js` }],
    ['script', { src: `${baseTemp.base}siteinfo.js` }]
  ],
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    build: {
      assetsInlineLimit: 0, // so we can tell whether we have created inlined images or not, we don't let vite inline them
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin)
      md.use(mathjax3)
      md.use(footnote)

      // **our container setup**
      const types = [
        'tip',
        'note',
        'info',
        'warning',
        'danger',
        'todo',
        'definition',
        'property',
        'remark',
        'theorem'
      ]
      for (const t of types) {
        md.use(markdownItContainer, t)
      }

      const defaultRender = md.renderer.rules.container || ((tokens, idx, opts, env, self) =>
        self.renderToken(tokens, idx, opts)
      )

      md.renderer.rules.container = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        if (token.nesting === 1) {
          const info = token.info.trim()
          const firstSpaceIndex = info.indexOf(' ')
          let type = ''
          let title = ''
          if (firstSpaceIndex === -1) {
            type = info
            title = ''
          } else {
            type = info.slice(0, firstSpaceIndex)
            title = info.slice(firstSpaceIndex + 1)
          }
          const typeCapitalized = capitalize(type)
          const fullTitle = title ? `${typeCapitalized}: ${title}` : typeCapitalized

          return `<div class="custom-block ${type}">\n` +
                 `<div class="custom-block-title">${fullTitle}</div>\n` +
                 `<div class="custom-block-content">\n`
        }
        if (token.nesting === -1) {
          return `</div>\n</div>\n`
        }
        return defaultRender(tokens, idx, options, env, self)
      }
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  themeConfig: {
    outline: 'deep',
    // https://vitepress.dev/reference/default-theme-config
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        prefix: true,
        boost: { title: 100 },
        fuzzy: 2,
      }
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'slack', link: 'https://julialang.org/slack/' }
    ],
    footer: {
      message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable/" target="_blank"><strong>DocumenterVitepress.jl</strong></a> <br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
  }
})
