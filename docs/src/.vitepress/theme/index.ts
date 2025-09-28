// .vitepress/theme/index.ts
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'

import {
  NolebaseEnhancedReadabilitiesMenu,
  NolebaseEnhancedReadabilitiesScreenMenu,
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from '@/VersionPicker.vue'
import StarUs from '@/StarUs.vue'
import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css'
import './docstrings.css'

import markdownItContainer from 'markdown-it-container'

// Capitalize the container type for prefix
function capitalize(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1)
}

export const Theme: ThemeConfig = {
  extends: DefaultTheme,

  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => [
        h(StarUs),
        h(NolebaseEnhancedReadabilitiesMenu),
      ],
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },

  enhanceApp({ app, router, siteData, page, markdown }) {
    enhanceAppWithTabs(app)
    app.component('VersionPicker', VersionPicker)

    // Add markdown-it container plugin and override renderer
    markdown.config = (md) => {
      // Register some common containers to enable parsing
      md.use(markdownItContainer, 'tip')
      md.use(markdownItContainer, 'note')
      md.use(markdownItContainer, 'info')
      md.use(markdownItContainer, 'warning')
      md.use(markdownItContainer, 'danger')
      md.use(markdownItContainer, 'todo')
      md.use(markdownItContainer, 'definition')
      md.use(markdownItContainer, 'property')
      md.use(markdownItContainer, 'remark')
      md.use(markdownItContainer, 'theorem')

      const defaultRender = md.renderer.rules.container || function (tokens, idx, options, env, self) {
        return self.renderToken(tokens, idx, options)
      }

      md.renderer.rules.container = (tokens, idx, options, env, self) => {
        const token = tokens[idx]

        if (token.nesting === 1) {
          // opening tag
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

          return `<div class="custom-block custom-block-${type}">\n` +
                 `<div class="custom-block-title">${fullTitle}</div>\n` +
                 `<div class="custom-block-content">\n`
        }

        if (token.nesting === -1) {
          // closing tag
          return `</div>\n</div>\n`
        }

        return defaultRender(tokens, idx, options, env, self)
      }
    }
  }
}

export default Theme
