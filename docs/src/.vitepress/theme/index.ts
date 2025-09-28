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

import { defineContainer } from 'vitepress'

// Capitalize the container type for prefix
function capitalize(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1)
}

// Register a universal container renderer that matches all container types
const universalContainer = defineContainer({
  name: /.*/, // Match everything: tip, theorem, foobar...
  class: (info) => `custom-block ${info}`,
  level: 2,
  render: ({ info, title, slot }) => {
    const type = capitalize(info)
    const hasTitle = title != null && title !== ''
    const fullTitle = hasTitle ? `${type}: ${title}` : type

    return [
      {
        tag: 'div',
        props: { class: 'custom-block-title' },
        children: fullTitle
      },
      {
        tag: 'div',
        props: { class: 'custom-block-content' },
        children: slot
      }
    ]
  }
})

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

  enhanceApp({ app }) {
    enhanceAppWithTabs(app)
    app.component('VersionPicker', VersionPicker)

    // Register one universal container
    app.use(universalContainer)
  }
}

export default Theme
