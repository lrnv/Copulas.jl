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

// Note: **No** import of markdown-it-container here, and no markdown.config override

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
    // Nothing else container-related here
  }
}

export default Theme