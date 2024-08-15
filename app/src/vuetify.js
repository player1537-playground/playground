import '@mdi/font/css/materialdesignicons.css'
import 'vuetify/styles'

import { createVuetify } from 'vuetify'
import { mdi } from 'vuetify/iconsets/mdi'
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';

// TODO: this is all horribly wrong, but I don't know what our theme is going to actually be yet so I'm just using defaults for most things and our early brand colors for primary/secondary/accent
const visualizTheme = {
  dark: false,
  colors: {
    // Default colors
    background: '#ffffff',
    surface: '#ffffff',
    error: '#b00020',
    info: '#2196f3',
    success: '#4caf50',
    warning: '#fb8c00',

    // Element colors
    'text-input': '#165486',
    'text-input-bg': '#f2f1f6',

    // Brand primary
    'primary-lighten-5': '#cce4f7',
    'primary-lighten-4': '#9ac9ee',
    'primary-lighten-3': '#67ade6',
    'primary-lighten-2': '#3492de',
    'primary-lighten-1': '#1e74b9',
    primary: '#165486',
    'primary-darken-1': '#124670',
    'primary-darken-2': '#0f3859',
    'primary-darken-3': '#0b2a43',
    'primary-darken-4': '#071c2d',
    'primary-darken-5': '#040e16',

    // Brand secondary
    'secondary-lighten-5': '#01aaff',
    'secondary-lighten-4': '#007cbb',
    'secondary-lighten-3': '#005e8c',
    'secondary-lighten-2': '#004e75',
    'secondary-lighten-1': '#002f47',
    secondary: '#002030',
    'secondary-darken-1': '#001c2a',
    'secondary-darken-2': '#001824',
    'secondary-darken-3': '#00141e',
    'secondary-darken-4': '#001018',
    'secondary-darken-5': '#00080c',

    // Brand accent
    'accent-lighten-5': '#f1fecb',
    'accent-lighten-4': '#e2fe97',
    'accent-lighten-3': '#d4fd63',
    'accent-lighten-2': '#c6fd2f',
    'accent-lighten-1': '#b4f503',
    accent: '#8ec102',
    'accent-darken-1': '#76a102',
    'accent-darken-2': '#5f8101',
    'accent-darken-3': '#476101',
    'accent-darken-4': '#2f4001',
    'accent-darken-5': '#182000'
  }
}

export default createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'visualizTheme',
    icons: {
      defaultSet: 'mdi',
      sets: {
        mdi,
      },
    },
    themes: {
      visualizTheme,
    },
  },
});