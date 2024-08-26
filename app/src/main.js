import { createApp } from 'vue';
import vuetify from './vuetify.js';
import router from './router.js';
import App from './App.vue';

const app = createApp(App);

app.use(vuetify);

app.use(router);

app.mount('#app');
