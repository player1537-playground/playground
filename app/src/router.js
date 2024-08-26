import { createRouter, createWebHistory } from 'vue-router';
import HelloView from './views/HelloView.vue';
import FooView from './views/FooView.vue';
import BarView from './views/BarView.vue';
import BazView from './views/BazView.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'hello',
      component: HelloView,
    },
    {
      path: '/foo/',
      name: 'foo',
      component: FooView,
    },
    {
      path: '/bar/',
      name: 'bar',
      component: BarView,
    },
    {
      path: '/baz/',
      name: 'baz',
      component: BazView,
    },
  ],
});

export default router;
