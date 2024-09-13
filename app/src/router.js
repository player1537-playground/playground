import { createRouter, createWebHistory } from 'vue-router';
import HelloView from './views/HelloView.vue';
import FooView from './views/FooView.vue';
import BarView from './views/BarView.vue';
import BazView from './views/BazView.vue';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Patients',
      component: () => import('./views/PatientListView.vue')
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
