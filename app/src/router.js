import { createRouter, createWebHistory } from 'vue-router';

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
      component: () => import('./views/FooView.vue')
    },
    {
      path: '/bar/',
      name: 'bar',
      component: () => import('./views/BarView.vue')
    },
    {
      path: '/baz/',
      name: 'baz',
      component: () => import('./views/BazView.vue')
    },
  ],
});

export default router;
