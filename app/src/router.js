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
      path: '/analyze/text/',
      name: 'foo',
      component: () => import('./views/FooView.vue')
    },
    {
      path: '/analyze/codes/',
      name: 'bar',
      component: () => import('./views/BarView.vue')
    },
    {
      path: '/analyze/sdoh/',
      name: 'baz',
      component: () => import('./views/BazView.vue')
    },
    {
      path: '/:id',
      name: 'Patient',
      component: () => import('./views/PatientDetails.vue')
    },
    {
      path: '/analyze/:hadm_id',
      name: 'Analyze',
      component: () => import('./views/AnalysisView.vue')
    }
  ]
})

export default router
