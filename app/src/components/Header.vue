<template>
  <v-app-bar color="primary">
    <template v-slot:prepend>
      <router-link to="/"><v-img :to="{ name: 'Patients' }" :src="logo_white" class="header-logo mx-4" height="32" width="128"></v-img></router-link>
    </template>
  </v-app-bar>
</template>

<script setup>
import logo_white from '@/assets/logo_white.svg'
</script>