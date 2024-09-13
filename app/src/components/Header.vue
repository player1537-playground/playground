<template>
  <v-app-bar app color="primary">
    <template v-slot:prepend>
      <v-img :src="logo_white" class="header-logo mx-4" height="32" width="128"></v-img>
    </template>
  </v-app-bar>
</template>

<script setup>
import logo_white from '@/assets/logo_white.svg'
</script>