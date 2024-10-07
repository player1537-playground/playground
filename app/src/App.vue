<template>
  <v-app>
    <Header />
    <v-main>
      <v-container>
    <RouterView />
        <Footer />
      </v-container>
    </v-main>
  </v-app>
</template>

<style scoped>
</style>

<script setup>
import Header from '@/components/Header.vue';
import Footer from '@/components/Footer.vue';
</script>
