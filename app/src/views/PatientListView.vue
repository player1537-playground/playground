<template>
  <Header />
  <v-container fluid class="fill-height">
    <v-main class="fill-height">
      <PatientsList />
    </v-main>
  </v-container>
  <Footer />
</template>

<script setup>
import Footer from '@/components/Footer.vue'
import Header from '@/components/Header.vue'
import PatientsList from '@/components/PatientsList.vue'
</script>
