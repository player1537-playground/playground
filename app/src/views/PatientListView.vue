<template>
  <v-container fluid fill-height>
    <Header />

    <v-main>
      <PatientsList />
    </v-main>

    <Footer />
  </v-container>
</template>

<script setup>
import Footer from '@/components/Footer.vue'
import Header from '@/components/Header.vue'
import PatientsList from '@/components/PatientsList.vue'
</script>
