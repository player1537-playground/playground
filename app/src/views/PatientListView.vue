<template>
  <Header />
  <v-container fluid fill-height>
    <v-main>
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
