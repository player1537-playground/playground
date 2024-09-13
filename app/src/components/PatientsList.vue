<template>
  <v-data-table :headers="admissionsHeaders" :items="admissions" :loading="loading" @update:options="loadItems">
    <template v-slot:loading>
      <v-skeleton-loader type="table-row@3"></v-skeleton-loader>
    </template>
    <template v-slot:item.actions="{ item }">
      <v-btn @click="router.push({ name: 'Patient', params: { id: item.hadm_id } })" color="primary" text>View</v-btn>
    </template>
  </v-data-table>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const admissions = ref([])
const admissionsHeaders = ref([
  { title: 'Visit ID', key: 'hadm_id' },
  { title: 'Patient ID', key: 'subject_id' },
  { title: 'Action', key: 'actions', sortable: false },
])
const loading = ref(true)

async function loadItems() {
  loading.value = true

  let response = await fetch('https://olive.is.mediocreatbest.xyz/937506e2/admissions')
  let data = await response.json()
  admissions.value = data

  loading.value = false
}
</script>