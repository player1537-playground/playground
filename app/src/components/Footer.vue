<template>
  <v-footer class="flex-wrap py-5" color="primary">
    <v-row justify="center" class="w-100">
      <v-col class="text-center pb-0" cols="12">
        <p><strong>&copy;{{ new Date().getFullYear() }}</strong> Visualiz.AI All rights reserved.</p>
      </v-col>
    </v-row>

    <v-row justify="center">
      <v-col class="text-center" cols="12">
        <nav>
          <a target="_blank" href="https://www.visualiz.ai/privacy-policy">Privacy Policy</a>
          <a target="_blank" href="https://www.visualiz.ai/tos">Terms of Service</a>
        </nav>
      </v-col>
    </v-row>
  </v-footer>
</template>

<script setup></script>

<style scoped>
footer a,
footer a:visited {
  color: white;
}

footer a:not(:last-child) {
  margin-right: 1rem;
}
</style>