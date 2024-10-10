<template>
  <v-row>
    <v-col offset="3" cols="6">
      <v-textarea v-model="text" :rows="10" auto-grow></v-textarea>
      <v-btn color="primary" @click="analyzeText" class="w-100" :loading="isInRequest" size="x-large">
        Analyze Text
      </v-btn>
    </v-col>
    <v-col cols="3">
      <v-card>
        <v-card-title>Codes</v-card-title>
        <v-card-text>
          <v-chip v-for="code in codes" :key="code.code" color="primary" class="ma-1">
            {{ code.code }}: {{ code.display }}
          </v-chip>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>
  <v-row>
    <v-col cols="6" offset="3" v-if="best.length">
      <h2 class="mb-3">Best Matches</h2>
      <v-card v-for="(match, index) in best" :key="index" class="mb-4">
        <v-card-item>
          <v-card-title>
            {{ match.code }}
          </v-card-title>
          <v-card-subtitle>
            {{ match.desc }}
          </v-card-subtitle>
        </v-card-item>
        <v-card-text>
          <span class="font-weight-light">{{ text.substring(0, match.chunk.offset) }}</span>
          <span class="font-weight-bold">{{ text.substring(match.chunk.offset, match.chunk.offset +
            match.chunk.length) }}</span>
          <span class="font-weight-light">{{ text.substring(match.chunk.offset + match.chunk.length)
            }}</span>
        </v-card-text>
        <v-card-subtitle>
          Distance: {{ match.distance }}
        </v-card-subtitle>
        <v-card-action>
          <v-btn icon="mdi-plus-circle" @click="codes.push({ code: match.code, display: match.desc })" flat></v-btn>
        </v-card-action>
      </v-card>
    </v-col>
  </v-row>
</template>

<script setup>
import config from '@/config';
import { ref } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();
const text = ref(route.query.text);
const codes = ref([]);
let isInRequest = ref(false);
const best = ref([]);

async function analyzeText() {
  try {
    best.value = [];
    isInRequest.value = true;

    let url = new URL(config.API_URL);
    url.pathname += `analyze`;
    url = url.toString();

    let response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document: text.value,
      }),
    });

    let json = await response.json();
    best.value = json.best;

  } finally {
    isInRequest.value = false;
  }
}
</script>
