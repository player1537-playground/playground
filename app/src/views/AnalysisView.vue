<template>
  <v-row>
    <v-col>
      <nav>
        <v-btn to="/" color="primary" text>Back to Visits</v-btn>
      </nav>
    </v-col>
  </v-row>
  <v-row>
    <v-col>
      <h2>Claim Summary</h2>
      <v-textarea v-model="dischargeSummary" readonly></v-textarea>
    </v-col>
  </v-row>

  <v-row>
    <v-col cols="12" md="6" class="search-results">
      <v-row>
        <v-col>
          <h2>Similar Claims</h2>
        </v-col>
      </v-row>
      <!-- <dl id="searchResults"></dl> -->
      <v-row>
        <v-col cols="12" v-if="!searchResultsCards.length">
          <v-skeleton-loader type="card"></v-skeleton-loader>
        </v-col>
        <v-col v-for="card in searchResultsCards" :key="card.title" cols="12">
          <v-card :color="card.rejected ? 'rgb(244, 165, 130)' : 'rgb(146, 197, 222)'">
            <v-card-title>Admission {{ card.hadm_id }}</v-card-title>
            <v-card-text>
              <div>
                <strong>Diagnostic Codes:</strong> <span v-html="card.dx"></span>
              </div>
              <div>
                <strong>Procedure Codes:</strong> <span v-html="card.pd"></span>
              </div>
              <div v-if="!card.rejected">
                <strong>Payment:</strong> {{ card.payment }}
                <strong>Underpayment:</strong> {{ card.underpayment }}
              </div>
            </v-card-text>
            <v-card-actions>
              <v-btn :to="{ name: 'Patient', params: { id: card.hadm_id } }" text>View Claim</v-btn>
              <v-spacer></v-spacer>
              <v-btn icon="mdi-thumb-up" @click="upvote(card.hadm_id)" />
              <v-btn icon="mdi-thumb-down" @click="downvote(card.hadm_id)" />
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-col>

    <v-col cols="12" md="6">
      <v-row>
        <v-col>
          <h2>Claim Denial Patterns</h2>
        </v-col>
      </v-row>

      <v-row>
        <v-col>
          <v-select label="Diagnosis Code Specificity" v-model="N_DX" :items="[1, 2, 3]" @change="displayHeatmap" />
        </v-col>
        <v-col>
          <v-select label="Procedure Code Specificity" v-model="N_PD" :items="[1, 2]" @change="displayHeatmap" />
        </v-col>
      </v-row>

      <v-row style="overflow-x: scroll;">
        <v-col>
          <div id="heatmapContainer"></div>
          <v-skeleton-loader v-if="heatmapLoading" type="card"></v-skeleton-loader>
        </v-col>
      </v-row>
    </v-col>
  </v-row>

  <v-row class="d-none">
    <v-col>
      <h2>Diagnoses</h2>
      <dl id="diagnoses"></dl>
    </v-col>

    <v-col>
      <h2>Procedures</h2>
      <dl id="procedures"></dl>
    </v-col>
  </v-row>
</template>

<script setup>
import config from '@/config';
import { ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import vegaEmbed from 'vega-embed';

const loading = ref(true);
const heatmapLoading = ref(true);
const route = useRoute();
const hadm_id = ref(route.params.hadm_id);

const N_LIMIT = 20;
const N_DX = ref(1);
const N_PD = ref(1);
let currentData = null;
const dischargeSummary = ref('');
const searchResultsCards = ref([]);

const diagnosisDescriptions = {};
const procedureDescriptions = {};

main();

watch([N_DX, N_PD], () => {
  document.getElementById('heatmapContainer').innerHTML = '';
  displayHeatmap(currentData.search.items || []);
});

/**
 * Fetch data from a URL. Cache the data in localStorage.
 *
 * @param {string} resource - The URL to fetch data from.
 * @returns {Promise<Object>} - The JSON data.
 */
async function fetchData(resource, options = {}, cache = true) {
  const key = `${resource}-${JSON.stringify(options)}`;

  if (cache && localStorage.getItem(key)) {
    return JSON.parse(localStorage.getItem(key));
  }

  const response = await fetch(resource, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const responseBody = await response.json();

  if (cache) {
    localStorage.setItem(key, JSON.stringify(responseBody));
  }
  return responseBody;
}

async function fetchSearchData() {
  let url = new URL(config.API_URL);
  url.pathname += `MIMIC-IV-Note/`
  url.pathname += `discharge/${encodeURIComponent(hadm_id.value)}/`;
  url = url.toString();

  try {
    const response = await fetchData(url, {}, false);
    dischargeSummary.value = response?.course || '';
  } catch (error) {
    console.error('Error fetching discharge summary:', error);
  }

  url = new URL(config.API_URL);
  url.pathname += `dip/`;
  url = url.toString();

  try {
    const response = await fetchData(url, {
      body: JSON.stringify({
        limit: N_LIMIT,
        text: dischargeSummary.value
      }),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    }, false);

    for (const item of response.search.items) {
      item.diagnosis = item.diagnosis.filter(dx => dx.dx10 !== 'NoDx');
      item.procedure = item.procedure.filter(pd => pd.pd10 !== 'NoPcs');
    }

    currentData = response;
  } catch (error) {
    console.error('Error fetching search results:', error);
  }
}

async function fetchDescription(code, type = 'dx') {
  if (type === 'dx') {
    if (diagnosisDescriptions[code]) {
      return diagnosisDescriptions[code];
    }
    const response = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${code}`, {}, false);
    diagnosisDescriptions[code] = response?.desc || '';
    return diagnosisDescriptions[code];
  } else {
    if (procedureDescriptions[code]) {
      return procedureDescriptions[code];
    }
    const response = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${code}`, {}, false);
    procedureDescriptions[code] = response?.desc || '';
    return procedureDescriptions[code];
  }
}

async function fetchDescriptions(data) {
  const dxCodes = new Set();
  const pdCodes = new Set();

  for (const item of data.items) {
    for (const diagnosis of item.diagnosis) {
      if (diagnosis.dx10 === 'NoDx') {
        continue;
      }
      const diagnosisCode = diagnosis.dx10.length > N_DX.value ? diagnosis.dx10.slice(0, N_DX.value) : diagnosis.dx10;
      dxCodes.add(diagnosisCode);
    }

    for (const procedure of item.procedure) {
      if (procedure.pd10 === 'NoP') {
        continue;
      }
      const procedureCode = procedure.pd10.length > N_PD.value ? procedure.pd10.slice(0, N_PD.value) : procedure.pd10;
      pdCodes.add(procedureCode);
    }
  }

  for (const diagnosisCode of dxCodes) {
    if (!diagnosisDescriptions[diagnosisCode]) {
      try {
        await fetchDescription(diagnosisCode, 'dx');
      } catch (error) {
        console.warn(`Error fetching diagnosis definition: ${diagnosisCode}`);

        diagnosisDescriptions[diagnosisCode] = '';
      }
    }
  }

  for (const procedureCode of pdCodes) {
    if (!procedureDescriptions[procedureCode] && procedureCode !== 'NoP') {
      try {
        await fetchDescription(procedureCode, 'pd');
      } catch (error) {
        console.error('Error fetching procedure definition:', error);

        procedureDescriptions[procedureCode] = '';
      }
    }
  }

  // for (const item of data.items) {
  //   for (const diagnosis of item.diagnosis) {
  //     if (diagnosis.dx10 === 'NoDx') {
  //       continue;
  //     }
  //     const diagnosisCode = diagnosis.dx10.length > N_DX.value ? diagnosis.dx10.slice(0, N_DX.value) : diagnosis.dx10;
  //     if (!diagnosisDescriptions[diagnosisCode]) {
  //       try {
  //         const definition = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${diagnosisCode}`, {}, false);
  //         diagnosisDescriptions[diagnosisCode] = definition?.desc || '';
  //       } catch (error) {
  //         console.warn(`Error fetching diagnosis definition: ${diagnosisCode}`);

  //         diagnosisDescriptions[diagnosisCode] = '';
  //       }
  //     }
  //   }


  //   // The endpoint providing the search results includes correct diagnosis descriptions.
  //   //for (const diagnosis of item.diagnosis) {
  //   //   if (diagnosis !== 'NoDx') {
  //   //        diagnosisDescriptions[diagnosis.dx10] = diagnosis.desc;
  //   //    }
  //   // }

  //   // NOTE The endpoint providing the search results includes incomplete procedure descriptions.
  //   const pd10s = new Set();
  //   for (const item of data.items) {
  //     item.procedure.forEach(
  //       procedure => {
  //         if (procedure.pd10.startsWith('N')) return;
  //         const pd10 = procedure.pd10.length > N_PD.value
  //           ? procedure.pd10.slice(0, N_PD.value)
  //           : procedure.pd10;
  //         pd10s.add(pd10);
  //       }
  //     );
  //   }

  //   for (const procedureCode of pd10s) {
  //     if (!procedureDescriptions[procedureCode] && procedureCode !== 'NoP') {
  //       try {
  //         const definition = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${procedureCode}`, {}, false);
  //         procedureDescriptions[procedureCode] = definition?.desc || '';
  //       } catch (error) {
  //         console.error('Error fetching procedure definition:', error);
  //       }
  //     }
  //   }
  // }
}

/**
 * @returns {Promise<number>} - The score.
 */
async function fetchScore(dxCodes, pdCodes, what = 'Positive-Negative') {
  let url = new URL(config.API_URL);
  url.pathname += `reward/`;
  url = url.toString();

  try {
    return await fetchData(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: what,
        dx: dxCodes.filter(dx => dx !== 'NoDx'),
        pd: pdCodes.filter(pd => pd !== 'NoP')
      })
    }, false);
  } catch (error) {
    console.error('Error fetching score:', error);
    return 0;
  }
}

async function fetchUnderpayment(dxCodes, pdCodes) {
  if (pdCodes.length === 0) {
    throw new Error('No procedure codes provided.');
  }

  let url = new URL(config.API_URL);
  url.pathname += `underpayment/`;
  url = url.toString();

  return await fetchData(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dxs: dxCodes.filter(dx => dx !== 'NoDx'),
      pds: pdCodes.filter(pd => pd !== 'NoP'),
      ndx: 2,
      npd: 2
    })
  }, false);
}

async function fetchHeatmapData() {
  let data = [];
  let acceptance = {};
  let rejections = {};

  let url = new URL(config.API_URL);
  url.pathname += `ring/`;
  url = url.toString();

  try {
    acceptance = await fetchData(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: 'Positive',
        pred: `DX${N_DX.value}`,
        prod: `PD${N_PD.value}`
      })
    }, false);
  } catch (error) {
    console.error('Error fetching heatmap data:', error);
  }

  url = new URL(config.API_URL);
  url.pathname += `ring/`;
  url = url.toString();

  try {
    rejections = await fetchData(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: 'Negative',
        pred: `DX${N_DX.value}`,
        prod: `PD${N_PD.value}`
      })
    }, false);
  } catch (error) {
    console.error('Error fetching heatmap data:', error);
  }

  // Let's convert the data to a format that Vega-Lite can understand.
  for (let dxId = 0; dxId < rejections.data.length; dxId++) {
    for (let pdId = 0; pdId < rejections.data[dxId].length; pdId++) {
      const diagnosis = rejections.index[dxId];
      //const diagnosisDesc = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${diagnosis}`, {}, false);

      const procedure = rejections.columns[pdId];
      //const procedureDesc = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${procedure}`, {}, false);

      // Percent rejected truncated to 4 decimal places.
      const acceptanceDxId = acceptance.index.findIndex(dx => dx === diagnosis);
      const acceptancePdId = acceptance.columns.findIndex(pd => pd === procedure);
      const rejectionsCount = rejections.data[dxId][pdId];
      const acceptanceCount = acceptance.data[acceptanceDxId][acceptancePdId];
      const percentAccepted = Math.round(acceptanceCount / (acceptanceCount + rejectionsCount) * 10000) / 100;
      const percentRejected = Math.round(rejectionsCount / (acceptanceCount + rejectionsCount) * 10000) / 100;

      data.push({
        diagnosis: rejections.index[dxId],
        //diagnosisDesc: diagnosisDesc.desc,
        procedure: rejections.columns[pdId],
        //procedureDesc: procedureDesc.desc,
        count: rejectionsCount + acceptanceCount,
        percentAccepted: percentAccepted,
        percentRejected: percentRejected
      });
    }
  }

  return data;
}

async function main() {

  await fetchSearchData();

  if (!currentData) return;

  await fetchDescriptions(currentData.search);
  displaySearchResults(currentData.search.items || []);
  displayHeatmap(currentData.search.items || []);
}

async function displaySearchResults(items) {
  for (const item of items) {
    const dxCodes = item.diagnosis.map(dx => dx.dx10);
    const pdCodes = item.procedure.map(pd => pd.pd10);
    const { score, dxs, pds } = await fetchScore(dxCodes, pdCodes);
    let payment, underpayment;
    try {
      const underpaymentData = await fetchUnderpayment(dxCodes, pdCodes);
      payment = !underpaymentData ? 0 : underpaymentData.cost.lo.tight.avg * underpaymentData.mult.lo.tight.avg;
      const maxPayment = !underpaymentData ? 0 : underpaymentData.cost.hi.tight.avg * underpaymentData.mult.hi.tight.avg;
      underpayment = (maxPayment - payment) / maxPayment * 100;

      payment = `$${payment.toFixed(2)}`;
      underpayment = `${underpayment.toFixed(1)}%`;
    } catch (error) {
      payment = 'N/A';
      underpayment = 'N/A';
    }

    const card = {
      hadm_id: item.hadm_id,
      dx: '',
      pd: '',
      rejected: score < 0,
      payment: payment,
      underpayment: underpayment
    };

    for (const dxCode of dxCodes) {
      if (dxCode === 'NoDx') continue;

      if (dxs.includes(dxCode)) {
        card.dx += ` <strong style="text-decoration: underline;">${dxCode}*</strong>`;
      } else {
        card.dx += ` ${dxCode}`;
      }
    }

    for (const pdCode of pdCodes) {
      if (pdCode === 'NoP') continue;

      if (pds.includes(pdCode)) {
        card.pd += ` <strong style="text-decoration: underline;">${pdCode}*</strong>`;
      } else {
        card.pd += ` ${pdCode}`;
      }
    }

    searchResultsCards.value.push(card);
  }
}

function injectDescription(code, type = 'dx') {
  const elements = document.getElementsByClassName(`${type}-desc`);
  for (const element of elements) {
    element.innerHTML = 'Loading...';
  }

  fetchDescription(code, type).then(description => {
    const elements = document.getElementsByClassName(`${type}-desc`);
    for (const element of elements) {
      element.innerHTML = description;
    }
  });
}

async function displayHeatmap(items) {
  heatmapLoading.value = true;
  const heatmapData = await fetchHeatmapData();

  const diagnoses = [...new Set(heatmapData.map(item => item.diagnosis))];
  const procedures = [...new Set(heatmapData.map(item => item.procedure))];

  const max = heatmapData.reduce((acc, item) => Math.max(acc, item.count), 0);
  const width = procedures.length * 26 / Math.max(N_DX.value, N_PD.value);
  const height = diagnoses.length * 26 / Math.max(N_DX.value, N_PD.value);

  const redHigh = '#ca0020';
  const redLow = '#f4a582';
  const blueLow = '#92c5de';
  const blueHigh = '#0571b0';

  const vegaSpec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": { "values": heatmapData.filter(item => item.count >= 100) },
    "width": width,
    "height": height,
    "encoding": {
      "x": { "field": "procedure", "type": "ordinal", "zero": false },
      "y": { "field": "diagnosis", "type": "ordinal", "zero": false },
      "tooltip": [
        { "field": "diagnosis", "type": "ordinal", "title": "Diagnosis" },
        { "field": "procedure", "type": "ordinal", "title": "Procedure" },
        { "field": "count", "type": "quantitative", "title": "Count" },
        { "field": "percentRejected", "type": "quantitative", "title": "Percent Rejected" }
      ]
    },
    "layer": [
      {
        "mark": "rect",
        "encoding": {
          "color": {
            "field": "percentRejected",
            "type": "quantitative",
            "title": "Percent Rejected",
            "legend": false,
            "scale": {
              "bins": [0, 40, 60, 100],
              "range": [blueLow, blueHigh, redLow, redHigh]
            }
          }
        }
      }
    ],
    "config": {
      "axis": { "grid": true, "tickBand": "extent" }
    }
  };

  const tooltipOptions = {
    "tooltip": {
      "theme": "heatmap",
      "formatTooltip": (value, sanitize) => `
                        <div class="dx-code">Diagnosis: ${sanitize(value.Diagnosis)}</div>
                        <div class="dx-desc">${injectDescription(value.Diagnosis, 'dx')}</div>
                        <hr />
                        <div class="pd-code">Procedure: ${sanitize(value.Procedure)}</div>
                        <div class="pd-desc">${injectDescription(value.Procedure, 'pd')}</div>
                        <hr />
                        <div class="count">Occurences: ${sanitize(value.Count)}</div>
                        <hr />
                        Denials: ${sanitize(value["Percent Rejected"])}%
                        <div style="columns: 1; margin-top: 10px;">
                            <div>
                                <h2>Recommended Actions</h2>
                                <ul>
                                    <li>Action 1</li>
                                    <li>Action 2</li>
                                    <li>Action 3</li>
                                </ul>
                            </div>
                        </div>
                    `
    }
  }

  await vegaEmbed('#heatmapContainer', vegaSpec, tooltipOptions);

  heatmapLoading.value = false;
}

function upvote(hadm_id) {
  console.log('Upvoting:', hadm_id);
}

function downvote(hadm_id) {
  console.log('Downvoting:', hadm_id);
}
</script>

<style scoped>
.grid-container>div {
  background-color: white;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  max-height: 150vh;
}

#heatmapContainer {
  max-width: 100%;
  max-height: 100%;
}

.item {
  margin-bottom: 16px;
  border-bottom: 1px solid #eee;
  padding: 10px 12px;
}

.item:first {
  border-top: 1px solid #eee;
}

.item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.error {
  color: #ff0000;
  font-weight: bold;
}

.no-desc {
  color: #999;
  font-style: italic;
}

.diagnosis-count {
  color: #666;
  font-size: 0.8em;
  font-style: italic;
  margin-top: .25em;
}

#vg-tooltip-element.vg-tooltip.heatmap-theme {
  border: 1px solid #ccc;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  padding: 1em;
  font-size: 1rem;
  max-width: 500px;
}

#vg-tooltip-element.vg-tooltip.heatmap-theme [class*="description"] {
  font-size: 0.75em;
}

#vg-tooltip-element.vg-tooltip.heatmap-theme h2 {
  font-size: 1em;
}

#vg-tooltip-element.vg-tooltip.heatmap-theme ul {
  list-style-type: none;
  padding: 0 0 0 1em;
  margin: 0;
  line-height: var(--golden-ratio);
}
</style>