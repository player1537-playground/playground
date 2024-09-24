<template>
  <v-container fluid>
    <Header />

    <v-main  class="fill-height">
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
          <!-- <textarea id="visit-summary" readonly></textarea> -->
          <v-textarea v-model="dischargeSummary" readonly></v-textarea>
        </v-col>
      </v-row>

      <v-row>
        <v-col class="search-results">
          <h2>Similar Claims</h2>
          <dl id="searchResults"></dl>
        </v-col>

        <v-col>
          <h2>Claim Denial Patterns</h2>

          <v-row>
            <v-col>
              <v-select label="Diagnosis Code Specificity" v-model="N_DX" :items="[1, 2, 3]" @change="displayHeatmap" />
            </v-col>
            <v-col>
              <v-select label="Procedure Code Specificity" v-model="N_PD" :items="[1, 2, 3]" @change="displayHeatmap" />
            </v-col>
          </v-row>

          <div id="heatmapContainer"></div>
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
    </v-main>

    <Footer />
  </v-container>
</template>

<script setup>
import Header from '@/components/Header.vue';
import Footer from '@/components/Footer.vue';

import { ref } from 'vue';
import { useRoute } from 'vue-router';
import vegaEmbed from 'vega-embed';

const loading = ref(true);
const route = useRoute();
const hadm_id = ref(route.params.hadm_id);

const N_LIMIT = 20;
const N_DX = ref(1);
const N_PD = ref(1);
let currentData = null;
const dischargeSummary = ref('');

const diagnosisDescriptions = {};
const procedureDescriptions = {};

main();

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

  const response = await fetch(`${resource}`, options);
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
  try {
    const response = await fetchData(`https://purple.is.mediocreatbest.xyz/MIMIC-IV-Note/discharge/${hadm_id.value}/`);
    dischargeSummary.value = response?.course || '';
  } catch (error) {
    console.error('Error fetching discharge summary:', error);
    displayError('Error fetching discharge summary');
  }

  try {
    const response = await fetchData(`https://purple.is.mediocreatbest.xyz/dip/`, {
      body: JSON.stringify({
        limit: N_LIMIT,
        text: dischargeSummary.value
      }),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    });

    for (const item of response.search.items) {
      item.diagnosis = item.diagnosis.filter(dx => dx.dx10 !== 'NoDx');
      item.procedure = item.procedure.filter(pd => pd.pd10 !== 'NoPcs');
    }

    currentData = response;
  } catch (error) {
    console.error('Error fetching search results:', error);
    displayError('Error fetching search results');
  }
}

async function fetchDescriptions(data) {
  for (const item of data.items) {
    // Code to pull Dx defintions, which I don't think will be required, but leaving here just in case.

    for (const diagnosis of item.diagnosis) {
      if (diagnosis.dx10 === 'NoDx') {
        continue;
      }
      const diagnosisCode = diagnosis.dx10.length > N_DX.value ? diagnosis.dx10.slice(0, N_DX.value) : diagnosis.dx10;
      if (!diagnosisDescriptions[diagnosisCode]) {
        try {
          const definition = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${diagnosisCode}`);
          diagnosisDescriptions[diagnosisCode] = definition?.desc || '';
        } catch (error) {
          console.warn(`Error fetching diagnosis definition: ${diagnosisCode}`);

          diagnosisDescriptions[diagnosisCode] = '';
        }
      }
    }


    // The endpoint providing the search results includes correct diagnosis descriptions.
    //for (const diagnosis of item.diagnosis) {
    //   if (diagnosis !== 'NoDx') {
    //        diagnosisDescriptions[diagnosis.dx10] = diagnosis.desc;
    //    }
    // }

    // NOTE The endpoint providing the search results includes incomplete procedure descriptions.
    const pd10s = new Set();
    for (const item of data.items) {
      item.procedure.forEach(
        procedure => {
          if (procedure.pd10.startsWith('N')) return;
          const pd10 = procedure.pd10.length > N_PD.value
            ? procedure.pd10.slice(0, N_PD.value)
            : procedure.pd10;
          pd10s.add(pd10);
        }
      );
    }

    for (const procedureCode of pd10s) {
      if (!procedureDescriptions[procedureCode] && procedureCode !== 'NoP') {
        try {
          const definition = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${procedureCode}`);
          procedureDescriptions[procedureCode] = definition?.desc || '';
        } catch (error) {
          console.error('Error fetching procedure definition:', error);
        }
      }
    }
  }
  console.debug('Diagnosis Descriptions:', diagnosisDescriptions);
  console.debug('Procedure Descriptions:', procedureDescriptions);
}

/**
 * @returns {Promise<number>} - The score.
 */
async function fetchScore(dxCodes, pdCodes, what = 'Positive-Negative') {
  try {
    return await fetchData('https://purple.is.mediocreatbest.xyz/reward/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: what,
        dx: dxCodes.filter(dx => dx !== 'NoDx'),
        pd: pdCodes.filter(pd => pd !== 'NoP')
      })
    });
  } catch (error) {
    console.error('Error fetching score:', error);
    //displayError('Error fetching score');
    return 0;
  }
}

async function fetchHeatmapData() {
  let data = [];
  let acceptance = {};
  let rejections = {};

  try {
    acceptance = await fetchData('https://purple.is.mediocreatbest.xyz/ring/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: 'Positive',
        pred: `DX${N_DX.value}`,
        prod: `PD${N_PD.value}`
      })
    });
  } catch (error) {
    console.error('Error fetching heatmap data:', error);
    //displayError('Error fetching heatmap data');
  }

  try {
    rejections = await fetchData('https://purple.is.mediocreatbest.xyz/ring/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        what: 'Negative',
        pred: `DX${N_DX.value}`,
        prod: `PD${N_PD.value}`
      })
    });

    console.debug(rejections);
  } catch (error) {
    console.error('Error fetching heatmap data:', error);
    //displayError('Error fetching heatmap data');
  }

  console.debug(rejections)

  // Let's convert the data to a format that Vega-Lite can understand.
  for (let dxId = 0; dxId < rejections.data.length; dxId++) {
    for (let pdId = 0; pdId < rejections.data[dxId].length; pdId++) {
      const diagnosis = rejections.index[dxId];
      const diagnosisDesc = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${diagnosis}`);

      const procedure = rejections.columns[pdId];
      const procedureDesc = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${procedure}`);

      // Percent rejected truncated to 4 decimal places.
      const acceptanceDxId = acceptance.index.findIndex(dx => dx === diagnosis);
      const acceptancePdId = acceptance.columns.findIndex(pd => pd === procedure);
      const rejectionsCount = rejections.data[dxId][pdId];
      const acceptanceCount = acceptance.data[acceptanceDxId][acceptancePdId];
      const percentAccepted = Math.round(acceptanceCount / (acceptanceCount + rejectionsCount) * 10000) / 100;
      const percentRejected = Math.round(rejectionsCount / (acceptanceCount + rejectionsCount) * 10000) / 100;

      data.push({
        diagnosis: rejections.index[dxId],
        diagnosisDesc: diagnosisDesc.desc,
        procedure: rejections.columns[pdId],
        procedureDesc: procedureDesc.desc,
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
  displayData(currentData);
}

function displayData(data) {
  displaySearchResults(data.search.items || []);
  displayHeatmap(data.search.items || []);
}

/**
  * @todo Add mouseover/tap tooltips on badges.
  */
async function displaySearchResults(items) {
  const container = document.getElementById('searchResults');
  container.innerHTML = ''; // Clear previous content

  for (const item of items) {
    const dxCodes = item.diagnosis.map(dx => dx.dx10);
    const pdCodes = item.procedure.map(pd => pd.pd10);
    const { score, dxs, pds } = await fetchScore(dxCodes, pdCodes);

    const div = document.createElement('div');
    div.className = 'item';
    div.style.backgroundColor = score > 0 ? '#92c5de' : '#f4a582';
    div.style.borderRadius = '5px';
    let divInnerHTML = `
                    <dt><a target="_blank" title="Score: ${score}" href="/index.html?hadm_id=${item.hadm_id}">HADM ${item.hadm_id}</a></dt>
                    <dd><br />
                        <span style="font-size: 0.8em; margin-bottom: 0.5em; line-height: calc(var(--golden-ratio) * 0.8);"><strong>Diagnostic Codes:</strong>`

    for (const dxCode of dxCodes) {
      if (dxCode === 'NoDx') continue;

      if (dxs.includes(dxCode)) {
        divInnerHTML += ` <strong style="text-decoration: underline;">${dxCode}*</strong>`;
      } else {
        divInnerHTML += ` ${dxCode}`;
      }
    }

    divInnerHTML += `
                        </span><br />
                        <span style="font-size: 0.8em; line-height: calc(var(--golden-ratio) * 0.8);"><strong>Procedure Codes:</strong>`
    for (const pdCode of pdCodes) {
      if (pdCode === 'NoP') continue;

      if (pds.includes(pdCode)) {
        divInnerHTML += ` <strong style="text-decoration: underline;">${pdCode}*</strong>`;
      } else {
        divInnerHTML += ` ${pdCode}`;
      }
    }

    divInnerHTML += `
                        </span>
                    </dd>
                `;
    //console.log(divInnerHTML)
    div.innerHTML = divInnerHTML;
    container.appendChild(div);
  }
}

function displayError(message) {
  const containers = ['searchResults', 'diagnoses', 'procedures'];
  containers.forEach(containerId => {
    const container = document.getElementById(containerId);
    container.innerHTML = `<p class="error">${message}</p>`;
  });
}

async function displayHeatmap(items) {
  //const heatmapData = new Map();
  const heatmapData = await fetchHeatmapData();

  //await fetchDescriptions(currentData.search);

  /*
  items.forEach(item => {
      item.diagnosis.forEach(diagnosis => {
          const diagnosisCode = diagnosis.dx10.length > N_DX.value ? diagnosis.dx10.slice(0, N_DX.value) : diagnosis.dx10;
          const diagnosisDesc = diagnosisDescriptions[diagnosisCode] || '';

          item.procedure.forEach(procedure => {
              const procedureCode = procedure.pd10.length > N_PD.value ? procedure.pd10.slice(0, N_PD.value) : procedure.pd10;
              const procedureDesc = procedureDescriptions[procedureCode] || '';
              const key = `${diagnosisCode}-${procedureCode}`;

              if (heatmapData.has(key)) {
                  heatmapData.get(key).count++;
              } else {
                  heatmapData.set(key, {
                      diagnosis: diagnosisCode,
                      diagnosisDesc: diagnosisDesc,
                      procedure: procedureCode,
                      procedureDesc: procedureDesc,
                      count: 1,
                  });
              }
          });
      });
  });

  const diagnoses = [...new Set(Array.from(heatmapData.values()).map(item => item.diagnosis))];
  const procedures = [...new Set(Array.from(heatmapData.values()).map(item => item.procedure))];
  */

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
        { "field": "diagnosisDesc", "type": "ordinal", "title": "Diagnosis Description" },
        { "field": "procedure", "type": "ordinal", "title": "Procedure" },
        { "field": "procedureDesc", "type": "ordinal", "title": "Procedure Description" },
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
      },
      /*
      {
          "mark": "text",
          "encoding": {
              "text": { "field": "count", "type": "quantitative" },
              "color": {
                  "condition": { "test": `datum['count'] < ${max * .9}`, "value": "black" },
                  "value": "white"
              }
          }
      }
      */
    ],
    "config": {
      "axis": { "grid": true, "tickBand": "extent" }
    }
  };

  const tooltipOptions = {
    "tooltip": {
      "theme": "heatmap",
      "formatTooltip": (value, sanitize) => `
                        <div class="diagnosis-code">Diagnosis: ${sanitize(value.Diagnosis)}</div>
                        <div class="diagnosis-description">${sanitize(value["Diagnosis Description"])}</div>
                        <hr />
                        <div class="procedure-code">Procedure: ${sanitize(value.Procedure)}</div>
                        <div class="procedure-description">${sanitize(value["Procedure Description"])}</div>
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

  document.getElementById('heatmapOptions').hidden = false;
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

.search-results a,
.search-results a:visited {
  color: #333;
  text-decoration: none;
}

h2 {
  margin-top: 0;
  color: #333;
}

textarea {
  width: 100%;
  height: 100px;
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