<template>
  <Header />
  <v-container fluid class="fill-height">
    <v-main class="fill-height">
      <nav id="actions-toolbar">
        <v-btn @click="router.back()" color="primary" text>Back to Visits</v-btn>

        <div class="button-group end">
          <v-btn color="primary" :to="analyzeBtnUrl">Analyze</v-btn>
          <v-btn color="primary" to="/foo/">Recommend Diagnosis Codes</v-btn>
          <v-btn color="primary" to="/baz/">Recommend Billing Codes</v-btn>
          <v-btn color="primary" to="/bar/">Recommend SDOH Codes</v-btn>
        </div>
      </nav>

      <v-row>
        <v-col cols="12" md="4">
          <v-card class="fill-height">
            <v-card-item>
              <v-card-title>Patient Information</v-card-title>
            </v-card-item>

            <v-card-text>
              <div id="patient-info"></div>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" md="4">
          <v-card class="fill-height">
            <v-card-item>
              <v-card-title>Diagnoses</v-card-title>
            </v-card-item>

            <v-card-text>
              <ul id="diagnoses-list"></ul>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" md="4">
          <v-card class="fill-height">
            <v-card-item>
              <v-card-title>Procedures</v-card-title>
            </v-card-item>

            <v-card-text>
              <ul id="procedures-list"></ul>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <v-row>
        <v-col>
          <h3>Discharge Notes</h3>
          <details>
            <summary>Full Record</summary>
            <textarea readonly></textarea>
          </details>
          <div id="discharge-notes"></div>
        </v-col>
      </v-row>
    </v-main>

  </v-container>
  <Footer />
</template>

<script setup>
import { ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import Footer from '@/components/Footer.vue';
import Header from '@/components/Header.vue';

const loading = ref(true);
const router = useRouter();
const route = useRoute();
const hadm_id = route.params.id;

const analyzeBtnUrl = ref(`/analyze/${hadm_id}`);

console.debug('BEGIN showVisitDetails', hadm_id);

loadData();

async function fetchData(url) {
  const response = await fetch(`https://olive.is.mediocreatbest.xyz/937506e2${url}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

async function loadData() {
  try {
    const [admission] = await fetchData(`/admissions?hadm_id=${hadm_id}`);
    console.debug('admission', admission);
    const [patient] = await fetchData(`/patients?subject_id=${admission.subject_id}`);
    console.debug('patient', patient);

    document.getElementById('patient-info').innerHTML = `
      <dl>
          <dt>Patient ID</dt><dd>${patient.subject_id}</dd>
          <dt>Gender</dt><dd>${patient.gender}</dd>
          <dt>Age Group</dt><dd>${patient.anchor_age}</dd>
          <dt>Anchor Year</dt><dd>${patient.anchor_year}</dd>
          <dt>Date of Death</dt><dd>${patient.dod || 'N/A'}</dd>
          <dt>Visit ID</dt><dd>${hadm_id}</dd>
      </dl>
    `;
  } catch (error) { }

  try {
    const diagnoses = await fetchData(`/diagnoses?hadm_id=${hadm_id}`);
    console.debug('diagnoses', diagnoses);
    const diagnosesList = document.getElementById('diagnoses-list');
    diagnosesList.innerHTML = '';
    diagnoses.forEach(diagnosis => {
      const li = document.createElement('li');
      li.textContent = `${diagnosis.dx10}`;
      diagnosesList.appendChild(li);
    });
  } catch (error) { }

  try {
    const procedures = await fetchData(`/procedures?hadm_id=${hadm_id}`);
    console.debug('procedures', procedures);
    const proceduresList = document.getElementById('procedures-list');
    proceduresList.innerHTML = '';
    procedures.forEach(procedure => {
      const li = document.createElement('li');
      li.textContent = `${procedure.pd10}`;
      proceduresList.appendChild(li);
    });
    if (procedures.length === 0) {
      const li = document.createElement('li');
      li.textContent = 'No procedures recorded';
      proceduresList.appendChild(li);
    }
  } catch (error) { }

  try {
    let discharges = await fetchData(`/discharges?hadm_id=${hadm_id}`);
    console.debug('discharges', discharges);
    if (hadm_id == '27269506') {
      discharges = [{ text: THE_GOOD_ONE }];
    }
    const dischargeNotes = document.getElementById('discharge-notes');
    dischargeNotes.innerHTML = '';
    discharges.forEach(note => {
      const formattedText = formatDischargeNotes(note.text);
      dischargeNotes.innerHTML += formattedText;
    });
  } catch (error) { }

  // Setup Analyze button
  // let analyzeBtnUrl = new URL('/analyze.html', window.location.href);
  // analyzeBtnUrl.searchParams.set('hadm_id', hadm_id);
  // analyzeBtn.href = analyzeBtnUrl.href;

  // Setup Recommend Billing Codes button
  //   let bazBtn = document.getElementById('bazBtn');
  //   let bazBtnUrl = new URL('https://red.is.mediocreatbest.xyz/baz/');
  //   bazBtnUrl.searchParams.set('dx', diagnoses.map(d => d.dx10).join(','));
  //   bazBtnUrl.searchParams.set('pd', procedures.map(p => p.pd10).join(','));
  //   bazBtn.href = bazBtnUrl.href;
}

function formatDischargeNotes(text) {
  document.querySelector('#discharge-content details textarea').value = text;

  const lines = text.split('\n');
  let formattedHtml = '';
  let currentSection = '';
  let currentText = '';

  lines.forEach(line => {
    if (line.trim().endsWith(':')) {
      if (currentSection) {
        if (currentSection === 'Brief Hospital Course:') {
          const fooBtn = document.getElementById('fooBtn');
          const fooBtnUrl = new URL('https://red.is.mediocreatbest.xyz/foo/');
          fooBtnUrl.searchParams.set('text', currentText);
          fooBtn.href = fooBtnUrl.href;
        }

        formattedHtml += `</div>`;
      }

      if (line === 'null:') {
        return;
      }

      formattedHtml += `<div class="note-section"><h4 class="summary-heading">${line}</h4>`;
      currentSection = line;
      currentText = '';
    } else {
      formattedHtml += `<p>${line}</p>`;
      currentText += line + ' ';
    }
  });

  if (currentSection) {
    formattedHtml += `</div>`;
  }

  return formattedHtml;
}

console.debug('END showVisitDetails', hadm_id);
</script>

<style scoped>
h3.summary-heading {
  margin-bottom: 5px;
  color: #2c3e50;
  font-size: 1.1em;
}

/*
#visit-details {
  margin: 20px;
}
*/

.loader {
  border: 5px solid #f3f3f3;
  border-top: 5px solid #3498db;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

section {
  background-color: var(--section-bg-color);
  padding: 20px;
  margin: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

section :first-child {
  margin-top: 0;
}

#visit-details {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
}

#actions-toolbar {
  grid-column: 1 / 4;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

#patient-content {
  grid-column: 1 / 2;
}

#patient-content dl {
  display: grid;
  grid-template-columns: max-content auto;
  gap: 10px;

}

#patient-content dt {
  grid-column-start: 1;
  font-weight: bold;
  text-align: end;
}

#patient-content dt::after {
  content: ':';
}

#patient-content dd {
  grid-column-start: 2;
  margin: 0;
}

#diagnoses-content {
  grid-column: 2 / 3;
}

#procedures-content {
  grid-column: 3 / 4;
}

#discharge-content {
  grid-column: 1 / 4;
}

#discharge-content details {
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin: 20px 0;
}

#discharge-content summary {
  margin-bottom: 10px;
  user-select: none;
}

#discharge-content textarea {
  display: block;
  resize: none;
  width: 100%;
  height: 80vh;
}

#discharge-notes {
  column-count: 3;
  column-gap: 20px;
}

.note-section {
  break-inside: avoid-column;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 5px;
  background-color: #eee;
}

.note-section :first-child {
  margin-top: 0;
}

.note-section h4 {
  color: var(--frame-bg-color);
}

.note-section p {
  margin: 0;
}

nav#actions-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0 20px 20px;
}

nav#actions-toolbar button {
  display: flex;
}

.button-group.end {
  justify-self: end;
}
</style>