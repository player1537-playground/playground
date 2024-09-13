<template>
<v-container fluid fill-height>
  <Header />

  <v-main>
    <nav id="actions-toolbar">
      <RouterLink to="/">Back to Visits</RouterLink>

      <div class="button-group end">
        <RouterLink to="/analyze">Analyze</RouterLink>"
        <RouterLink to="/foo">Recommend Diagnosis Codes</RouterLink>
        <RouterLink to="/baz">Recommend Billing Codes</RouterLink>
        <RouterLink to="/bar">Recommend SDOH Codes</RouterLink>
      </div>
    </nav>

    <div class="loader" :hidden="showSpinner"></div>

    <section id="patient-content">
      <h3>Patient Information</h3>
      <div id="patient-info"></div>
    </section>

    <section id="diagnoses-content">
      <h3>Diagnoses</h3>
      <ul id="diagnoses-list"></ul>
    </section>

    <section id="procedures-content">
      <h3>Procedures</h3>
      <ul id="procedures-list"></ul>
    </section>

    <section id="discharge-content">
      <h3>Discharge Notes</h3>
      <details>
        <summary>Full Record</summary>
        <textarea readonly></textarea>
      </details>
      <div id="discharge-notes"></div>
    </section>
  </v-main>

  <Footer />
</v-container>
</template>

<script setup>
import { ref } from 'vue';
import { useRoute } from 'vue-router';
import Footer from '@/components/Footer.vue';
import Header from '@/components/Header.vue';

console.debug('BEGIN showVisitDetails', hadm_id);

const loading = ref(true);
const route = useRoute();
const hadm_id = route.params.id;

const [admission] = await fetch(`/admissions?hadm_id=${hadm_id}`);
const [patient] = await fetchData(`/patients?subject_id=${admission.subject_id}`);

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

const diagnoses = await fetchData(`/diagnoses?hadm_id=${hadm_id}`);
const diagnosesList = document.getElementById('diagnoses-list');
diagnosesList.innerHTML = '';
diagnoses.forEach(diagnosis => {
  const li = document.createElement('li');
  li.textContent = `${diagnosis.dx10}`;
  diagnosesList.appendChild(li);
});

const procedures = await fetchData(`/procedures?hadm_id=${hadm_id}`);
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

let discharges = await fetchData(`/discharges?hadm_id=${hadm_id}`);
if (hadm_id == '27269506') {
  discharges = [{ text: THE_GOOD_ONE }];
}
const dischargeNotes = document.getElementById('discharge-notes');
dischargeNotes.innerHTML = '';
discharges.forEach(note => {
  const formattedText = formatDischargeNotes(note.text);
  dischargeNotes.innerHTML += formattedText;
});

// Setup Analyze button
let analyzeBtnUrl = new URL('/analyze.html', window.location.href);
analyzeBtnUrl.searchParams.set('hadm_id', hadm_id);
analyzeBtn.href = analyzeBtnUrl.href;

// Setup Recommend Billing Codes button
let bazBtn = document.getElementById('bazBtn');
let bazBtnUrl = new URL('https://red.is.mediocreatbest.xyz/baz/');
bazBtnUrl.searchParams.set('dx', diagnoses.map(d => d.dx10).join(','));
bazBtnUrl.searchParams.set('pd', procedures.map(p => p.pd10).join(','));
bazBtn.href = bazBtnUrl.href;

function formatDischargeNotes(text) {
  return;
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