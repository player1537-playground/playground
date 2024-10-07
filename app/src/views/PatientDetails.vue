<template>
  <v-navigation-drawer class="pl-4 pt-4" permanent persistent>
    <v-list-subheader>Navigation</v-list-subheader>
    <v-list-item title="Back to Admissions" :to="{ name: 'Patients' }"></v-list-item>
    <v-divider v-if="!loading"></v-divider>
    <v-list-subheader v-if="!loading">Actions</v-list-subheader>
    <v-list-item title="Analyze" :to="analyzeBtnUrl" v-if="!loading"></v-list-item>
    <v-list-item title="Recommend Diagnosis Codes" :to="fooBtnUrl" v-if="!loading"></v-list-item>
    <v-list-item title="Recommend Billing Codes" :to="bazBtnUrl" v-if="!loading"></v-list-item>
    <v-list-item title="Recommend SDOH Codes" :to="barBtnUrl" v-if="!loading"></v-list-item>
  </v-navigation-drawer>

  <v-row v-if="!loading">
    <v-col cols="12" md="4">
      <v-card class="fill-height">
        <v-card-item>
          <v-card-title>Patient Information</v-card-title>
        </v-card-item>

        <v-card-text>
          <div id="patient-info"></div>
          <v-table density="compact">
            <tbody>
              <tr>
                <td>Patient ID</td>
                <td>{{ patientInfo.subject_id }}</td>
              </tr>
              <tr>
                <td>Gender</td>
                <td>{{ patientInfo.gender }}</td>
              </tr>
              <tr>
                <td>Age</td>
                <td>{{ patientInfo.anchor_age }}</td>
              </tr>
              <tr>
                <td>Date of Visit</td>
                <td>{{ patientInfo.anchor_year_group }}</td>
              </tr>
              <tr hidden>
                <td>Date of Visit</td>
                <td>{{ patientInfo.anchor_year }}</td>
              </tr>
              <tr hidden>
                <td>Date of Death</td>
                <td>{{ patientInfo.dod || 'N/A' }}</td>
              </tr>
              <tr>
                <td>Visit ID</td>
                <td>{{ hadm_id }}</td>
              </tr>
            </tbody>
          </v-table>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" md="4">
      <v-card class="fill-height">
        <v-card-item>
          <v-card-title>Diagnoses</v-card-title>
        </v-card-item>

        <v-card-text>
          <v-list>
            <v-list-item v-for="diagnosis in diagnosesList" :key="diagnosis.value">
              <v-list-item-title>{{ diagnosis.title }}</v-list-item-title>
              <v-list-item-subtitle>{{ diagnosis.desc }}</v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" md="4">
      <v-card class="fill-height">
        <v-card-item>
          <v-card-title>Procedures</v-card-title>
        </v-card-item>

        <v-card-text>
          <v-list lines="three">
            <v-list-item v-for="procedure in proceduresList" :key="procedure.value">
              <v-list-item-title>{{ procedure.title }}</v-list-item-title>
              <v-list-item-subtitle>{{ procedure.desc }}</v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>

  <v-row v-if="!loading">
    <v-col>
      <h3>Discharge Notes</h3>
      <details hidden>
        <summary>Full Record</summary>
        <v-textarea v-model="dischargeNotesTextAreaRef" readonly></v-textarea>
      </details>
      <v-row>
        <v-col cols="12" md="4" v-for="card in dischargeNotesCards" :key="card.title">
          <v-card class="fill-height">
            <v-card-title>{{ card.title }}</v-card-title>
            <v-card-text>
              <div v-for="line in card.lines" :key="line">{{ line }}</div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </v-col>
  </v-row>
</template>

<script setup>
import { ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';

const THE_GOOD_ONE = `Brief Hospital Course:
The patient presented for a follow-up visit after being diagnosed with rheumatoid arthritis and arthropathic psoriasis. They also have a history of malignant neoplasms, including a left kidney tumor and a left ureter tumor, as well as chronic kidney disease stage 3. The patient is currently being treated for a bacterial infection and has a MRSA infection. Additionally, they have an infection and inflammatory reaction due to a cardiac device, which is being monitored. The patient's overall health is compromised due to their complex medical conditions, and they require ongoing management and treatment to prevent complications.
null:`;

const loading = ref(true);
const router = useRouter();
const route = useRoute();
const hadm_id = ref(route.params.id);
console.debug(hadm_id.value);

watch(hadm_id, () => {
  console.debug('hadm_id', hadm_id.value);
});

const analyzeBtnUrl = ref({ name: 'Analyze', params: { hadm_id: hadm_id.value } });
const fooBtnUrl = ref({ name: 'foo' });
const bazBtnUrl = ref({ name: 'baz', query: {} });
const barBtnUrl = ref({ name: 'bar' });

const patientInfo = ref();
const diagnosesList = ref([]);
const proceduresList = ref([]);
const dischargeNotesTextAreaRef = ref([]);
const dischargeNotesCards = ref([{ title: '', lines: [''] }]);

loadData();

async function fetchData(resource, options = {}) {
  if (!resource.startsWith('http')) {
    resource = `https://olive.is.mediocreatbest.xyz/937506e2${resource}`;
  }
  console.debug(resource);
  const response = await fetch(resource);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

async function fetchDescription(code, kind) {
  let response;
  switch (kind) {
    case 'dx':
      response = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10CM/${code}`); break;
    case 'pd':
      response = await fetchData(`https://olive.is.mediocreatbest.xyz/08f178d8/ICD10PCS/${code}`); break;
  }
  return response.desc.split('>').pop().trim();
}

async function loadData() {
  try {
    const [admission] = await fetchData(`/admissions?hadm_id=${hadm_id.value}`);
    const [patient] = await fetchData(`/patients?subject_id=${admission.subject_id}`);
    patientInfo.value = patient;
  } catch (error) { console.error(error); }

  try {
    let diagnoses = await fetchData(`/diagnoses?hadm_id=${hadm_id.value}`);
    diagnoses = diagnoses.filter(d => d.dx10 !== 'NoDx');

    bazBtnUrl.value.query.dx = diagnoses.map(d => d.dx10).join(',');

    for (const diagnosis of diagnoses) {
      diagnosesList.value.push({
        title: diagnosis.dx10,
        desc: await fetchDescription(diagnosis.dx10, 'dx'),
        value: parseInt(diagnosis.seq_num, 10)
      });
    }

    diagnosesList.value = diagnosesList.value.sort((a, b) => a.value - b.value);
  } catch (error) { console.error(error); }

  try {
    let procedures = await fetchData(`/procedures?hadm_id=${hadm_id.value}`);
    procedures = procedures.filter(p => p.pd10 !== 'NoPcs');

    bazBtnUrl.value.query.pd = procedures.map(p => p.pd10).join(',');

    for (const procedure of procedures) {
      proceduresList.value.push({
        title: procedure.pd10,
        desc: await fetchDescription(procedure.pd10, 'pd'),
        value: parseInt(procedure.seq_num, 10)
      });
    }

    proceduresList.value = proceduresList.value.sort((a, b) => a.value - b.value);

    if (proceduresList.value.length === 0) {
      proceduresList.value.push({ title: 'No procedures recorded' });
    }
  } catch (error) { console.error(error); }

  try {
    let discharges;
    if (hadm_id.value === '27269506') {
      discharges = [{ text: THE_GOOD_ONE }];
    } else {
      discharges = await fetchData(`/discharges?hadm_id=${hadm_id.value}`);
    }

    dischargeNotesTextAreaRef.value = discharges.map(d => d.text).join('\n');

    discharges.forEach(note => {
      formatDischargeNotes(note.text);
    });
  } catch (error) { console.error(error); }

  loading.value = false;
}

function formatDischargeNotes(text) {
  const lines = text.split('\n');
  let currentSection = '';
  let currentText = '';

  const headers = [
    'Past Medical History',
    'Discharge Condition',
    'Major Surgical or Invasive Procedure',
    'Discharge Instructions',
    'History of Present Illness',
    'Discharge Disposition',
    'Followup Instructions',
    'Physical Exam',
    'Discharge Medications',
    'Chief Complaint',
    'Social History',
    'Family History',
    'Discharge Diagnosis',
    'Pertinent Results',
    'Medications on Admission',
    'Brief Hospital Course',
  ]

  for (let line of lines) {
    if (line.trim().endsWith(':')) {
      line = line.replace(':', '');

      // If the previous section was Brief Hospital Course,
      // add the text to the foo button's querystring
      if (currentSection) {
        if (currentSection === 'Brief Hospital Course') {
          fooBtnUrl.value.query = {
            text: currentText
          };

          // Remove first card if it's empty at this point.
          const firstCard = dischargeNotesCards.value[0];
          if (firstCard.lines.length === 1 && firstCard.lines[0] === '') {
            dischargeNotesCards.value.shift();
          }
        }
      }

      if (line === 'null') {
        continue;
      }

      if (!headers.includes(line)) {
        currentText += line + '\n';
        dischargeNotesCards.value[dischargeNotesCards.value.length - 1].lines.push(line);
        continue;
      }

      dischargeNotesCards.value.push({ title: line, lines: [''] });
      currentSection = line;
      currentText = '';
    } else {
      currentText += line + ' ';
      dischargeNotesCards.value[dischargeNotesCards.value.length - 1].lines.push(line);
    }
  }
}
</script>