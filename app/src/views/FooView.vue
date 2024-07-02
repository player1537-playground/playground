<template>
  <v-container fluid fill-height>
    <v-row>
      <v-col cols=12>
        <h1>Page 1: Recommend Codes given Visit Summary</h1>
      </v-col>
    </v-row>
    <v-row>
      <v-col cols="3">
        <v-card>
          <v-card-title>Passages</v-card-title>
          <v-card-text>
            <v-list>
              <v-list-item
                v-for="(passage, index) in passages"
                :key="index"
                @click="text = passage"
              >
                <v-list-item-title>{{ passage }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="6">
        <v-textarea
          v-model="text"
          :rows="10"
          auto-grow
        ></v-textarea>
        <v-col cols="12" class="text-center">
          <v-btn
            color="primary"
            @click="analyzeText"
            :style="{ width: '100%' }"
            :disabled="isInRequest"
          >
            Analyze Text
          </v-btn>
        </v-col>
        <v-col cols="12" class="text-center">
          <v-card>
            <v-card-title>Best Matches</v-card-title>
            <v-card-text>
              <v-list>
                <v-list-item
                  v-for="(match, index) in best"
                  :key="index"
                >
                  <v-list-item-title>
                    {{ match.code }}: {{ match.desc }}
                  </v-list-item-title>
                  <v-list-item-subtitle>
                    Distance: {{ match.distance }}
                  </v-list-item-subtitle>
                  <v-list-item-action>
                    <v-btn
                      color="primary"
                      @click="codes.push({ code: match.code, display: match.desc })"
                    >
                      Add
                    </v-btn>
                  </v-list-item-action>
                  <span class="font-weight-light">{{ text.substring(0, match.chunk.offset) }}</span>
                  <span class="font-weight-bold">{{ text.substring(match.chunk.offset, match.chunk.offset + match.chunk.length) }}</span>
                  <span class="font-weight-light">{{ text.substring(match.chunk.offset + match.chunk.length) }}</span>
                </v-list-item>
              </v-list>
            </v-card-text>
          </v-card>
        </v-col>
      </v-col>
      <v-col cols=3>
        <v-card>
          <v-card-title>Codes</v-card-title>
          <v-card-text>
            <v-chip
              v-for="code in codes"
              :key="code.code"
              color="primary"
              class="ma-1"
            >
              {{ code.display }}
            </v-chip>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { defineComponent } from 'vue';

const PASSAGES = [];
PASSAGES.push(`The patient presented for a follow-up visit after being diagnosed with rheumatoid arthritis and arthropathic psoriasis. They also have a history of malignant neoplasms, including a left kidney tumor and a left ureter tumor, as well as chronic kidney disease stage 3. The patient is currently being treated for a bacterial infection and has a MRSA infection. Additionally, they have an infection and inflammatory reaction due to a cardiac device, which is being monitored. The patient's overall health is compromised due to their complex medical conditions, and they require ongoing management and treatment to prevent complications.`);
PASSAGES.push(`The patient presented for a routine follow-up visit. They have a history of chronic kidney disease, stage 3, and hypertension, which is well-controlled with medication. They also have a history of traumatic fracture, malignant neoplasm, and hypertension. In the past year, they were hospitalized for acute respiratory failure due to COVID-19. Currently, they are experiencing symptoms of pneumonia and are being treated accordingly. Additionally, they have a history of thrombocytopenia, unspecified. The patient also reported having a history of inhalation of air with the intention of reducing the pressure of the body or the lungs. They are up-to-date on their immunizations. The provider will continue to monitor their chronic kidney disease and hypertension, and adjust medication as needed.`);
PASSAGES.push(`This patient is an elderly individual with a history of age-related cognitive decline and long-term use of selective estrogen receptor modulators (SERMs). They presented to the emergency department with acute respiratory failure and were diagnosed with COVID-19. During their stay, they developed sepsis due to Enterococcus, which led to acute kidney failure and hyperosmolality and hypernatremia. The patient also had reduced appetite and dehydration. Additionally, they experienced asymptomatic microscopic hematuria and cerebral infarction. Due to their advanced age and comorbidities, the patient's code status was changed to do not resuscitate. The patient was transferred to the intensive care unit for further management and was receiving palliative care for their symptoms. Unfortunately, the patient's condition continued to decline, and they developed immune effector cell-associated neurotoxicity syndrome and acidosis. Despite aggressive treatment, the patient's condition was deemed unstable, and they were ultimately discharged to hospice care with a diagnosis of unspecified dementia. The patient was also identified as a tobacco user.`);
PASSAGES.push(`The patient presented for a routine follow-up visit, reporting a history of well-controlled Type 2 diabetes mellitus without complications. They also mentioned being managed for Essential (primary) hypertension and Adult failure to thrive. The patient had a recent Urinary tract infection, site not specified, which has been treated and is currently resolving. Additionally, they have been experiencing mild symptoms of Unspecified glaucoma and have been referred to an ophthalmologist for further evaluation. The patient also complained of joint pain consistent with Arthritis. Notably, they have been experiencing cognitive decline and have been diagnosed with Unspecified dementia. The patient had a recent episode of Syncope and collapse, which was investigated and found to be related to an abnormal blood-gas level. The patient is currently being monitored and managed by multiple healthcare providers.`);
PASSAGES.push(`The patient presented for a routine follow-up visit, where they reported a recent exposure to a communicable disease. Laboratory tests revealed asymptomatic HIV infection, and electrocardiogram showed signs of complete atrioventricular block. The patient's medical history includes chronic kidney disease, unspecified, and a family history of stroke. Additionally, they were diagnosed with opioid dependence and cocaine abuse. The patient's blood work showed signs of acute kidney failure, and they were found to have type 2 diabetes mellitus. The patient was advised to continue treatment for their chronic kidney disease and to follow up with their primary care physician for further monitoring and management of their comorbidities.`);
PASSAGES.push(`A 30-year-old female presented to the emergency department with a recent delivery of a single liveborn infant via cesarean section. The infant was delivered without complications. However, the patient reported that the infant's immunizations were not completed as scheduled, with the patient citing an unknown reason for not receiving the immunizations. Further evaluation and follow-up were recommended to address the immunization status and ensure the infant's overall health and well-being.`);
PASSAGES.push(`The patient presented to the clinic with a complex medical history. They have a history of cardiomyopathy due to drug and external agent use, as well as a history of alcohol dependence. They also have gastro-esophageal reflux disease without esophagitis, a personal history of malignant neoplasm of breast, and dyspnea. Additionally, they have been diagnosed with unspecified asthma and have recently been treated for COVID-19. The patient's respiratory status has been further compromised by acute respiratory failure, and they have also been poisoned by antineoplastic and immunosuppressive drugs. They are currently experiencing pneumonia. The patient's medical history is significant for a personal history of irradiation. Today, the patient's symptoms of dyspnea and cough have worsened, and they are experiencing chest tightness. A thorough examination and diagnostic workup are needed to fully evaluate the patient's condition and develop an appropriate treatment plan.`);
PASSAGES.push(`This patient is a 65-year-old male who presented for a routine follow-up visit. He has a history of hypertension, hyperlipidemia, and type 2 diabetes mellitus, which is well-controlled. He has also been experiencing chronic kidney disease, but his kidney function has been stable. He has a personal history of cardiovascular disease and has undergone antineoplastic chemotherapy for a previous diagnosis of diffuse large B-cell lymphoma. He is currently taking anticoagulants and has a history of benign prostatic hyperplasia. He also has a presence of an implantable cardiac defibrillator. During the visit, he reported a recent onset of acute cough, which is being investigated. He also has a pressure ulcer on his back, which is being treated. He is otherwise in good health and has no other significant symptoms.`);
PASSAGES.push(`A 37-year-old female presented to the clinic for a postpartum visit following a cesarean delivery. She reported difficulty breastfeeding her newborn, noting that the infant was having trouble latching and feeding at the breast. The infant was found to be hypothermic, and the mother reported exposure to environmental tobacco smoke during the perinatal period. Physical examination of the newborn revealed hypotonia. The patient was given instructions on proper breastfeeding techniques and referred to a lactation specialist for further support. She also received a routine immunization at this visit.`);

export default defineComponent({
  name: 'FooView',

  data() {
    return {
      passages: PASSAGES,
      text: PASSAGES[0],
      codes: [],
      isInRequest: false,
      best: [],
    };
  },

  methods: {

    async analyzeText() {
      let {
        text,
        best,
        isInRequest,
      } = this;

      try {
        best = [];
        isInRequest = true;
        Object.assign(this, {
          best,
          isInRequest,
        });

        let response = await fetch('https://purple.is.mediocreatbest.xyz/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            document: text,
          }),
        });

        let json = await response.json();
        console.log({ json });

        best = json.best;
        Object.assign(this, {
          best,
        });

      } finally {
        isInRequest = false;

        Object.assign(this, {
          isInRequest,
        });
      }
    },
  
  },

});
</script>
