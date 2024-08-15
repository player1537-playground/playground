<template>
  <v-container fluid fill-height class=main>
    <v-app-bar app color="primary">
      <template v-slot:prepend>
        <v-img
          :src="logo_white"
          class="header-logo mx-4"
          height="32"
          width="128"
        ></v-img>
      </template>

      <v-toolbar-title>Connect</v-toolbar-title>
    </v-app-bar>

    <!-- <v-spacer></v-spacer> -->
     <div style="height: 5rem"></div>

    <!-- <v-row>
      <v-col cols=12 class=outline style="margin-bottom: 1rem">
        <h3>Patient Visit Summary</h3>
        <v-textarea v-model="summary" outlined full-width rows="6"></v-textarea>
      </v-col>
    </v-row> -->

    <v-row>
      <v-col cols="6">
        <v-row>
          <v-col offset=1 cols=10
            :class="{ outline: true, flash: flashDiagnosticSearch }"
            @animationend="flashDiagnosticSearch = false"
          >
            <h3>Search Diagnostic Codes</h3>
            <v-text-field v-model="diagnosticSearch" outlined full-width
              @keyup.enter="bing"
            >
            </v-text-field>

            <v-col cols=12>
              <v-btn color="primary" block
                @click="bing"
              >
                <v-icon class="search-icon">mdi-magnify</v-icon>
              </v-btn>
            </v-col>

          </v-col>
        </v-row>
      </v-col>
      <v-col cols=6>
        <v-row>
          <v-col offset=1 cols=10
            :class="{ outline: true, flash: flashProcedureSearch }"
            @animationend="flashProcedureSearch = false"
          >
            <h3>Search Procedure Codes</h3>
            <v-text-field v-model="procedureSearch" outlined full-width
              @keyup.enter="ping"
            >
            </v-text-field>

            <v-col cols=12>
              <v-btn color="primary" block
                @click="ping"
              >
                <v-icon class="search-icon">mdi-magnify</v-icon>
              </v-btn>
            </v-col>
            
          </v-col>
        </v-row>
      </v-col>
    </v-row>

    <v-row>
      <v-col cols="12">

        <div style="height: 1rem"></div>

        <v-row>
          <v-col cols=6>
            <v-row>
              <v-col offset="1" cols="10"
                :class="{ outline: true, flash: flashDiagnosticCodes }"
                @animationend="flashDiagnosticCodes = false"
              >
                <h3>Selected Diagnostic Codes</h3>
                <v-list class="limitHeight">
                  <v-list-item v-for="(dx, index) in diagnosticCodes" :key="index">
                    <template v-slot:prepend="{ isActive }">
                      <v-btn variant="outlined"
                        @click="diagnosticCodes.splice(index, 1)"
                      >
                        <v-icon>mdi-minus</v-icon>
                      </v-btn>
                      <v-btn variant="outlined"
                        @click="foo(dx)"
                      >
                        <v-icon>mdi-source-branch</v-icon>
                        <v-tooltip activator=parent location=end>
                          Recommend Similar-in-Meaning DX Codes
                        </v-tooltip>
                      </v-btn>
                      <!-- XXX(th): Do this without nbsp; -->
                      <pre>  </pre>
                    </template>
                    <v-list-item-title>{{ dx.dx }}: {{ dx.desc }}</v-list-item-title>
                  </v-list-item>
                </v-list>


                <div style="height: 1rem"></div>

              </v-col>
            </v-row>
          </v-col>
          <v-col cols=6>
            <v-row>
              <v-col offset="1" cols="10"
                :class="{ outline: true, flash: flashProcedureCodes }"
                @animationend="flashProcedureCodes = false"
              >
                <h3>Selected Procedure Codes</h3>
                <v-list class="limitHeight">
                  <v-list-item v-for="(pd, index) in procedureCodes" :key="index">
                    <template v-slot:prepend="{ isActive }">
                      <v-btn variant="outlined"
                        @click="procedureCodes.splice(index, 1)"
                      >
                        <v-icon>mdi-minus</v-icon>
                      </v-btn>
                      <v-btn variant="outlined"
                        @click="bar(pd)"
                      >
                        <v-icon>mdi-source-branch</v-icon>
                        <v-tooltip activator=parent location=end>
                          Recommend Similar-in-Meaning PD Codes
                        </v-tooltip>
                      </v-btn>
                      <!-- XXX(th): Do this without nbsp; -->
                      <pre>  </pre>
                    </template>
                    <v-list-item-title>{{ pd.pd }}: {{ pd.desc }}</v-list-item-title>
                  </v-list-item>
                </v-list>



                <div style="height: 1rem"></div>

              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-col>
    </v-row>

    <v-row>
      <v-col cols="12">

        <div style="height: 1rem"></div>

        <v-row>
          <v-col cols=6>
            <v-row>
              <v-col offset=1 cols="10"
                :class="{ outline: true, flash: flashRecommendedDiagnosticCodes }"
                @animationend="flashRecommendedDiagnosticCodes = false"
              >
                <h3>Recommended Diagnostic Codes</h3>
                <v-list class="limitHeight">
                  <v-list-item v-for="(dx, index) in recommendedDiagnosticCodes" :key="index">
                    <template v-slot:prepend="{ isActive }">
                      <v-btn variant="outlined"
                        @click="diagnosticCodes.push(dx)"
                      >
                        <v-icon>mdi-plus</v-icon>
                      </v-btn>
                      <!-- XXX(th): Do this without nbsp; -->
                      <pre>  </pre>
                    </template>
                    <v-list-item-title>{{ dx.dx }}: {{ dx.desc }}</v-list-item-title>
                  </v-list-item>
                </v-list>

                <div style="height: 1rem"></div>

                <v-row>
                  <v-col cols="6">
                    <v-btn color="primary" block
                      @click="dx2dx"
                    >DX Codes Recommend DX Codes</v-btn>
                  </v-col>
                  <v-col cols="6">
                    <v-btn color="primary" block
                      @click="pd2dx"
                    >PD Codes Recommend DX Codes</v-btn>
                  </v-col>
                </v-row>

              </v-col>

            </v-row>
          </v-col>
          <v-col cols=6>
            <v-row>
              <v-col offset=1 cols="10" :class="{ outline: true, flash: flashRecommendedProcedureCodes }" @animationend="flashRecommendedProcedureCodes = false">
                <h3>Recommended Procedure Codes</h3>
                <v-list class="limitHeight">
                  <v-list-item v-for="(pd, index) in recommendedProcedureCodes" :key="index">
                    <template v-slot:prepend="{ isActive }">
                      <v-btn variant="outlined"
                        @click="procedureCodes.push(pd)"
                      >
                        <v-icon>mdi-plus</v-icon>
                      </v-btn>
                      <!-- XXX(th): Do this without nbsp; -->
                      <pre>  </pre>
                    </template>
                    <v-list-item-title>{{ pd.pd }}: {{ pd.desc }}</v-list-item-title>
                  </v-list-item>
                </v-list>

                <div style="height: 1rem"></div>

                <v-row>
                  <v-col cols="6">
                    <v-btn color="primary" block
                      @click="dx2pd"
                    >DX Codes Recommend PD Codes</v-btn>
                  </v-col>
                  <v-col cols="6">
                    <v-btn color="primary" block
                      @click="pd2pd"
                    >PD Codes Recommend PD Codes</v-btn>
                  </v-col>
                </v-row>

              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { defineComponent } from 'vue';
import logo_white from '@/assets/logo_white.svg';
import ICD10CM from '@/assets/TabularICD10CM.json';
import ICD10PCS from '@/assets/ICD10PCS.json';

const DEFAULT_SUMMARY = `The patient presented for a follow-up visit after being diagnosed with rheumatoid arthritis and arthropathic psoriasis. They also have a history of malignant neoplasms, including a left kidney tumor and a left ureter tumor, as well as chronic kidney disease stage 3. The patient is currently being treated for a bacterial infection and has a MRSA infection. Additionally, they have an infection and inflammatory reaction due to a cardiac device, which is being monitored. The patient's overall health is compromised due to their complex medical conditions, and they require ongoing management and treatment to prevent complications.`;
const DEFAULT_DIAGNOSTIC_CODES = [
  { dx: 'I251', desc: `Atherosclerotic heart disease of native coronary artery` },
  { dx: 'I489', desc: `Unspecified atrial fibrillation` },
  { dx: 'Z878', desc: `Personal history of traumatic fracture` },
  { dx: 'Z951', desc: `Presence of aortocoronary bypass graft` },
  { dx: 'E785', desc: `Hyperlipidemia, unspecified` },
  { dx: 'Z790', desc: `Long term (current) use of anticoagulants` },
  { dx: 'I503', desc: `Unspecified diastolic (congestive) heart failure` },
  { dx: 'N400', desc: `Benign prostatic hyperplasia without lower urinary tract symptoms` },
  { dx: 'G409', desc: `Epilepsy, unspecified, not intractable` },
  { dx: 'Z950', desc: `Presence of cardiac pacemaker` },
  { dx: 'Y920', desc: `Kitchen of unspecified non-institutional (private) residence as the place of occurrence of the external cause` },
  { dx: 'I130', desc: `Hypertensive heart and chronic kidney disease with heart failure and stage 1 through stage 4 chronic kidney disease, or unspecified chronic kidney disease` },
  { dx: 'Z858', desc: `Personal history of malignant neoplasm` },
  { dx: 'D649', desc: `Anemia, unspecified` },
  { dx: 'R791', desc: `Abnormal coagulation profile` },
  { dx: 'E875', desc: `Hyperkalemia` },
  { dx: 'N189', desc: `Chronic kidney disease, unspecified` },
  { dx: 'L031', desc: `Cellulitis` },
  { dx: 'W010', desc: `Fall on same level from slipping, tripping and stumbling without subsequent striking against object` },
  { dx: 'T455', desc: `Description of the parent code 'T455' with a focus on its reported codes and their related descriptions.` },
  { dx: 'G250', desc: `Essential tremor` },
  { dx: 'S810', desc: `Description: Unspecified open wound, right knee, initial encounter` },
];
const DEFAULT_PROCEDURE_CODES = [
  { pd: '0JB', desc: `Excision of Right Lower Leg Subcutaneous Tissue and Fascia, Open Approach` },
];


export default defineComponent({
  name: 'BazView',

  data() {
    let diagnosticCodes = (
      DEFAULT_DIAGNOSTIC_CODES
        .slice(0, 5)
    );

    if (this.$route.query.hasOwnProperty('dx')) {
      diagnosticCodes.splice(0);
      if (this.$route.query.dx)
      for (let dx of this.$route.query.dx.split(',')) {
        let desc = ICD10CM[dx];
        dx = dx.substring(0, 4);
        diagnosticCodes.push({ dx, desc });
      }
    }

    let procedureCodes = (
      DEFAULT_PROCEDURE_CODES
        .slice(0, 5)
    );

    if (this.$route.query.hasOwnProperty('pd')) {
      procedureCodes.splice(0);
      if (this.$route.query.pd)
      for (let pd of this.$route.query.pd.split(',')) {
        let desc = ICD10PCS[pd];
        pd = pd.substring(0, 3);
        procedureCodes.push({ pd, desc });
      }
    }

    return {
      summary: DEFAULT_SUMMARY,
      diagnosticCodes,
      procedureCodes,
      recommendedDiagnosticCodes: [
      ],
      recommendedProcedureCodes: [
      ],
      diagnosticSearch: '',
      procedureSearch: '',
      flashDiagnosticSearch: false,
      flashProcedureSearch: false,
      flashDiagnosticCodes: false,
      flashProcedureCodes: false,
      flashRecommendedProcedureCodes: false,
      flashRecommendedDiagnosticCodes: false,
      logo_white,
    };
  },

  methods: {

    async dx2pd() {
      let {
        diagnosticCodes,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashDiagnosticCodes: true,
        flashRecommendedProcedureCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/dx2pd/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dx: diagnosticCodes.map(({ dx }) => dx),
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let best of json.best) {
        recommendedProcedureCodes.push({
          pd: best.pd,
          desc: best.desc,
        });
      }
    },

    async pd2dx() {
      let {
        procedureCodes,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashProcedureCodes: true,
        flashRecommendedDiagnosticCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/pd2dx/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pd: procedureCodes.map(({ pd }) => pd),
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let best of json.best) {
        recommendedDiagnosticCodes.push({
          dx: best.dx,
          desc: best.desc,
        });
      }
    },

    async dx2dx() {
      let {
        diagnosticCodes,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashDiagnosticCodes: true,
        flashRecommendedDiagnosticCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/dx2dx/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dx: diagnosticCodes.map(({ dx }) => dx),
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let best of json.best) {
        recommendedDiagnosticCodes.push({
          dx: best.dx,
          desc: best.desc,
        });
      }
    },

    async pd2pd() {
      let {
        procedureCodes,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashProcedureCodes: true,
        flashRecommendedProcedureCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/pd2pd/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pd: procedureCodes.map(({ pd }) => pd),
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let best of json.best) {
        recommendedProcedureCodes.push({
          pd: best.pd,
          desc: best.desc,
        });
      }
    },

    async foo(diagnosticCode) {
      let {
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashDiagnosticCodes: true,
        flashRecommendedDiagnosticCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/foo/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dx: diagnosticCode.dx,
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let code of json.best) {
        recommendedDiagnosticCodes.push({
          dx: code.dx,
          desc: code.desc,
        });
      }
    },

    async bar(procedureCode) {
      let {
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashProcedureCodes: true,
        flashRecommendedProcedureCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/bar/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pd: procedureCode.pd,
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let code of json.best) {
        recommendedProcedureCodes.push({
          pd: code.pd,
          desc: code.desc,
        });
      }
    },

    async bing() {
      let {
        diagnosticSearch: search,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashDiagnosticSearch: true,
        flashRecommendedDiagnosticCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/bing/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          search,
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let code of json.best) {
        recommendedDiagnosticCodes.push({
          dx: code.dx,
          desc: code.desc,
        });
      }
    },

    async ping() {
      let {
        procedureSearch: search,
        recommendedDiagnosticCodes,
        recommendedProcedureCodes,
      } = this;

      recommendedDiagnosticCodes.splice(0);
      recommendedProcedureCodes.splice(0);
      Object.assign(this, {
        flashProcedureSearch: true,
        flashRecommendedProcedureCodes: true,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/ping/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          search,
        }),
      });

      let response = await fetch(request);

      let json = await response.json();

      for (let code of json.best) {
        recommendedProcedureCodes.push({
          pd: code.pd,
          desc: code.desc,
        });
      }
    },
  
  },

});
</script>

<style scoped>
.limitHeight {
  max-height: 25vh;
  min-height: 25vh;
  overflow-y: scroll;
  box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.3);
}
@keyframes flash {
  0% {
    background-color: transparent;
  }
  50% {
    background-color: yellow;
  }
  100% {
    background-color: transparent;
  }
}
.flash {
  animation: flash 0.5s ease-out;
}
.outline {
  border: 1px solid #707D87;
  border-radius: 0.5rem;
  padding: 1rem;
  background-color: rgb(var(--v-theme-surface-light));
}
.main {
}
</style>
