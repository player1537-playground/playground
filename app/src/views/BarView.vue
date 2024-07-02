<template>
  <div class="grid">
    <div class="a">

      <v-expansion-panels
        multiple
      >
        <v-expansion-panel
          style="padding-right: 0;"
        >
          <v-expansion-panel-title>
            <template #actions>
              <v-btn
                @click.stop="console.log('click')"
              >
                <v-icon
                  icon=mdi-eye-arrow-right
                ></v-icon>
              </v-btn>
            </template>

            {{ whatever.identity }}
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <v-expansion-panels
              multiple
            >
              <v-expansion-panel
                v-for="magazine in whatever.magazines"
                :key="magazine.identity"
              >
                <v-expansion-panel-title>
                  <template #actions>
                    <v-btn
                      @click.stop="console.log('click')"
                    >
                      <v-icon
                        icon=mdi-eye-arrow-right
                      ></v-icon>
                    </v-btn>
                  </template>

                  {{ magazine.identity }}
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <v-list>
                    <v-list-item
                      v-for="property in magazine.properties" 
                      :key="property.identity"
                    >
                      <template #append>
                        <v-btn
                          @click.stop="console.log('click')"
                        >
                          <v-icon
                            icon=mdi-eye-arrow-right
                          ></v-icon>
                        </v-btn>
                      </template>

                      {{ property.identity }}
                      <v-slider
                        :min=-1.5
                        :max=1.5
                        :step=0.1
                        thumb-label="always"
                        show-ticks
                        style="margin-left: 1rem; margin-right: 1rem; margin-top: 0.5rem;"
                      ></v-slider>
                    </v-list-item>
                  </v-list>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
          </v-expansion-panel-text>
        </v-expansion-panel>

      </v-expansion-panels>
    </div>
    <div class="b" ref="geograph">
    </div>
    <div class="c">
      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue. Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim. Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh. Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam sodales hendrerit.
    </div>
  </div>
</template>

<script>
import { defineComponent, readonly, markRaw } from 'vue';
import YAML from 'yaml';
import * as vega from 'vega';

let DEFAULT_WHATEVER = YAML.parse(String.raw`

identity: "Z6372: Alcoholism and drug addiction in family"

magazines:
- identity: "Poverty Rate"
  properties:
  - identity: "P07FDFE2: Poverty Level Percent (2 Decimal Places With Decimal Point), Rounded"
  - identity: "P074936D: Proportion Of Population Below Poverty Level"
  - identity: "P04529F4: Population Where Income Is Below Poverty Level"
  - identity: "P071CB3E: Percent of Population At or Below Poverty Threshold"

- identity: "Unemployment Rate"
  properties:
  - identity: "P0204018: Proportion Of Population 16+ Unemployed"
  - identity: "P031BE1E: Total Unemployed Male Population 16 And Over"
  - identity: "P04E1477: Total Female Population 16 And Over - Unemployed"
  - identity: "P0D8F40D: Hail - Expected Annual Loss Rate - Population"

- identity: "Education Level"
  properties:
  - identity: "P0F3A328: Number of High Schools (Highest Grade > 9)"
  - identity: "P037A4B0: Adult (25+ Years) Female Population With Only 11th Grade Education"
  - identity: "P0D3A301: Proportion With Less Than High School Diploma"
  - identity: "P0FB0DD9: Adult (25+ Years) Female Population With Only 10th Grade Education"

- identity: "Median Household Income"
  properties:
  - identity: "P0D61060: Median Household Income"
  - identity: "P04F7D2E: Median Family Income"
  - identity: "P056137B: Median Household Income, Householder Of Some Other Race"

- identity: "Percent of Population with Less than a High School Diploma"
  properties:
  - identity: "P0D3A301: Proportion With Less Than High School Diploma"
  - identity: "P0FF22E3: Proportion With High School Diploma And/Or Some College"
  - identity: "P097BBE6: Proportion With Bachelor'S Degree Or Higher"
  - identity: "P0F3A328: Number of High Schools (Highest Grade > 9)"

- identity: "Percent of Population Living in Multigenerational Households"
  properties:
  - identity: "P089E6E5: % Households With Any Computing Device (Laptop/Desktop Or Smartphone/Tablet)"
  - identity: "P0E750E0: % Households With Any Computing Device And Any Type Of Broadband"
  - identity: "P0049744: Total Population In Households"
  - identity: "P05618F4: Percent of Senior (Aged 65+ Years) Population in Tract"

- identity: "Crime Rate"
  properties:
  - identity: "P0B3AF65: Number of Jails/Prisons Per 1000 People"
  - identity: "P00F3074: Number of Active Other Law Enforcement Orgs Per 1000 People"
  - identity: "P08E3D49: Number of Active Law Enforcement Organizations (All Kinds) per 1000 People"
  - identity: "P07ABD58: Number of Law Enforcement Organizations (All Kinds) Per 1000 People"

- identity: "Access to Healthcare"
  properties:
  - identity: "P0C8F522: Is Low-Access (>1 mile) to Supermarket"
  - identity: "P0FF2693: Is Low-Access (>1/2 mile) to Supermarket"
  - identity: "P0404619: Population with Low-Access (>1 mile) to Supermarket"
  - identity: "P04AE5E8: Population with Low-Access (>1/2 mile) to Supermarket"

- identity: "Food Insecurity Rate"
  properties:
  - identity: "P01FDD41: Population with Low Food Access"
  - identity: "P02AA73A: Social Vulnerability And Community Resilience Adjusted Expected Annual Loss Rate - National Percentile - Composite"
  - identity: "P01DA57D: Community Resilience - Value"
  - identity: "P0428954: Percent of Population with Low-Income and Low-Access (>1/2 mile) to Supermarket"

- identity: "Housing Instability Rate"
  properties:
  - identity: "P0D69D37: Landslide - Expected Annual Loss Rate - Building"
  - identity: "P0D73E35: Total Renter-Occupied Housing Units With Household Income Less Than $10,000 - 50.0 Or More Percent"
  - identity: "P0BC4B5F: Total Renter-Occupied Housing Units With Household Income Less Than $10,000 - Less Than 20 Percent"
  - identity: "P0ADA95A: Total Renter-Occupied Housing Units With Household Income $100,000 Or More - Less Than 20.0 Percent"

- identity: "Social Isolation Rate"
  properties:
  - identity: "P0A91B2A: Social Vulnerability Percentile Within State"
  - identity: "P02B9A83: Social Vulnerability Score"
  - identity: "P02AA73A: Social Vulnerability And Community Resilience Adjusted Expected Annual Loss Rate - National Percentile - Composite"
  - identity: "P052441F: Number of All Social Services with Employees Per 1000 People"
`);

function Whatever(what) {
  let whatever = {};
  whatever.identity = what.identity;
  whatever.magazines = [];

  for (let m of what.magazines) {
    let magazine = {};
    magazine.identity = m.identity;
    magazine.properties = [];

    for (let p of m.properties) {
      let identity = p.identity;
      // identity = "P0A91B2A: Social Vulnerability Percentile Within State"
      identity = identity.split(': ', 1)[0];
      // identity = "P02B9A83"

      let property = {};
      property.identity = identity;
      magazine.properties.push(property);
    }

    whatever.magazines.push(magazine);
  }

  return whatever;
}

function GEOGRAPH({
  whatever,
}) {
  let params = new Map();
  let expr = [];
  let emit = expr.push.bind(expr);

  emit(`0`);
  for (let i=0, n=whatever.magazines.length; i<n; i++) {
    let magazine = whatever.magazines[i];

    let name = `m${i}`;
    params.set(name, {
      name,
      value: 1.0,
    });

    emit(`+ ${name} * (0`);

    for (let property of magazine.properties) {
      let name = `m${property.identity}`;
      params.set(name, {
        name,
        value: 1.0,
      });

      emit(`+ ${name} * datum[${JSON.stringify(property.identity)}]`);
    }

    emit(`)`)
  }

  expr = expr.join('');
  console.log({ expr, params });
  
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    width: 600,
    height: 400,
    params,
    datasets: {
      fingle: [],
    },
    data: {
      name: 'fingle',
    },
    mark: {
      type: 'geoshape',
      tooltip: { content: 'data' },
    },
    projection: {
      type: 'albersUsa',
    },
    transform: [
      { calculate: expr, as: "value" },
    ],
    encoding: {
      color: {
        field: 'value',
        type: 'quantitative',
      },
    },
  };
} // GEOGRAPH

export default defineComponent({

  name: 'BarView',

  data() {
    return {
      whatever: readonly(Whatever(DEFAULT_WHATEVER)),
      sliderValue: 50,
      needFingleTimeout: null,
      fingle: null,
      geograph: null,
      measure: null,
    };
  },

  mounted() {
    let {
      geograph,
    } = this.$refs;
    let {
      whatever,
    } = this;

    geograph = new vega.View(vega.parse(GEOGRAPH({ whatever })))
      .initialize(geograph)
      .renderer('svg')
      .hover()
      .run();
    geograph = markRaw(geograph);

    Object.assign(this, {
      geograph,
    });
  },

  beforeUnmount() {
    let {
      geograph,
    } = this;

    geograph.finalize();
  },

  watch: {

    whatever: {
      immediate: true,
      handler() {
        this.needFingle();
      },
    },

    fingle() {
      let {
        fingle,
        geograph,
      } = this;

      if (!fingle) return;
      if (!geograph) return;

      geograph
        .data('fingle', fingle)
        .run();
    },

  },

  methods: {

    needFingle() {
      let {
        needFingleTimeout: timeout,
      } = this;

      clearTimeout(timeout);
      timeout = setTimeout(() => {
        this._needFingle();
      }, 0);

      Object.assign(this, {
        needFingleTimeout: timeout,
      });
    },

    async _needFingle() {
      let {
        whatever,
        fingle,
      } = this;

      fingle = null;

      Object.assign(this, {
        fingle,
      });

      let request = new Request('https://purple.is.mediocreatbest.xyz/fingle/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          whatever,
        }),
      });

      let response = await fetch(request);

      fingle = await response.json();
      fingle = readonly(fingle);

      Object.assign(this, {
        fingle,
      });
    },

  },

});
</script>

<style scoped>
.grid {
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-areas:
    "a b"
    "a c"
  ;
  grid-template-columns:
    30% 1fr
  ;
  grid-template-rows:
    1fr
    1fr
  ;
  gap: 1rem;
}

/* .grid > * {
  width: 100%;
  height: 100%;
} */

.a {
  grid-area: a;
  max-height: 100vh;
  overflow-y: scroll;
}

.b {
  grid-area: b;
}

.c {
  grid-area: c;
}
</style>
