<template>
  <div style="height: 3rem"></div>

  <div class="grid">
    <div class="a">

      <v-expansion-panels multiple>
        <v-expansion-panel v-for="whatever in whatevers" style="padding-right: 0;">
          <v-expansion-panel-title>
            <template #actions>
              <v-btn @click.stop="show('whatever', whatever.identity)">
                <v-icon icon=mdi-eye-arrow-right
                  :color="overrideValues.get(whatever.identity) ? 'default' : 'grey'"></v-icon>
              </v-btn>
            </template>

            {{ whatever.identity }}
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <v-expansion-panels multiple>
              <v-expansion-panel v-for="magazine in whatever.magazines" :key="magazine.identity">
                <v-expansion-panel-title>
                  <template #actions>
                    <v-btn @click.stop="show('magazine', whatever.identity, magazine.identity)">
                      <v-icon icon=mdi-eye-arrow-right
                        :color="overrideValues.get(whatever.identity + '-' + magazine.identity) ? 'default' : 'grey'"></v-icon>
                    </v-btn>
                  </template>

                  {{ magazine.identity }}
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <v-slider :model-value="sliderValues.get(whatever.identity + '-' + magazine.identity)"
                    @update:model-value="v => sliderValues.set(whatever.identity + '-' + magazine.identity, v)"
                    :min=-1.5 :max=1.5 :step=0.1 thumb-label="always" show-ticks
                    style="margin-left: 1rem; margin-right: 1rem; margin-top: 0.5rem;"></v-slider>

                  <v-list>
                    <v-list-item v-for="property in magazine.properties" :key="property.identity">
                      <template #append>
                        <v-btn @click.stop="show('property', whatever.identity, magazine.identity, property.identity)">
                          <v-icon icon=mdi-eye-arrow-right
                            :color="overrideValues.get(whatever.identity + '-' + magazine.identity + '-' + property.identity) ? 'default' : 'grey'"></v-icon>
                        </v-btn>
                      </template>

                      {{ property.fullname }}
                      <v-slider
                        :model-value="sliderValues.get(whatever.identity + '-' + magazine.identity + '-' + property.identity)"
                        @update:model-value="v => sliderValues.set(whatever.identity + '-' + magazine.identity + '-' + property.identity, v)"
                        :min=-1.5 :max=1.5 :step=0.1 thumb-label="always" show-ticks
                        style="margin-left: 1rem; margin-right: 1rem; margin-top: 0.5rem;"></v-slider>
                    </v-list-item>
                  </v-list>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
          </v-expansion-panel-text>
        </v-expansion-panel>

      </v-expansion-panels>
    </div>
    <div class="b">
      <canvas :width="1600 * 2 / 3" :height="900 * 2 / 3" ref="geograph"></canvas>
    </div>
    <div class="c">
    </div>
  </div>
</template>

<script>
import config from '@/config';
import { defineComponent, readonly, markRaw } from 'vue';
import logo_white from '@/assets/logo_white.svg';
import YAML from 'yaml';
import * as vega from 'vega';
import Z6372 from '@/assets/Z6372.yaml?raw';
import Z6833 from '@/assets/Z6833.yaml?raw';
import Z6842 from '@/assets/Z6842.yaml?raw';
import O9882 from '@/assets/O9882.yaml?raw';
import Z550 from '@/assets/Z550.yaml?raw';
import Z560 from '@/assets/Z560.yaml?raw';
import Z570 from '@/assets/Z570.yaml?raw';
import Z590 from '@/assets/Z590.yaml?raw';
import Z600 from '@/assets/Z600.yaml?raw';
import Z620 from '@/assets/Z620.yaml?raw';
import Z630 from '@/assets/Z630.yaml?raw';
import Z640 from '@/assets/Z640.yaml?raw';
import Z650 from '@/assets/Z650.yaml?raw';

function Whatever(what) {
  let whatever = {};
  whatever.identity = what.identity;
  whatever.magazines = [];

  for (let m of what.magazines) {
    let magazine = {};
    magazine.identity = m.identity;
    magazine.properties = [];

    for (let p of m.properties) {
      // e.g. p.identity = "P0A91B2A: Social Vulnerability Percentile Within State"
      let [identity, fullname] = p.identity.split(': ', 2);
      // e.g. identity = "P02B9A83"
      // e.g. fullname = "Social Vulnerability Percentile Within State"

      let property = {};
      property.identity = identity;
      property.fullname = fullname;

      magazine.properties.push(property);
    }

    whatever.magazines.push(magazine);
  }

  return whatever;
}

export default defineComponent({

  name: 'BarView',

  data() {
    let whatevers = [];
    whatevers.push(readonly(Whatever(YAML.parse(Z6372))))
    // whatevers.push(readonly(Whatever(YAML.parse(Z6833))))
    // whatevers.push(readonly(Whatever(YAML.parse(Z6842))))
    // whatevers.push(readonly(Whatever(YAML.parse(O9882))))
    whatevers.push(readonly(Whatever(YAML.parse(Z550))))
    whatevers.push(readonly(Whatever(YAML.parse(Z560))))
    whatevers.push(readonly(Whatever(YAML.parse(Z570))))
    whatevers.push(readonly(Whatever(YAML.parse(Z590))))
    whatevers.push(readonly(Whatever(YAML.parse(Z600))))
    whatevers.push(readonly(Whatever(YAML.parse(Z620))))
    whatevers.push(readonly(Whatever(YAML.parse(Z630))))
    whatevers.push(readonly(Whatever(YAML.parse(Z640))))
    whatevers.push(readonly(Whatever(YAML.parse(Z650))))

    return {
      whatevers,
      sliderValues: new Map(),
      overrideValues: new Map(),
      needFingleTimeout: null,
      ctx: null,
      logo_white,
    };
  },

  watch: {

    whatevers: {
      immediate: true,
      handler() {
        this.needSliderValues();
        this.needOverrideValues();
        this.needFingle();
      },
    },

    sliderValues: {
      deep: true,
      handler() {
        this.needFingle();
      },
    },

    overrideValues: {
      deep: true,
      handler() {
        this.needFingle();
      },
    },

  },

  methods: {

    needSliderValues() {
      let {
        sliderValues,
        whatevers,
      } = this;

      for (let whatever of whatevers) {
        for (let magazine of whatever.magazines) {
          let identity = whatever.identity + '-' + magazine.identity;
          if (!sliderValues.has(identity)) {
            sliderValues.set(identity, 1.0);
          }

          for (let property of magazine.properties) {
            let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
            if (!sliderValues.has(identity)) {
              sliderValues.set(identity, 1.0);
            }
          }
        }
      }
    },

    needOverrideValues() {
      let {
        overrideValues,
        whatevers,
      } = this;

      let theDefault = true;
      for (let whatever of whatevers) {
        let identity = whatever.identity;
        if (!overrideValues.has(identity)) {
          overrideValues.set(identity, theDefault);
        }

        for (let magazine of whatever.magazines) {
          let identity = whatever.identity + '-' + magazine.identity;
          if (!overrideValues.has(identity)) {
            overrideValues.set(identity, theDefault);
          }

          for (let property of magazine.properties) {
            let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
            if (!overrideValues.has(identity)) {
              overrideValues.set(identity, theDefault);
            }
          }
        }

        theDefault = false;
      }
    },

    show(kind, ...identities) {
      let {
        whatevers,
        overrideValues,
      } = this;

      for (let whatever of whatevers) {
        let identity = whatever.identity;
        overrideValues.set(identity, false);

        for (let magazine of whatever.magazines) {
          let identity = whatever.identity + '-' + magazine.identity;
          overrideValues.set(identity, false);

          for (let property of magazine.properties) {
            let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
            overrideValues.set(identity, false);
          }
        }
      }

      if (kind === 'whatever') {
        let [whateverIdentity] = identities;
        for (let whatever of whatevers) {
          if (whatever.identity !== whateverIdentity) {
            continue;
          }

          let identity = whatever.identity;
          overrideValues.set(whatever.identity, true);

          for (let magazine of whatever.magazines) {
            let identity = whatever.identity + '-' + magazine.identity;
            overrideValues.set(identity, true);

            for (let property of magazine.properties) {
              let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
              overrideValues.set(identity, true);
            }
          }
        }
      }

      if (kind === 'magazine') {
        let [whateverIdentity, magazineIdentity] = identities;
        for (let whatever of whatevers) {
          if (whatever.identity !== whateverIdentity) {
            continue;
          }

          let identity = whatever.identity;
          overrideValues.set(whatever.identity, true);

          for (let magazine of whatever.magazines) {
            if (magazine.identity !== magazineIdentity) {
              continue;
            }

            let identity = whatever.identity + '-' + magazine.identity;
            overrideValues.set(identity, true);

            for (let property of magazine.properties) {
              let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
              overrideValues.set(identity, true);
            }
          }
        }

      } else if (kind === 'property') {
        let [whateverIdentity, magazineIdentity, propertyIdentity] = identities;

        for (let whatever of whatevers) {
          if (whatever.identity !== whateverIdentity) {
            continue;
          }

          let identity = whatever.identity;
          overrideValues.set(whatever.identity, true);

          for (let magazine of whatever.magazines) {
            if (magazine.identity !== magazineIdentity) {
              continue;
            }

            let identity = whatever.identity + '-' + magazine.identity;
            overrideValues.set(identity, true);

            for (let property of magazine.properties) {
              if (property.identity !== propertyIdentity) {
                continue;
              }

              let identity = whatever.identity + '-' + magazine.identity + '-' + property.identity;
              overrideValues.set(identity, true);
            }
          }
        }
      }
    },

    needFingle() {
      let {
        needFingleTimeout: timeout,
      } = this;

      clearTimeout(timeout);
      timeout = setTimeout(() => {
        this._needFingle();
      }, 300);

      Object.assign(this, {
        needFingleTimeout: timeout,
      });
    },

    async _needFingle() {
      let {
        whatevers,
        ctx,
        $refs: {
          geograph,
        },
      } = this;

      let whatever = whatevers.find(whatever => this.overrideValues.get(whatever.identity));

      let url = new URL(config.API_URL);
      url.pathname += `fingle/`;
      url = url.toString();

      let request = new Request(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          whatever: {
            identity: whatever.identity,
            magazines: whatever.magazines.map(magazine => ({
              identity: magazine.identity,
              multiply: (
                this.overrideValues.get(whatever.identity + '-' + magazine.identity)
                  ? this.sliderValues.get(whatever.identity + '-' + magazine.identity)
                  : 0.0
              ),
              properties: magazine.properties.map(property => ({
                identity: property.identity,
                multiply: (
                  this.overrideValues.get(whatever.identity + '-' + magazine.identity + '-' + property.identity)
                    ? this.sliderValues.get(whatever.identity + '-' + magazine.identity + '-' + property.identity)
                    : 0.0
                ),
              })),
            })),
          },
        }),
      });

      let response = await fetch(request);

      // get image, draw to ctx
      let image = await response.blob();
      url = URL.createObjectURL(image);
      let img = new Image();
      let promise = new Promise((resolve) => {
        img.onload = resolve;
      });
      img.src = url;
      await promise;

      ctx = geograph.getContext('2d');
      ctx.drawImage(img, 0, 0, geograph.width, geograph.height);
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
    30% 1fr;
  grid-template-rows:
    1fr 1fr;
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
