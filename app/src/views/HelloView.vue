<template>
    <div>
        <h1>Demo Applications</h1>
        <ul>
            <li>
                <router-link to="/foo/">Recommend diagnosis codes based on patient visit summary</router-link>
            </li>
            <li>
                <router-link to="/bar/">Recommend diagnosis codes based on patient's location</router-link>
            </li>
            <li>
                <router-link to="/baz/">Recommend diagnosis (and procedure) codes based on diagnosis (and procedure) codes</router-link>
            </li>
        </ul>
    </div>
</template>

<script>
import { defineComponent } from 'vue';

export default defineComponent({
    name: 'HelloView',
});
</script>
