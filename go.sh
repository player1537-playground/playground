#!/usr/bin/env bash
# vim :set ts=4 sw=4 sts=4 et:
die() { printf $'Error: %s\n' "$*" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
self=${BASH_SOURCE[0]:?}
project=${root##*/}
pexec() { >&2 printf exec; >&2 printf ' %q' "$@"; >&2 printf '\n'; exec "$@"; }
prun() { >&2 printf exec; >&2 printf ' %q' "$@"; >&2 printf '\n'; "$@"; }
go() { "go-$@"; }
next() { "${FUNCNAME[1]:?}-$@"; }
#---

--scribe-vars() {
    SCRIBE_ROOT_DIR=${scribe_root_dir:?}
    export SCRIBE_ROOT_DIR

    LLAMA_API_KEY=${llama_api_key:?}
    export LLAMA_API_KEY

    NOMIC_API_KEY=${nomic_api_key:?}
    export NOMIC_API_KEY

    VAINL_API_URL=${vainl_api_url:?}
    export VAINL_API_URL

    VAINL_API_KEY=${vainl_api_key:?}
    export VAINL_API_KEY
}

declare -A server_host
declare -A server_bind
declare -A server_port

server_host["red"]="https://red.is.mediocreatbest.xyz/"
server_bind["red"]="127.55.84.71"
server_port["red"]=46072

server_host["purple"]="https://purple.is.mediocreatbest.xyz/"
server_bind["purple"]="127.102.167.41"
server_port["purple"]=51399


#---

app=${root:?}/app
app_session_name=${project,,}-app
app_server_host=${server_host["red"]:?}
app_server_bind=${server_bind["red"]:?}
app_server_port=${server_port["red"]:?}

go-app() {
    next "$@"
}

go-app-Initialize-Environment() {
    cd "${app:?}" \
    && \
    pexec npm install \
    ##
}

go-app-Invoke-Server() {
    cd "${app:?}" \
    && \
    pexec node_modules/.bin/vite \
        --host "${app_server_bind:?}" \
        --port "${app_server_port:?}" \
        --strictPort \
    ##
}

go-app-Start-Server() {
    pexec tmux new-session \
        -A \
        -s "${app_session_name:?}" \
    "${self:?}" app Invoke-Server \
    ##
}

#---

www=${root:?}/www
www_virtualenv_path=${www:?}/venv

go-www() {
    next "$@"
}

go-www-New-Virtualenv() {
    pexec python3 -m venv \
        "${www_virtualenv_path:?}" \
    ##
}

go-www-Initialize-Virtualenv() {
    pexec "${www_virtualenv_path:?}/bin/pip" install \
        -r "${www:?}/requirements.txt" \
    ##
}

go-www-Invoke-Migrations() {
    pexec "${www_virtualenv_path:?}/bin/python" \
    "${www:?}/manage.py" migrate \
        "$@" \
    ##
}

go-www-Invoke-Server() {
    pexec "${www_virtualenv_path:?}/bin/python" \
    "${www:?}/manage.py" runserver \
        "$@" \
    ##
}

#---

web=${root:?}/web
web_virtualenv_path=${web:?}/venv
web_session_name=${project,,}-web
web_server_host=${server_host["purple"]:?}
web_server_bind=${server_bind["purple"]:?}
web_server_port=${server_port["purple"]:?}

go-web() {
    next "$@"
}

go-web-New-Virtualenv() {
    pexec python3 -m venv \
        "${web_virtualenv_path:?}" \
    ##
}

go-web-Initialize-Environment() {
    pexec "${web_virtualenv_path:?}/bin/pip" install \
        -r "${web:?}/requirements.txt" \
        -e "${web:?}" \
    ##
}

go-web-Invoke-Server() {
    --scribe-vars

    pexec "${web_virtualenv_path:?}/bin/uvicorn" \
        --host "${web_server_bind:?}" \
        --port "${web_server_port:?}" \
        web.server:app \
    ##
}

go-web-Start-Server() {
    pexec tmux new-session \
        -A \
        -s "${web_session_name:?}" \
    "${self:?}" web Invoke-Server \
    ##
}

#---
test -f "${root:?}/env.sh" && source "${_:?}"
go "$@"
