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

docker_source_dir=${root:?}
read -r -d '' dockerfile <<'__DOCKERFILE__'
FROM ubuntu:24.04 AS base

RUN /bin/bash <<'__INSTALL_DEPENDENCIES__'
set -euo pipefail on

export DEBIAN_FRONTEND=noninteractive

apt-get update \
##

apt-get upgrade \
    --yes \
##

packages=(
    python3.12
    python3.12-venv
    python3.12-dev
    python3-pip
    libgdbm6
    python3.12-gdbm
)

apt-get install -y \
"${packages[@]}" \
##

__INSTALL_DEPENDENCIES__

__DOCKERFILE__

docker_image_tag=${project,,}:latest
docker_container_name=${project,,}

go-Build-Image() {
    <<<"${dockerfile:?}" \
    pexec docker build \
        --progress=plain \
        --tag="${docker_image_tag:?}" \
        --target=base \
        --file=- \
        "${docker_source_dir:?}" \
    ##
}

go-Start-Container() {
    pexec docker run \
        --rm \
        --init \
        --detach \
        --ulimit=core=0 \
        --cap-add=SYS_PTRACE \
        --net=host \
        --name="${docker_container_name:?}" \
        --mount="type=bind,src=${root:?},dst=${root:?},readonly=false" \
        --mount="type=bind,src=${HOME:?},dst=${HOME:?},readonly=false" \
        --mount="type=bind,src=/etc/passwd,dst=/etc/passwd,readonly=true" \
        --mount="type=bind,src=/etc/group,dst=/etc/group,readonly=true" \
        --mount="type=bind,src=/mnt/seenas2/data,dst=/mnt/seenas2/data,readonly=false" \
        --mount="type=bind,src=/mnt/data,dst=/mnt/data,readonly=false" \
        "${docker_image_tag:?}" \
        sleep infinity \
    ##
}

go-Stop-Container() {
    pexec docker stop \
        --time=0 \
        "${docker_container_name:?}" \
    ##
}

go-Invoke-Container() {
    local tty
    if [ -t 0 ]; then
        tty=
    fi

    pexec docker exec \
        ${tty+--tty} \
        --interactive \
        --detach-keys="ctrl-q,ctrl-q" \
        --user="$(id -u):$(id -g)" \
        --env=USER \
        --env=HOSTNAME \
        --workdir="${PWD:?}" \
        "${docker_container_name:?}" \
        "${@:?Invoke-Container: missing command}" \
    ##
}

--scribe-web-vars() {
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

--scribe-app-vars() {
    VITE_SCRIBE_API_URL=${scribe_api_url:?}
    export VITE_SCRIBE_API_URL
}

declare -A server_host
declare -A server_bind
declare -A server_port

server_host["red"]=https://red.is.mediocreatbest.xyz/
server_bind["red"]=127.55.84.71
server_port["red"]=46072

server_host["purple"]=https://purple.is.mediocreatbest.xyz/
server_bind["purple"]=127.102.167.41
server_port["purple"]=51399


#---

app=${root:?}/app
app_session_name=${project,,}-app
app_server_host=${server_host["red"]:?}
app_server_bind=${server_bind["red"]:?}
app_server_port=${server_port["red"]:?}

go-Initialize-AppEnvironment() {
    --scribe-app-vars

    cd "${app:?}" \
    && \
    pexec npm install \
    ##
}

go-Invoke-AppServer() {
    --scribe-app-vars

    cd "${app:?}" \
    && \
    pexec node_modules/.bin/vite \
        --host "${app_server_bind:?}" \
        --port "${app_server_port:?}" \
        --strictPort \
    ##
}

go-Start-AppServer() {
    --scribe-app-vars

    pexec tmux new-session \
        -A \
        -s "${app_session_name:?}" \
    "${self:?}" app Invoke-AppServer \
    ##
}

go-Build-AppDist() {
    --scribe-app-vars

    cd "${app:?}" \
    && \
    pexec npm run build \
    ##
}

go-Package-AppDist() {
    --scribe-app-vars

    cd "${app:?}" \
    && \
    rm -f dist.zip \
    && \
    pexec zip \
        -r \
        dist.zip \
        dist \
    ##
}

#---

web=${root:?}/web
web_virtualenv_path=${web:?}/venv
web_session_name=${project,,}-web
web_server_host=${server_host["purple"]:?}
web_server_bind=${server_bind["purple"]:?}
web_server_port=${server_port["purple"]:?}

go-New-WebEnvironment() {
    pexec "${self:?}" Invoke-Container "${self:?}" --New-WebEnvironment "$@"
}

go---New-WebEnvironment() {
    pexec python3 -m venv \
        "${web_virtualenv_path:?}" \
    ##
}

go-Initialize-WebEnvironment() {
    pexec "${self:?}" Invoke-Container "${self:?}" --Initialize-WebEnvironment "$@"
}

go---Initialize-WebEnvironment() {
    pexec "${web_virtualenv_path:?}/bin/pip" install \
        -r "${web:?}/requirements.txt" \
        -e "${web:?}" \
    ##
}

go-Invoke-WebServer() {
    pexec "${self:?}" Invoke-Container "${self:?}" --Invoke-WebServer "$@"
}

go---Invoke-WebServer() {
    --scribe-web-vars

    pexec "${web_virtualenv_path:?}/bin/uvicorn" \
        --host "${web_server_bind:?}" \
        --port "${web_server_port:?}" \
        web.server:app \
    ##
}

go-Start-WebServer() {
    pexec tmux new-session \
        -A \
        -s "${web_session_name:?}" \
    "${self:?}" Invoke-WebServer \
    ##
}

#---
test -f "${root:?}/env.sh" && source "${_:?}"
if ! (return 0 2>/dev/null); then
    go "$@"
fi
