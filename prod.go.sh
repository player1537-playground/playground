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

docker_image_tag=${project,,}--prod:latest
docker_container_name=${project,,}--prod
docker_source_dir=${root:?}

read -r -d '' dockerfile <<'__DOCKERFILE__'
FROM ubuntu:24.04 AS dev

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
)

apt-get install -y \
    "${packages[@]}" \
##

__INSTALL_DEPENDENCIES__

FROM dev AS web

COPY web/requirements.txt /tmp/requirements.txt
RUN /bin/bash <<'__INSTALL_REQUIREMENTS__'
set -euo pipefail on

python3.12 -m venv \
    /venv \
##

/venv/bin/pip install \
    --upgrade pip \
##

/venv/bin/pip install \
    --requirement /tmp/requirements.txt \
##

__INSTALL_REQUIREMENTS__

WORKDIR /opt/web
COPY web /opt/web
RUN /bin/bash <<'__INSTALL_WEB__'
set -euo pipefail on

/venv/bin/pip install \
    /opt/web \
##

__INSTALL_WEB__


CMD ["/venv/bin/uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]

__DOCKERFILE__

go-Build-Image() {
    <<<"${dockerfile:?}" \
    pexec docker build \
        --progress=plain \
        --tag="${docker_image_tag:?}" \
        --target=web \
        --file=- \
        "${docker_source_dir:?}" \
    ##
}

server_bind=127.102.167.41  # ,address https://purple.is.mediocreatbest.xyz/
server_port=51399  # ,address https://purple.is.mediocreatbest.xyz/

go-Invoke-Server() {
    local tty
    if [ -t 0 ]; then
        tty=
    fi

    SCRIBE_ROOT_DIR=${scribe_root_dir:?} \
    LLAMA_API_KEY=${llama_api_key:?} \
    NOMIC_API_KEY=${nomic_api_key:?} \
    VAINL_API_URL=${vainl_api_url:?} \
    VAINL_API_KEY=${vainl_api_key:?} \
    pexec docker run \
        ${tty+--tty} \
        ${tty+--interactive} \
        --detach-keys="ctrl-q,ctrl-q" \
        --user="$(id -u):$(id -g)" \
        --env=USER \
        --env=HOSTNAME \
        --publish="${server_bind:?}:${server_port:?}:8000/tcp" \
        --mount="type=bind,src=/etc/passwd,dst=/etc/passwd,readonly=true" \
        --mount="type=bind,src=/etc/group,dst=/etc/group,readonly=true" \
        --mount="type=bind,src=/mnt/seenas2/data,dst=/mnt/seenas2/data,readonly=true" \
        --mount="type=bind,src=/mnt/seenas2/data${root:?},dst=/mnt/seenas2/data${root:?},readonly=false" \
        --mount="type=bind,src=/mnt/data,dst=/mnt/data,readonly=true" \
        --mount="type=bind,src=${root:?}/data-prod,dst=/opt/data,readonly=false" \
        --workdir="${PWD:?}" \
        --env=SCRIBE_ROOT_DIR \
        --env=LLAMA_API_KEY \
        --env=NOMIC_API_KEY \
        --env=VAINL_API_URL \
        --env=VAINL_API_KEY \
        "${docker_container_name:?}" \
    ##
}

#---
test -f "${root:?}/prod.env.sh" && source "${_:?}"
go "$@"
