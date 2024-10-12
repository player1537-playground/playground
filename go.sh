#!/usr/bin/env bash
# vim :set ts=4 sw=4 sts=4 et:
die() { printf $'Error: %s\n' "$*" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
self=${BASH_SOURCE[0]:?}
project=${root##*/}
pexec() { >&2 printf exec; >&2 printf ' %q' "$@"; >&2 printf '\n'; exec "$@"; }
prun() { >&2 printf run; >&2 printf ' %q' "$@"; >&2 printf '\n'; "$@"; }
go() { "go-$@"; }
next() { "${FUNCNAME[1]:?}-$@"; }
#---

environment=${root:?}/venv

go-New-Environment() {
    pexec python3 -m venv \
        "${environment:?}" \
    ##
}

go-Initialize-Environment() {
    packages=(
        "fastapi"
        "uvicorn"
        "jinja2"
        "python-multipart"
    )

    pexec "${environment:?}/bin/pip" install \
        "${packages[@]}" \
    ##
}

--environment() {
    UPLOAD_ROOT_DIR=${upload_root_dir:?}
    export UPLOAD_ROOT_DIR

    UPLOAD_ENCRYPTION_KEY=${upload_encryption_key:?}
    export UPLOAD_ENCRYPTION_KEY

    UPLOAD_LOGIN_USERNAME=${upload_login_username:?}
    export UPLOAD_LOGIN_USERNAME

    UPLOAD_LOGIN_PASSWORD=${upload_login_password:?}
    export UPLOAD_LOGIN_PASSWORD
}

server_host="https://purple.is.mediocreatbest.xyz/upload/"
server_bind=127.128.78.216
server_port=33580

go-Invoke-Server() {
    --environment

    PYTHONPATH="${root:?}/src" \
    pexec "${environment:?}/bin/uvicorn" \
        "upload.server:app" \
        --host "${server_bind:?}" \
        --port "${server_port:?}" \
    ##
}

#---
test -f "${root:?}/env.sh" && source "${_:?}"
"go-$@"
