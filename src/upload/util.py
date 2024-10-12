"""

"""

from __future__ import annotations
from ._auto import auto


def checksum(
    inp: auto.typing.IO[bytes],
) -> str:
    h = auto.hashlib.sha256()
    while (chunk := inp.read(4096)):
        h.update(chunk)

    return h.hexdigest()


def encrypt(
    *,
    inp: auto.typing.IO[bytes],
    out: auto.typing.IO[bytes],
    key_name: str,
):
    def Writer(f, g):
        auto.shutil.copyfileobj(f, g)
        g.close()

    def Reader(f, g):
        auto.shutil.copyfileobj(f, g)
        f.close()

    args = [
        'openssl',
        'enc',
        '-pbkdf2',
        '-aes-256-cbc',
        '-pass', f'env:{key_name}',
        '-in', '/dev/stdin',
        '-out', '/dev/stdout',
    ]

    with auto.contextlib.ExitStack() as stack:
        p = stack.enter_context( auto.subprocess.Popen(
            args,
            stdin=auto.subprocess.PIPE,
            stdout=auto.subprocess.PIPE,
        ) )

        writer = auto.threading.Thread(
            target=Writer,
            args=(inp, p.stdin),
        )
        stack.callback( writer.join )
        writer.start()

        reader = auto.threading.Thread(
            target=Reader,
            args=(p.stdout, out),
        )
        stack.callback( reader.join )
        reader.start()

        p.wait()


def decrypt(
    *,
    inp: auto.typing.IO[bytes],
    out: auto.typing.IO[bytes],
    key_name: str,
):
    def Writer(f, g):
        auto.shutil.copyfileobj(f, g)
        g.close()

    def Reader(f, g):
        auto.shutil.copyfileobj(f, g)
        f.close()

    args = [
        'openssl',
        '-d',
        '-aes-256-cbc',
        '-pbkdf2',
        '-pass', f'env:{key_name}',
        '-in', '/dev/stdin',
        '-out', '/dev/stdout',
    ]

    with auto.contextlib.ExitStack() as stack:
        p = stack.enter_context( auto.subprocess.Popen(
            args,
            stdin=auto.subprocess.PIPE,
            stdout=auto.subprocess.PIPE,
        ) )

        writer = auto.threading.Thread(
            target=Writer,
            args=(inp, p.stdin),
        )
        stack.callback( writer.join )
        writer.start()

        reader = auto.threading.Thread(
            target=Reader,
            args=(p.stdout, out),
        )
        stack.callback( reader.join )
        reader.start()

        p.wait()


def SQLQuery(s: str, /, **kwargs):
    environment = auto.jinja2.Environment()
    environment.filters['tosqlref'] = lambda x: '"' + str(x).replace('"', '""') + '"'
    environment.filters['tosqlstr'] = lambda x: "'" + str(x).replace("'", "''") + "'"
    environment.filters['tosqlint'] = lambda x: str(int(str(x)))
    environment.globals['auto'] = auto

    template = environment.from_string(s)

    return template.render(**kwargs)


def upload(
    *,
    client: auto.boto3.client,
    file,
    bucket: str,
    name: str,
):
    try:
        client.upload_fileobj(
            file,
            bucket,
            name,
        )

    except auto.botocore.exceptions.ClientError as e:
        raise e
