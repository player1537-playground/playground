"""

"""

from __future__ import annotations
from ._auto import auto
from ._config import config
from . import util


def _S3Client() -> auto.boto3.client:
    client = auto.boto3.client(
        's3',
        aws_access_key_id=config.aws.access_key_id,
        aws_secret_access_key=config.aws.secret_access_key,
    )

    return client


@auto.functools.cache
def S3Client() -> auto.boto3.client:
    client = _S3Client()

    return client


def _Database(
    *,
    path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    name: str = 'Database.sqlite3',
) -> auto.pathlib.Path:
    if path is ...:
        if root is ...:
            root = config.datadir
        path = root / name

    with auto.contextlib.closing( auto.sqlite3.connect(
        path,
        check_same_thread=False,
    ) ) as database:
        sqlquery = util.SQLQuery(r'''
            CREATE TABLE IF NOT EXISTS upload (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                dechash TEXT NOT NULL,
                enchash TEXT NOT NULL
            );
        ''')

        database.execute(sqlquery)
        database.commit()

    return path


@auto.functools.cache
def Database() -> auto.sqlite3.Connection:
    path = _Database()

    database = auto.sqlite3.connect(
        path,
        check_same_thread=False,
    )

    return database


def auth(request: auto.fastapi.Request) -> bool:
    auth = request.headers.get('Authorization', None)
    if auth is None:
        return False

    basic, auth = auth.split(' ', 1)
    if basic != 'Basic':
        return False

    username, password = auto.base64.b64decode(auth).decode().split(':', 1)

    # XXX(th): Vulnerable to a timing attack.
    if username != config.login.username:
        return False

    # XXX(th): Vulnerable to a timing attack.
    if password != config.login.password:
        return False

    return True


app = auto.fastapi.FastAPI()

templates = auto.fastapi.templating.Jinja2Templates(
    directory=config.codedir / 'templates',
)


@app.get('/upload/')
def upload(
    *,
    request: auto.fastapi.Request,
    auth: auto.typing.Annotated[
        bool,
        auto.fastapi.Depends(auth),
    ],
):
    if not auth:
        return auto.fastapi.Response(
            status_code=401,
            headers={'WWW-Authenticate': 'Basic'},
        )

    return templates.TemplateResponse(
        'upload.html',
        dict(
            request=request,
        ),
    )


@app.post('/upload/')
async def upload_post(
    *,
    request: auto.fastapi.Request,
    uploads: auto.typing.Annotated[
        list[bytes],
        auto.fastapi.File(),
    ],
    auth: auto.typing.Annotated[
        bool,
        auto.fastapi.Depends(auth),
    ],
    database: auto.typing.Annotated[
        auto.sqlite3.Connection,
        auto.fastapi.Depends(Database),
    ],
    client: auto.typing.Annotated[
        auto.boto3.client,
        auto.fastapi.Depends(S3Client),
    ],
):
    if not auth:
        return auto.fastapi.Response(
            status_code=401,
            headers={'WWW-Authenticate': 'Basic'},
        )

    filenames = []
    for upload in (await request.form()).getlist('uploads'):
        filenames.append(upload.filename)

    hashes = []
    for filename, upload in zip(filenames, uploads):
        decfile = auto.io.BytesIO(upload)

        dechash = await auto.asyncio.to_thread(util.checksum, decfile)
        decfile.seek(0)

        encfile = auto.io.BytesIO()
        await auto.asyncio.to_thread(
            util.encrypt,
            inp=decfile,
            out=encfile,
            key_name=config.encryption_key_name,
        )

        enchash = await auto.asyncio.to_thread(util.checksum, encfile)
        encfile.seek(0)
        
        util.upload(
            client = client,
            file = encfile,
            bucket = 'visualiz-upload',
            name = f'{dechash[:2]}/{dechash[2:]}'
        )

        hashes.append((dechash, enchash))

        sqlquery = util.SQLQuery(r'''
            INSERT INTO upload (filename, dechash, enchash)
            VALUES (?, ?, ?);
        ''')

        database.execute(sqlquery, (filename, dechash, enchash))
        database.commit()

    return templates.TemplateResponse(
        'upload_post.html',
        dict(
            filenames=filenames,
            hashes=hashes,
            request=request,
        ),
    )
