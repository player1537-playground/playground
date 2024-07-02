"""

"""

from __future__ import annotations
from ._auto import auto
from ._config import config
from . import util
from . import vainl


@auto.functools.cache
def LLM(
    arg: str | None = None,
    /,
    *,
    cache: auto.typing.Literal[...] | None = ...,
) -> util.LLM:
    host, model = arg.split('/', 1)

    api_url = {
        ('devcloud', 'llama'):
            'https://completion.on.devcloud.is.mediocreatbest.xyz/llama/',
        ('sahara', 'llama'):
            'https://completion.on.sahara.is.mediocreatbest.xyz/llama/',
        ('kavir', 'llama'):
            'https://completion.on.kavir.is.mediocreatbest.xyz/llama/',
        ('nebula', 'llama'):
            'https://completion.on.nebula.is.mediocreatbest.xyz/llama/',

        ('sahara', 'tinyllama'):
            'https://completion.on.sahara.is.mediocreatbest.xyz/tinyllama/',
        ('kavir', 'tinyllama'):
            'https://completion.on.kavir.is.mediocreatbest.xyz/tinyllama/',
        ('nebula', 'tinyllama'):
            'https://completion.on.nebula.is.mediocreatbest.xyz/tinyllama/',

        ('sahara', 'nomic'):
            'https://completion.on.sahara.is.mediocreatbest.xyz/nomic/',
        ('kavir', 'nomic'):
            'https://completion.on.kavir.is.mediocreatbest.xyz/nomic/',
        ('nebula', 'nomic'):
            'https://completion.on.nebula.is.mediocreatbest.xyz/nomic/',
    }[host, model]
    
    api_key = config.llama.api_key

    if cache is ...:
        global __d28cb327
        try: __d28cb327
        except NameError: __d28cb327 = {}
        if model not in __d28cb327:
            cache = auto.shelve.open(
                str(config.datadir / f'{model}.cache'),
            )
            __d28cb327[model] = cache
        else:
            cache = __d28cb327[model]

    prompt_kwargs = dict(
        max_tokens=300,
        temperature=0.0,
        frequency_penalty=0,
        presence_penalty=0,
        cache_prompt=True,
    ) | {
        'llama': dict(
            stop=[
                '<|eot_id|>',
            ],
        ),
        'tinyllama': dict(
            stop=[
                # '</s>',
                '<|endoftext|>',
            ],
        ),
    }.get(model, {})

    llm = util.LLM(
        model=model,
        api_url=api_url,
        api_key=api_key,
        cache=cache,
        prompt_kwargs=prompt_kwargs,
    )

    return llm


@auto.functools.cache
def TruncatedICD10CM():
    root = config.datadir
    path = root / 'Truncated ICD-10-CM.csv'
    
    df = auto.pd.read_csv(
        path,
        index_col='code',
        dtype=str,
        quoting=auto.csv.QUOTE_NONNUMERIC,
    )

    return df


@auto.functools.cache
def ICD10CM():
    def scope(url) -> auto.pathlib.Path:
        root = config.datadir
        parts = auto.urllib.parse.urlparse(url)
        path = auto.pathlib.Path(parts.path)
        path = root / path.name

        if not path.exists():
            with path.open('wb') as f:
                with auto.requests.get(url, stream=True) as r:
                    for chunk in r.iter_content(chunk_size=10240):
                        f.write(chunk)
        assert path.exists()
        assert path.stat().st_size > 0

        return path

    table_and_index = \
        scope('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024-Update/icd10cm-Table%20and%20Index-April-2024.zip')
    descriptions = \
        scope('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024-Update/icd10cm-Codes-Descriptions-April-2024.zip')
    addenda = \
        scope('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024-Update/icd10cm-addenda-files-April-2024.zip')

    # for path in [table_and_index, descriptions, addenda]:
    #     print(path.name)
    #     with auto.zipfile.ZipFile(path, 'r') as arc:
    #         for info in arc.infolist():
    #             print(info)

    root = auto.zipfile.Path(descriptions)
    path = root / 'icd10cm-codes-April-2024.txt'

    # with auto.mediocreatbest.Textarea():
    #     with path.open('r') as f:
    #         # lines = auto.more_itertools.take(100, f)
    #         lines = auto.collections.deque(f, maxlen=100)
    #         for line in lines:
    #             print(line, end='')

    #=> "Z949    Transplanted organ and tissue status, unspecified"
    #=> "Z950    Presence of cardiac pacemaker"
    #=> "Z951    Presence of aortocoronary bypass graft"
    #=> "Z952    Presence of prosthetic heart valve"
    #=> "Z953    Presence of xenogenic heart valve"
    #=> "Z954    Presence of other heart-valve replacement"
    #=> "Z955    Presence of coronary angioplasty implant and graft"
    #=> "Z95810  Presence of automatic (implantable) cardiac defibrillator"
    #=> "Z95811  Presence of heart assist device"
    #=> "Z95812  Presence of fully implantable artificial heart"
    #=> "Z95818  Presence of other cardiac implants and grafts"

    df = []
    with path.open('r') as f:
        for line in f:
            code, desc = auto.re.split(r'\s+', line, maxsplit=1)
            code = code.strip()
            desc = desc.strip()
            df.append((code, desc))

    df = auto.pandas.DataFrame(
        df,
        columns=['code', 'desc'],
    )
    df = df.set_index('code')

    return df


@auto.functools.cache
def CompleteICD10CM(
) -> auto.pd.DataFrame:
    icd10cm = ICD10CM()
    trunc = TruncatedICD10CM()

    df = auto.pd.concat([
        icd10cm,
        trunc,
    ])

    # df = df.query('code.str.len() <= 4')

    return df


@auto.contextlib.asynccontextmanager
async def lifespan(app: auto.fastapi.FastAPI):
    yield


app = auto.fastapi.FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    auto.fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(auto.pydantic.BaseModel):
    document: str


class AnalyzeResponseItemChunk(auto.pydantic.BaseModel):
    offset: int
    length: int
    text: str


class AnalyzeResponseItem(auto.pydantic.BaseModel):
    distance: float
    code: str
    desc: str
    chunk: AnalyzeResponseItemChunk

class AnalyzeResponse(auto.pydantic.BaseModel):
    best: list[AnalyzeResponseItem]


@app.post("/analyze")
async def analyze(
    *,
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    llm = LLM('sahara/nomic')
    icd10cm = CompleteICD10CM()

    def Passage(*, code: str) -> str:
        desc = icd10cm.loc[code, 'desc']
        return (
            f"search_document: "
            # f"The ICD-10 code {code} is {desc}."
            f"The diagnosis is {desc}"
        )

    def Query(*, text: str) -> str:
        return (
            f"search_query: "
            # f"The ICD-10 code [MASK] is {text}."
            # f"What ICD-10 code corresponds to this passage: "
            # f"{text}"
            f"What diagnosis corresponds to this passage: "
            f"{text}"
        )

    passages = []
    for code in icd10cm.index:
        passage = Passage(code=code)
        passages.append(passage)

    pembeds = llm.embed([
        *passages,
    ], verbose=True)
    
    best = []
    auto.heapq.heapify(best)
    
    def emit(*, distance: float, code: str, chunk: util.Chunk):
        item = (distance, code, chunk)
        if len(best) < 100:
            auto.heapq.heappush(best, item)
        else:
            auto.heapq.heappushpop(best, item)
    
    for n, k in [
        (60, 9),
        (100, 7),
        (140, 5),
    ]:
        chunks = util.Chunks(request.document, avg_size=n//k)
        chunks = util.Overlap(k, chunks=chunks)

        it = chunks
        it = auto.tqdm.auto.tqdm(it, total=len(chunks))

        penalty = None
        for i, chunk in enumerate(it):
            query = Query(text=chunk.text)
            qembed = llm.embed(
                query,
            )

            cdist = auto.scipy.spatial.distance.cdist(
                pembeds,
                [qembed],
                metric='cosine',
            )

            if penalty is not None:
                # query = Passage(code=penalty)
                query = Query(text=penalty)
                qembed = llm.embed(
                    query,
                )

                cdist -= auto.scipy.spatial.distance.cdist(
                    pembeds,
                    [qembed],
                    metric='cosine',
                )

            assert cdist.shape == (len(pembeds), 1)
            cdist = cdist[:, 0]

            dist = cdist.min()
            code = icd10cm.index[cdist.argmin()]
            
            emit(
                distance=dist,
                code=code,
                chunk=chunk,
            )
    
    best = sorted(best)
    
    return AnalyzeResponse(
        best=[
            AnalyzeResponseItem(
                distance=distance,
                code=code,
                desc=icd10cm.loc[code, 'desc'],
                chunk=AnalyzeResponseItemChunk(
                    offset=chunk.offset,
                    length=chunk.length,
                    text=chunk.text,
                ),
            )
            for distance, code, chunk in best
        ],
    )


class FingleProperty(auto.pydantic.BaseModel):
    identity: str
    multiply: float = 1.0


class FingleMagazine(auto.pydantic.BaseModel):
    identity: str
    properties: list[FingleProperty]
    multiply: float = 1.0


class FingleWhatever(auto.pydantic.BaseModel):
    identity: str
    magazines: list[FingleMagazine]


class FingleRequest(auto.pydantic.BaseModel):
    whatever: FingleWhatever


@app.post("/fingle/")
async def fingle(
    *,
    request: FingleRequest,
):
    whatever = request.whatever
    
    ideation = vainl.Ideation('By Census Tract')
    material = vainl.Material(
        properties=[
            'geometry',
        ] + sorted(set([
            property.identity
            for magazine in whatever.magazines
            for property in magazine.properties
        ])),
        ideation=ideation,
    )
    
    df = material
    df = df[df.index.str.startswith('47')]
    df = df[auto.pd.notna(df['geometry'])]
    
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    
    df -= mean
    df /= std
    df.fillna(0, inplace=True)
    df.clip(-3, 3, inplace=True)
    
    totalal = auto.pd.Series(0, index=df.index)
    for magazine in whatever.magazines:
        total = auto.pd.Series(0, index=df.index)
        for property in magazine.properties:
            total += df[property.identity] * property.multiply
        total.clip(-1, 1, inplace=True)
        totalal += total * magazine.multiply
    totalal.clip(-1, 1, inplace=True)
    
    assert isinstance(df, auto.geopandas.GeoDataFrame), \
        f'Expected a GeoDataFrame, got {type(df)}'
    ax = df.plot(
        totalal,
        legend=True,
        cmap='coolwarm',
    )
    fig = ax.get_figure()
    
    fig.tight_layout()
    fig.savefig((io := auto.io.BytesIO()), format='png')
    auto.plt.close(fig)
    
    return auto.responses.FileResponse(
        io,
        media_type='image/png',
    )


@app.get("/")
async def index():
    return {"Hello": "World"}
