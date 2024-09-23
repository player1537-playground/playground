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
def ICD10CM() -> auto.pd.DataFrame:
    return util.ICD10CM()


@auto.functools.cache
def ICD10PCS() -> auto.pd.DataFrame:
    return util.ICD10PCS()


@auto.functools.cache
def DX2DX():
    root = config.datadir
    path = root / 'DX2DX.csv'
    assert path.exists(), path

    df = auto.pd.read_csv(
        path,
        index_col=0,
        dtype=auto.collections.defaultdict(lambda: int) | {
            0: str,
        },
        quoting=auto.csv.QUOTE_NONNUMERIC,
    )
    df.index.name = 'dx'

    if 'NoDx' in df.columns:
        df.drop(columns=['NoDx'], inplace=True)
        df.drop(index='NoDx', inplace=True)

    return df


@auto.functools.cache
def DX2PD():
    root = config.datadir
    path = root / 'DX2PD.csv'
    assert path.exists(), path

    df = auto.pd.read_csv(
        path,
        index_col=0,
        dtype=auto.collections.defaultdict(lambda: int) | {
            0: str,
        },
        quoting=auto.csv.QUOTE_NONNUMERIC,
    )
    df.index.name = 'dx'

    if 'NoDx' in df.index:
        df.drop(index='NoDx', inplace=True)
    if 'NoP' in df.columns:
        df.drop(columns=['NoP'], inplace=True)

    return df


@auto.functools.cache
def PD2DX():
    df = DX2PD()
    df = df.T
    return df


@auto.functools.cache
def PD2PD():
    root = config.datadir
    path = root / 'PD2PD.csv'
    assert path.exists(), path

    df = auto.pd.read_csv(
        path,
        index_col=0,
        dtype=auto.collections.defaultdict(lambda: int) | {
            0: str,
        },
        quoting=auto.csv.QUOTE_NONNUMERIC,
    )
    df.index.name = 'pd'

    if 'NoP' in df.index:
        df.drop(index='NoP', inplace=True)
    if 'NoP' in df.columns:
        df.drop(columns=['NoP'], inplace=True)

    return df


@auto.functools.cache
def __Database():
    def scope():
        root = config.datadir
        
        path = root / 'MIMIC-IV-Note.sqlite3'
        assert path.exists(), path
        conn = auto.sqlite3.connect(
            path,
            check_same_thread=False,
        )
        
        path = root / 'MIMIC-IV.sqlite3'
        assert path.exists(), path
        conn.execute(util.SQLQuery(r'''
            ATTACH DATABASE {{ path |tosqlstr }} AS "MIMICIV";
        ''', path=path))
        
        path = root / 'Database.sqlite3'
        assert path.exists(), path
        conn.execute(util.SQLQuery(r'''
            ATTACH DATABASE {{ path |tosqlstr }} AS "MIMICIVAlt";
        ''', path=path))
        
        return conn

    queue = auto.queue.Queue()
    for _ in range(8):
        queue.put_nowait(scope())
    
    @auto.contextlib.contextmanager
    def scope():
        conn = queue.get()
        try:
            yield conn
        finally:
            queue.put_nowait(conn)
    
    return scope


Database = __Database()


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


class icd10cmResponse(auto.pydantic.BaseModel):
    code: str
    desc: str


@app.get('/icd10cm/{code}/')
async def icd10cm(
    code: str,
    *,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> icd10cmResponse:
    try:
        desc = icd10cm.loc[code, 'desc']
    except KeyError:
        desc = None
    
    if desc is None:
        raise auto.fastapi.HTTPException(
            status_code=404,
            detail='Not found',
        )
    
    return icd10cmResponse(
        code=code,
        desc=desc,
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
    icd10cm = ICD10CM()

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
    
    gdf = df
    df = gdf.drop(columns=['geometry'])
    
    try:
        mean = df.mean(axis=0)
    except TypeError:
        for col in df.columns:
            try:
                _ = df[col].mean()
            except TypeError:
                print(col)
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
        total /= (sum(abs(property.multiply) for property in magazine.properties) or 1)
        total.clip(-1, 1, inplace=True)
        totalal += total * magazine.multiply
    totalal /= (sum(abs(magazine.multiply) for magazine in whatever.magazines) or 1)
    totalal.clip(-1, 1, inplace=True)
    
    assert isinstance(gdf, auto.geopandas.GeoDataFrame), \
        f'Expected a GeoDataFrame, got {type(gdf)}'
    ax = gdf.plot(
        totalal,
        legend=True,
        cmap='coolwarm',
        figsize=(16, 9),
    )
    fig = ax.get_figure()
    
    # Draw one census tract with a black outline
    
    gdf_one = gdf.query('where == "47093006800"')
    totalal_one = totalal[gdf_one.index]
    gdf_one.plot(
        totalal_one,
        ax=ax,
        facecolor='none',
        edgecolor='black',
    )
    
    ax.set_xlim(-84.5, -83.5)
    ax.set_ylim(35.75, 36.25)

    fig.tight_layout()
    fig.savefig((io := auto.io.BytesIO()), format='png')
    auto.plt.close(fig)
    
    return auto.fastapi.Response(
        content=io.getvalue(),
        media_type='image/png',
    )


class dx2pdRequest(auto.pydantic.BaseModel):
    dx: list[str]


class dx2pdResponseItem(auto.pydantic.BaseModel):
    pd: str
    desc: str
    score: float

class dx2pdResponse(auto.pydantic.BaseModel):
    best: list[dx2pdResponseItem]


@app.post("/dx2pd/")
async def dx2pd(
    *,
    request: dx2pdRequest,
    dx2pd: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            DX2PD,
        ),
    ],
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
) -> dx2pdResponse:
    df = dx2pd.loc[dx2pd.index.intersection(request.dx)]
    
    # counts to probabilities
    df = df.divide(1e-3 + df.sum(axis=0), axis=1)
    
    # bayesian
    row = df.prod(axis=0)
    
    # normalize probabilities
    row = row.divide(1e-3 + row.sum(), axis=0)
    
    row = row.sort_values(ascending=False)
    row = row.head(10)
    
    response = dx2pdResponse(
        best=[],
    )
    
    for pd, score in row.items():
        desc = icd10pcs.loc[pd, 'desc']
        
        response.best.append(dx2pdResponseItem(
            pd=pd,
            desc=desc,
            score=score,
        ))
    
    return response


class dx2dxRequest(auto.pydantic.BaseModel):
    dx: list[str]


class dx2dxResponseItem(auto.pydantic.BaseModel):
    dx: str
    desc: str
    score: float


class dx2dxResponse(auto.pydantic.BaseModel):
    best: list[dx2dxResponseItem]


@app.post("/dx2dx/")
async def dx2dx(
    *,
    request: dx2dxRequest,
    dx2dx: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            DX2DX,
        ),
    ],
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> dx2dxResponse:
    df = dx2dx.loc[dx2dx.index.intersection(request.dx)]
    
    # counts to probabilities
    df = df.divide(1e-3 + df.sum(axis=0), axis=1)
    
    # bayesian
    row = df.prod(axis=0)
    
    # normalize probabilities
    row = row.divide(1e-3 + row.sum(), axis=0)
    
    row = row.sort_values(ascending=False)
    row = row.head(10)
    
    response = dx2pdResponse(
        best=[],
    )
    
    for dx, score in row.items():
        desc = icd10cm.loc[dx, 'desc']
        
        response.best.append(dx2dxResponseItem(
            dx=dx,
            desc=desc,
            score=score,
        ))
    
    return response


class pd2dxRequest(auto.pydantic.BaseModel):
    pd: list[str]


class pd2dxResponseItem(auto.pydantic.BaseModel):
    dx: str
    desc: str
    score: float


class pd2dxResponse(auto.pydantic.BaseModel):
    best: list[pd2dxResponseItem]


@app.post("/pd2dx/")
async def pd2dx(
    *,
    request: pd2dxRequest,
    pd2dx: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            PD2DX,
        ),
    ],
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> pd2dxResponse:
    df = pd2dx.loc[pd2dx.index.intersection(request.pd)]
    
    # counts to probabilities
    df = df.divide(1e-3 + df.sum(axis=0), axis=1)
    
    # bayesian
    row = df.prod(axis=0)
    
    # normalize probabilities
    row = row.divide(1e-3 + row.sum(), axis=0)
    
    row = row.sort_values(ascending=False)
    row = row.head(10)
    
    response = pd2dxResponse(
        best=[],
    )
    
    for dx, score in row.items():
        desc = icd10cm.loc[dx, 'desc']
        
        response.best.append(pd2dxResponseItem(
            dx=dx,
            desc=desc,
            score=score,
        ))
    
    return response


class pd2pdRequest(auto.pydantic.BaseModel):
    pd: list[str]


class pd2pdResponseItem(auto.pydantic.BaseModel):
    pd: str
    desc: str
    score: float


class pd2pdResponse(auto.pydantic.BaseModel):
    best: list[pd2pdResponseItem]


@app.post("/pd2pd/")
async def pd2pd(
    *,
    request: pd2pdRequest,
    pd2pd: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            PD2PD,
        ),
    ],
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
) -> pd2pdResponse:
    df = pd2pd.loc[pd2pd.index.intersection(request.pd)]
    
    # counts to probabilities
    df = df.divide(1e-3 + df.sum(axis=0), axis=1)
    
    # bayesian
    row = df.prod(axis=0)
    
    # normalize probabilities
    row = row.divide(1e-3 + row.sum(), axis=0)
    
    row = row.sort_values(ascending=False)
    row = row.head(10)
    
    response = pd2pdResponse(
        best=[],
    )
    
    for pd, score in row.items():
        desc = icd10pcs.loc[pd, 'desc']
        
        response.best.append(pd2pdResponseItem(
            pd=pd,
            desc=desc,
            score=score,
        ))
    
    return response


class fooRequest(auto.pydantic.BaseModel):
    dx: str


class fooResponseItem(auto.pydantic.BaseModel):
    dx: str
    desc: str
    score: float


class fooResponse(auto.pydantic.BaseModel):
    best: list[fooResponseItem]


@app.post("/foo/")
async def foo(
    *,
    request: fooRequest,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> fooResponse:
    def Passage(*, code: str) -> str:
        desc = icd10cm.loc[code, 'desc']
        return (
            f"search_document: "
            f"The diagnosis is "
            f"{code}: {desc}"
        )
    
    def Query(*, code: str) -> str:
        desc = icd10cm.loc[code, 'desc']
        return (
            f"search_query: "
            f"What equivalent diagnosis corresponds to this passage: "
            f"{code}: {desc}"
        )
    
    icd10cm = icd10cm.query('code.str.len() <= 4')
    
    llm = LLM('sahara/nomic')
    
    passages = []
    for code in icd10cm.index:
        passage = Passage(code=code)
        passages.append(passage)
    
    pembeds = llm.embed([
        *passages,
    ], verbose=True)
    
    query = Query(code=request.dx)
    
    qembed = llm.embed(
        query,
    )
    
    cdist = auto.scipy.spatial.distance.cdist(
        pembeds,
        [qembed],
        metric='cosine',
    )
    assert cdist.shape == (len(pembeds), 1), \
        f'Expected shape {(len(pembeds), 1)}, got {cdist.shape}'
    cdist = cdist[:, 0]
    
    indices = cdist.argsort()
    indices = indices[:10]
    
    response = fooResponse(
        best=[],
    )
    
    for i in indices:
        code = icd10cm.index[i]
        desc = icd10cm.loc[code, 'desc']
        
        response.best.append(fooResponseItem(
            dx=code,
            desc=desc,
            score=cdist[i],
        ))
    
    return response


class barRequest(auto.pydantic.BaseModel):
    pd: str


class barResponseItem(auto.pydantic.BaseModel):
    pd: str
    desc: str
    score: float


class barResponse(auto.pydantic.BaseModel):
    best: list[barResponseItem]


@app.post("/bar/")
async def bar(
    *,
    request: barRequest,
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
) -> barResponse:
    def Passage(*, code: str) -> str:
        desc = icd10pcs.loc[code, 'desc']
        return (
            f"search_document: "
            f"The procedure is "
            f"{code}: {desc}"
        )
    
    def Query(*, code: str) -> str:
        desc = icd10pcs.loc[code, 'desc']
        return (
            f"search_query: "
            f"What equivalent procedure corresponds to this passage: "
            f"{code}: {desc}"
        )
    
    icd10pcs = icd10pcs.query('PD.str.len() <= 3')
    
    llm = LLM('sahara/nomic')
    
    passages = []
    for code in icd10pcs.index:
        passage = Passage(code=code)
        passages.append(passage)
    
    pembeds = llm.embed([
        *passages,
    ], verbose=True)
    
    query = Query(code=request.pd)
    
    qembed = llm.embed(
        query,
    )
    
    cdist = auto.scipy.spatial.distance.cdist(
        pembeds,
        [qembed],
        metric='cosine',
    )
    assert cdist.shape == (len(pembeds), 1), \
        f'Expected shape {(len(pembeds), 1)}, got {cdist.shape}'
    cdist = cdist[:, 0]
    
    indices = cdist.argsort()
    indices = indices[:10]
    
    response = barResponse(
        best=[],
    )
    
    for i in indices:
        code = icd10pcs.index[i]
        desc = icd10pcs.loc[code, 'desc']
        
        response.best.append(barResponseItem(
            pd=code,
            desc=desc,
            score=cdist[i],
        ))
    
    return response


class bingRequest(auto.pydantic.BaseModel):
    search: str


class bingResponseItem(auto.pydantic.BaseModel):
    dx: str
    desc: str
    score: float


class bingResponse(auto.pydantic.BaseModel):
    best: list[bingResponseItem]


@app.post("/bing/")
async def bing(
    *,
    request: bingRequest,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> bingResponse:
    def Passage(*, code: str) -> str:
        desc = icd10cm.loc[code, 'desc']
        return (
            f"search_document: "
            f"The diagnosis is "
            f"{code}: {desc}"
        )
    
    def Query(*, text: str) -> str:
        return (
            f"search_query: "
            f"What equivalent diagnosis corresponds to this passage: "
            f"{text}"
        )
    
    icd10cm = icd10cm.query('code.str.len() <= 4')
    
    llm = LLM('sahara/nomic')
    
    passages = []
    for code in icd10cm.index:
        passage = Passage(code=code)
        passages.append(passage)
    
    pembeds = llm.embed([
        *passages,
    ], verbose=True)
    
    query = Query(text=request.search)
    
    qembed = llm.embed(
        query,
    )
    
    cdist = auto.scipy.spatial.distance.cdist(
        pembeds,
        [qembed],
        metric='cosine',
    )
    assert cdist.shape == (len(pembeds), 1), \
        f'Expected shape {(len(pembeds), 1)}, got {cdist.shape}'
    cdist = cdist[:, 0]
    
    indices = cdist.argsort()
    indices = indices[:10]
    
    response = bingResponse(
        best=[],
    )
    
    for i in indices:
        code = icd10cm.index[i]
        desc = icd10cm.loc[code, 'desc']
        
        response.best.append(bingResponseItem(
            dx=code,
            desc=desc,
            score=cdist[i],
        ))
    
    return response


class pingRequest(auto.pydantic.BaseModel):
    search: str


class pingResponseItem(auto.pydantic.BaseModel):
    pd: str
    desc: str
    score: float


class pingResponse(auto.pydantic.BaseModel):
    best: list[pingResponseItem]


@app.post("/ping/")
async def ping(
    *,
    request: pingRequest,
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
) -> pingResponse:
    def Passage(*, code: str) -> str:
        desc = icd10pcs.loc[code, 'desc']
        return (
            f"search_document: "
            f"The procedure is "
            f"{code}: {desc}"
        )
    
    def Query(*, text: str) -> str:
        return (
            f"search_query: "
            f"What equivalent procedure corresponds to this passage: "
            f"{text}"
        )
    
    icd10pcs = icd10pcs.query('PD.str.len() <= 3')
    
    llm = LLM('sahara/nomic')
    
    passages = []
    for code in icd10pcs.index:
        passage = Passage(code=code)
        passages.append(passage)
    
    pembeds = llm.embed([
        *passages,
    ], verbose=True)
    
    query = Query(text=request.search)
    
    qembed = llm.embed(
        query,
    )
    
    cdist = auto.scipy.spatial.distance.cdist(
        pembeds,
        [qembed],
        metric='cosine',
    )
    assert cdist.shape == (len(pembeds), 1), \
        f'Expected shape {(len(pembeds), 1)}, got {cdist.shape}'
    cdist = cdist[:, 0]
    
    indices = cdist.argsort()
    indices = indices[:10]
    
    response = pingResponse(
        best=[],
    )
    
    for i in indices:
        code = icd10pcs.index[i]
        desc = icd10pcs.loc[code, 'desc']
        
        response.best.append(pingResponseItem(
            pd=code,
            desc=desc,
            score=cdist[i],
        ))
    
    return response

class rewardRequest(auto.pydantic.BaseModel):
    what: auto.typing.Literal['Positive', 'Negative', 'Positive-Negative']
    dx: list[str]
    pd: list[str]


class rewardResponse(auto.pydantic.BaseModel):
    score: float
    dxs: list[str]
    pds: list[str]


@app.post("/reward/")
async def reward(
    *,
    request: rewardRequest,
) -> rewardResponse:
    global __416b2eb7
    try: __416b2eb7
    except NameError:
        def scope():
            root = config.datadir
            path = root / '__416b2eb7.json'
            if not path.exists(): return ([], [])
            requests, responses = [], []
            with path.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                
                    json = auto.json.loads(line)
                    request = rewardRequest.parse_obj(json['request'])
                    response = rewardResponse.parse_obj(json['response'])
                    
                    requests.append(request)
                    responses.append(response)
            return (requests, responses)
        __416b2eb7 = scope()
    for req, res in zip(*__416b2eb7):
        if req == request:
            return res
    
    if request.what == 'Positive-Negative':
        positive = util.Score(
            what='Positive',
            dxs=request.dx,
            pds=request.pd,
        )
        negative, dxs, pds = util.Score(
            what='Negative',
            dxs=request.dx,
            pds=request.pd,
            top=1,
        )
        score = positive - negative

    else:
        score, dxs, pds = util.Score(
            what=request.what,
            dxs=request.dx,
            pds=request.pd,
            top=1,
        )
    
    response = rewardResponse(
        score=score,
        dxs=dxs,
        pds=pds,
    )
    
    def scope():
        root = config.datadir
        path = root / '__416b2eb7.json'
        with path.open('a') as f:
            auto.json.dump({
                'request': request.dict(),
                'response': response.dict(),
            }, f)
            f.write('\n')
        return (__416b2eb7[0] + [request], __416b2eb7[1] + [response])
    __416b2eb7 = scope()

    return response


class bopRequest(auto.pydantic.BaseModel):
    what: auto.typing.Literal['Positive', 'Negative']
    dx: list[str]
    pd: list[str]
    n: int
    prefix: str


class bopResponse(auto.pydantic.BaseModel):
    baseline: float
    recommendations: list[bopResponseItem]


class bopResponseItem(auto.pydantic.BaseModel):
    score: float
    dx: str
    desc: str
    disp: str


@app.post("/bop/")
async def bop(
    *,
    request: bopRequest,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
) -> bopResponse:
    global __e52074cc
    try: __e52074cc
    except NameError:
        def scope():
            root = config.datadir
            path = root / '__e52074cc.json'
            if not path.exists(): return (None, None)
            with path.open('r') as f:
                json = auto.json.load(f)
            request = bopRequest.parse_obj(json['request'])
            response = bopResponse.parse_obj(json['response'])
            return (request, response)
        __e52074cc = scope()
    if request == __e52074cc[0]:
        return __e52074cc[1]
    
    baseline = util.Score(
        what=request.what,
        dxs=request.dx,
        pds=request.pd,
    )

    it = set(
        dx
        for dx in icd10cm.index
        if len(dx) == request.n
        if dx.startswith(request.prefix)
    )
    it -= set(
        dx[:n]
        for dx in request.dx
        for n in range(1, 1+len(dx))
    )
    it = sorted(it)
    
    with auto.tqdm.auto.tqdm() as pbar:
        scores = auto.pd.Series(
            0,
            index=it,
            dtype=float,
        )
        pbar.reset(total=len(it))
        for dx in it:
            pbar.update()

            score = util.Score(
                dxs=request.dx + [dx],
                pds=request.pd,
            )
            scores[dx] = score

    scores.sort_values(ascending=False, inplace=True)
    scores = scores.head(100)
    # /display scores

    response = bopResponse(
        baseline=baseline,
        recommendations=[],
    )

    for dx, score in scores.items():
        desc = icd10cm.loc[dx, 'desc']
        disp = icd10cm.loc[dx, 'disp']

        response.recommendations.append(bopResponseItem(
            score=score,
            dx=dx,
            desc=desc,
            disp=disp,
        ))

    def scope():
        root = config.datadir
        path = root / '__e52074cc.json'
        with path.open('w') as f:
            auto.json.dump({
                'request': request.dict(),
                'response': response.dict(),
            }, f)
        return (request, response)
    __e52074cc = scope()

    return response


class ringRequest(auto.pydantic.BaseModel):
    what: auto.typing.Literal['Positive', 'Negative', 'Positive-Negative', 'Positive/Negative']
    pred: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4']
    prod: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4']


@app.post("/ring/")
async def ring(
    *,
    request: ringRequest,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
):
    if request.what == 'Positive':
        matrix = util.Matrix(
            what='Positive',
            pred=request.pred,
            prod=request.prod,
        )

    elif request.what == 'Negative':
        matrix = util.Matrix(
            what='Negative',
            pred=request.pred,
            prod=request.prod,
        )

    elif request.what == 'Positive-Negative':
        positive = util.Matrix(
            what='Positive',
            pred=request.pred,
            prod=request.prod,
        )
        negative = util.Matrix(
            what='Negative',
            pred=request.pred,
            prod=request.prod,
        )
        matrix = positive - negative
    
    elif request.what == 'Positive/Negative':
        positive = util.Matrix(
            what='Positive',
            pred=request.pred,
            prod=request.prod,
        )
        negative = util.Matrix(
            what='Negative',
            pred=request.pred,
            prod=request.prod,
        )
        matrix = positive / (1e-9 + negative)

    else:
        raise ValueError(request.what)

    d = matrix.to_dict(
        orient='split',
    )

    return auto.fastapi.Response(
        content=auto.json.dumps(d),
        media_type='application/json',
    )


class dipRequest(auto.pydantic.BaseModel):
    text: str
    limit: int


class dipResponseDiagnosis(auto.pydantic.BaseModel):
    dx10: str


class dipResponseProcedure(auto.pydantic.BaseModel):
    pd10: str


class dipResponseItem(auto.pydantic.BaseModel):
    hadm_id: str
    rank: float
    course: str
    diagnosis: list[dipResponseDiagnosis]
    procedure: list[dipResponseProcedure]


class dipResponseSearch(auto.pydantic.BaseModel):
    items: list[dipResponseItem]


class dipResponse(auto.pydantic.BaseModel):
    search: dipResponseSearch


@app.post("/dip/")
async def dip(
    *,
    request: dipRequest,
) -> dipResponse:
    global __2497fac0
    try: __2497fac0
    except NameError:
        def scope():
            root = config.datadir
            path = root / '__2497fac0.ndjson'
            if not path.exists(): return ([], [])
            requests, responses = [], []
            with path.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                
                    json = auto.json.loads(line)
                    if 'diagnoses' in json['response']: del json['response']['diagnoses']
                    if 'procedures' in json['response']: del json['response']['procedures']
                    for item in json['response']['search']['items']:
                        for diagnosis in item['diagnosis']:
                            if 'desc' in diagnosis: del diagnosis['desc']
                        for procedure in item['procedure']:
                            if 'desc' in procedure: del procedure['desc']
                    request = dipRequest.parse_obj(json['request'])
                    response = dipResponse.parse_obj(json['response'])
                    
                    requests.append(request)
                    responses.append(response)
            return (requests, responses)
        __2497fac0 = scope()
    for req, res in zip(*__2497fac0):
        if req == request:
            return res
    
    it = request.text
    it = auto.re.sub(r'[^\w\s]', ' ', it)
    it = auto.re.split(r'\s+', it)
    it = map(str.lower, it)
    it = sorted(set(it))
    
    query = ' OR '.join(
        '"' + x.replace('"', '""') + '"'
        for x in it
    )
    
    def scope():
        with Database() as database:
            count ,= database.execute(util.SQLQuery(r'''
                SELECT MAX(ROWID)
                FROM "discharge"
                LIMIT 1
            ''')).fetchone()
        
        return count
    global __2cfbada9
    try: __2cfbada9
    except NameError: __2cfbada9 = scope()
    count = __2cfbada9
    
    def scope(*, beg, end):
        sqlquery = util.SQLQuery(r'''
            SELECT
                json_object
                ( 'hadm_id', "discharge"."hadm_id"
                , 'rank', "rank"
                , 'course', "fts___discharge"."Brief Hospital Course"
                , 'diagnosis', (
                    SELECT
                        json_group_array(json_object
                        ( 'dx10', "diagnosis"."dx10"
                        ) )
                    FROM "diagnosis"
                    WHERE "diagnosis"."hadm_id" = "discharge"."hadm_id"
                    AND "diagnosis"."dx10" IS NOT NULL
                    )
                , 'procedure', (
                    SELECT
                        json_group_array(json_object
                        ( 'pd10', "procedure"."pd10"
                        ) )
                    FROM "procedure"
                    WHERE "procedure"."hadm_id" = "discharge"."hadm_id"
                    AND "procedure"."pd10" IS NOT NULL
                    )
                ) AS "item"
            FROM "fts___discharge"
            JOIN "discharge"
              ON "discharge"."rowid" = "fts___discharge"."rowid"
            WHERE "fts___discharge"."rowid" >= {{ beg }}
              AND "fts___discharge"."rowid" < {{ end }}
              AND "fts___discharge"."Brief Hospital Course" MATCH ?
            ORDER BY "rank"
            LIMIT {{ limit }};
        ''', limit=request.limit, beg=beg, end=end)
        
        with Database() as database:
            df = auto.pd.read_sql_query(sqlquery, database, params=[query])
            # /display df
        
        row = df.iloc[0]
        
        items = []
        for _, row in df.iterrows():
            item = row['item']
            item = dipResponseItem.parse_raw(item)
            
            items.append(item)

        return items
    
    N = 8
    batch = int(auto.math.ceil(count / N))

    it = [
        (beg, min(count, beg + batch))
        for beg in range(0, count, batch)
    ]
    assert N-1 <= len(it) <= N+1, f'Expected ~{N} batches, got {len(it)}'

    with auto.concurrent.futures.ThreadPoolExecutor(len(it)) as pool:
        futures = []
        for beg, end in it:
            future = pool.submit(scope, beg=beg, end=end)
            futures.append(future)
        
        it = auto.concurrent.futures.as_completed(futures)
        it = auto.tqdm.auto.tqdm(it, total=len(futures))
        
        itemses = []
        for future in it:
            items = future.result()
            itemses.extend(items)

    itemses.sort(key=lambda x: x.rank)
    itemses = itemses[:request.limit]

    response = dipResponse(
        search=dipResponseSearch(
            items=itemses,
        ),
    )

    def scope():
        root = config.datadir
        path = root / '__2497fac0.ndjson'
        with path.open('a') as f:
            auto.json.dump({
                'request': request.dict(),
                'response': response.dict(),
            }, f)
            f.write('\n')
        return (__2497fac0[0] + [request], __2497fac0[1] + [response])
    __2497fac0 = scope()
    
    return response


class dischargeResponse(auto.pydantic.BaseModel):
    hadm_id: str
    course: str


@app.get("/MIMIC-IV-Note/discharge/{hadm_id}/")
async def discharge(
    *,
    hadm_id: str,
) -> dischargeResponse:
    with Database() as database:
        sqlquery = util.SQLQuery(r'''
            SELECT
                "discharge"."hadm_id",
                "discharge"."Brief Hospital Course"
            FROM "discharge"
            WHERE "discharge"."hadm_id" = ?
        ''')
        
        df = auto.pd.read_sql_query(sqlquery, database, params=[hadm_id])
        # /display df
    
    row = df.iloc[0]
    
    response = dischargeResponse(
        hadm_id=row['hadm_id'],
        course=row['Brief Hospital Course'],
    )
    
    return response


class underpaymentRequest(auto.pydantic.BaseModel):
    dxs: list[str]
    pds: list[str]
    ndx: auto.typing.Literal[1, 2, 3]
    npd: auto.typing.Literal[1, 2, 3]


@app.post('/underpayment/')
def underpayment(
    *,
    request: underpaymentRequest,
    icd10cm: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10CM,
        ),
    ],
    icd10pcs: auto.typing.Annotated[
        auto.pd.DataFrame,
        auto.fastapi.Depends(
            ICD10PCS,
        ),
    ],
) -> dict:
    underpayment = util.Underpayment(
        dxs=request.dxs,
        pds=request.pds,
        ndx=request.ndx,
        npd=request.npd,
        
        icd10cm=icd10cm,
        icd10pcs=icd10pcs,
    )
    
    return underpayment


@app.get("/")
async def index():
    return {"Hello": "World"}
