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


@app.get("/")
async def index():
    return {"Hello": "World"}
