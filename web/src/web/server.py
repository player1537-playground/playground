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


@auto.functools.cache
def TabularICD10CM(
) -> auto.pd.DataFrame:
    csv_root = config.datadir
    csv_path = csv_root / 'TabularICD10CM.csv'
    assert csv_path.exists(), csv_path
    
    df = auto.pd.read_csv(
        csv_path,
        dtype=str,
        na_filter=False,
    )
    df.set_index('dx10', inplace=True)
    
    return df


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
def ICD10PCS():
    href = 'https://www.cms.gov/files/zip/2023-icd-10-pcs-order-file-long-and-abbreviated-titles-updated-01/11/2023.zip'
    root = config.datadir
    path = root / 'ICD10PCS.zip'
    if not path.exists():
        with auto.requests.request(
            'GET',
            href,
            stream=True,
        ) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    assert path.exists()

    # with auto.zipfile.ZipFile(path, 'r') as arc:
    #     /auto.pprint.pp arc.infolist()
    # [<ZipInfo filename='icd10pcsOrderFile.pdf' compress_type=deflate external_attr=0x20 file_size=183599 compress_size=173971>,
    #  <ZipInfo filename='icd10pcs_order_2023.txt' compress_type=deflate external_attr=0x20 file_size=12118243 compress_size=1127543>,
    #  <ZipInfo filename='order_addenda_2023.txt' compress_type=deflate external_attr=0x20 file_size=6480 compress_size=1060>]

    # with auto.zipfile.ZipFile(path, 'r') as arc:
    #     with arc.open('icd10pcsOrderFile.pdf', 'r') as f:
    #         with auto.pdfplumber.open(f) as pdf:
    #             print(pdf.pages[0].extract_text())
    # ICD-10-PCS Order File
    # The ICD-10-PCS order file contains a unique “order number” for each valid code or header, a
    # flag distinguishing valid codes from headers, and both long and short descriptions combined in a
    # single file so they are easier to find and use. icd10pcs_order_[year].txt contains ICD-10-PCS
    # (procedure) codes valid beginning October 1 of the upcoming fiscal year.
    # For each ICD-10-PCS code, the order file provides a unique five-digit “order number”. The
    # codes are numbered in “tabular order,” i.e., the order in which the contents of the code system
    # are displayed in the official document containing the system. This includes “headers,” which are
    # not valid codes and are included as a convenience for other uses. The ICD-10-PCS order files
    # will be updated every time the ICD-10-PCS official documents are updated. The order numbers
    # are likely to change with each update.
    # To determine which of two ICD-10-PCS codes comes first in the official document, look up each
    # code in the order file and find its order number. The code with the lower order comes first if its
    # order number is lower, regardless of the characters used in forming the codes themselves.
    # The order file can be used to obtain a standard interpretation of a “range of codes” – any
    # expression composed of two codes with a dash between them (for example 0270-0273). To
    # obtain the list of codes contained in a range, look up the order number of the lower code and the
    # order number of the higher code. Only codes whose order number is at least as high as the lower
    # order number and no higher than the higher order number are in the range.
    # Each line of the order file contains one code. A line is of variable length but never longer than
    # 400 characters maximum. Fields are defined for the ICD-10-PCS order file as follows:
    # Position Length Contents
    # 1 5 Order number, right justified, zero filled.
    # 6 1 Blank
    # 7 7 ICD-10-PCS code
    # 14 1 Blank
    # 15 1 0 if the code is a “header” –not valid for HIPAA-covered transactions.
    # 1 if the code is valid for submission for HIPAA-covered transactions.
    # 16 1 Blank
    # 17 60 Short description
    # 77 1 Blank
    # 78 To end Long description

    root = auto.zipfile.Path(path)
    path = root / 'icd10pcs_order_2023.txt'

    # with path.open('r') as f:
    #     print(f.read(2000))
    # 00001 001     0 Central Nervous System and Cranial Nerves, Bypass            Central Nervous System and Cranial Nerves, Bypass
    # 00002 0016070 1 Bypass Cereb Vent to Nasophar with Autol Sub, Open Approach  Bypass Cerebral Ventricle to Nasopharynx with Autologous Tissue Substitute, Open Approach
    # 00003 0016071 1 Bypass Cereb Vent to Mastoid Sinus w Autol Sub, Open         Bypass Cerebral Ventricle to Mastoid Sinus with Autologous Tissue Substitute, Open Approach
    # 00004 0016072 1 Bypass Cereb Vent to Atrium with Autol Sub, Open Approach    Bypass Cerebral Ventricle to Atrium with Autologous Tissue Substitute, Open Approach
    # 00005 0016073 1 Bypass Cereb Vent to Blood Vess w Autol Sub, Open            Bypass Cerebral Ventricle to Blood Vessel with Autologous Tissue Substitute, Open Approach
    # 00006 0016074 1 Bypass Cereb Vent to Pleural Cav w Autol Sub, Open           Bypass Cerebral Ventricle to Pleural Cavity with Autologous Tissue Substitute, Open Approach
    # 00007 0016075 1 Bypass Cereb Vent to Intestine with Autol Sub, Open Approach Bypass Cerebral Ventricle to Intestine with Autologous Tissue Substitute, Open Approach
    # 00008 0016076 1 Bypass Cereb Vent to Periton Cav w Autol Sub, Open           Bypass Cerebral Ventricle to Peritoneal Cavity with Autologous Tissue Substitute, Open Approach
    # 00009 0016077 1 Bypass Cereb Vent to Urinary Tract w Autol Sub, Open         Bypass Cerebral Ventricle to Urinary Tract with Autologous Tissue Substitute, Open Approach
    # 00010 0016078 1 Bypass Cereb Vent to Bone Mar with Autol Sub, Open Approach  Bypass Cerebral Ventricle to Bone Marrow with Autologous Tissue Substitute, Open Approach
    # 00011 001607A 1 Bypass Cereb Vent to Subgaleal with Autol Sub, Open Approach Bypass Cerebral Ventricle to Subgaleal Space with Autologous Tissue Substitute, Open Approach
    # 00012 001607B 1 Bypass Cereb Vent to Cereb Cistern w Autol Sub, Open         Bypass Cerebral Ventricle to Cerebral Cisterns with Autologous Tissue Substitute, Open Approach
    # 00013 00160J0 1 Byp

    colspecs = []
    names = []

    i = 0
    def scope(n: str, s: str | None):
        names.append(n)

        nonlocal i
        if s is not None:
            colspecs.append((i, i + len(s)))
            i += len(s) + 1
        else:
            colspecs.append((i, auto.sys.maxsize))

    scope('Order', '00001')
    scope('PD', '0016070')
    scope('Is Header', '0')
    scope('Short', 'Central Nervous System and Cranial Nerves, Bypass           ')
    scope('Long', None)

    with path.open('r') as f:
        df = auto.pd.read_fwf(
            f,
            names=names,
            header=0,
            colspecs=colspecs,
            dtype=str,
            na_filter=False,
        )

    df.drop(columns=[
        'Order',
    ], inplace=True)

    df['Is Header'] = df['Is Header'].replace({
        # NOTE(th): This is confusing, but it's how it's coded.
        '0': True,
        '1': False,
    })
    
    df['PD'] = df['PD'].str.strip()
    df.set_index('PD', inplace=True)

    df['Short'] = df['Short'].str.strip()
    df['Long'] = df['Long'].str.strip()

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
            TabularICD10CM,
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
        desc = icd10pcs.loc[pd, 'Long']
        
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
            CompleteICD10CM,
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
            CompleteICD10CM,
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
        desc = icd10pcs.loc[pd, 'Long']
        
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
            CompleteICD10CM,
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
        desc = icd10pcs.loc[code, 'Long']
        return (
            f"search_document: "
            f"The procedure is "
            f"{code}: {desc}"
        )
    
    def Query(*, code: str) -> str:
        desc = icd10pcs.loc[code, 'Long']
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
        desc = icd10pcs.loc[code, 'Long']
        
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
            CompleteICD10CM,
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
        desc = icd10pcs.loc[code, 'Long']
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
        desc = icd10pcs.loc[code, 'Long']
        
        response.best.append(pingResponseItem(
            pd=code,
            desc=desc,
            score=cdist[i],
        ))
    
    return response


@app.get("/")
async def index():
    return {"Hello": "World"}
