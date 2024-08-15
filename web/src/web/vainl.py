from __future__ import annotations
from ._auto import auto
from ._config import config


@auto.functools.cache
def VAINL_API_URL():
    return config.vainl.api_url


@auto.functools.cache
def VAINL_API_KEY():
    return config.vainl.api_key


def Identity(x: auto.typing.Any, /) -> str:
    if hasattr(x, 'identity'):
        x = x.identity
    elif isinstance(x, str):
        pass
    elif 'identity' in x:
        x = x['identity']
    else:
        raise ValueError(f'cannot identity {x!r}')
    return x


def Ideation(
    ideation: str,
    /,
    api_url: str | auto.typing.Literal[...] = ...,
    api_key: str | auto.typing.Literal[...] | None = ...,
    cache: dict | auto.typing.Literal[...] | None = ...,
):
    if api_url is Ellipsis:
        api_url = VAINL_API_URL()
    if api_key is Ellipsis:
        api_key = VAINL_API_KEY()
    if cache is Ellipsis:
        global __796c44a7
        try: __796c44a7
        except NameError: __796c44a7 = {}
        cache = __796c44a7

    ideation = Identity(ideation)

    url = api_url
    url = f'{url}linen_den/'
    url = f'{url}ideation/{auto.urllib.parse.quote(ideation)}/'

    headers = {}
    headers['Accept'] = 'application/json'
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'

    ckey = url
    if cache is None or ckey not in cache:
        print(url)
        with auto.requests.request(
            'GET',
            url,
            headers=headers,
        ) as response:
            response.raise_for_status()
            json = response.json()

        if cache is not None:
            cache[ckey] = json

    else:
        json = cache[ckey]

    def Ideation(json):
        json = json.copy()
        identity = json.pop('identity')
        location = json.pop('location')
        material = json.pop('material')
        # magazines_ = json.pop('magazines')
        properties_ = json.pop('properties')
        catalysts_ = json.pop('catalysts')
        assert not json, list(json)

        material = Material(material)

        # magazines = []
        # for magazine in magazines_:
        #     magazine = Magazine(magazine)
        #     magazines.append(magazine)

        properties = []
        for property in properties_:
            property = Property(property)
            properties.append(property)

        catalysts = []
        for catalyst in catalysts_:
            catalyst = Catalyst(catalyst)
            catalysts.append(catalyst)

        return auto.types.SimpleNamespace(
            identity=identity,
            location=location,
            ideation=ideation,
            material=material,
            # magazines=magazines,
            properties=properties,
            catalysts=catalysts,
        )

    def Material(json):
        json = json.copy()
        location = json.pop('location')
        assert not json, list(json)

        return auto.types.SimpleNamespace(
            location=location,
        )

    def Magazine(json):
        json = json.copy()
        identity = json.pop('identity')
        location = json.pop('location')
        assert not json, list(json)

        return auto.types.SimpleNamespace(
            identity=identity,
            location=location,
        )

    def Property(json):
        json = json.copy()
        identity = json.pop('identity')
        fullname = json.pop('fullname')
        delegates = json.pop('delegates')
        metadata = json.pop('metadata')
        assert not json, list(json)

        return auto.types.SimpleNamespace(
            identity=identity,
            fullname=fullname,
            delegates=delegates,
            metadata=metadata,
        )

    def Catalyst(json):
        json = json.copy()
        identity = json.pop('identity')
        assert not json, list(json)

        return auto.types.SimpleNamespace(
            identity=identity,
        )

    json = json.copy()
    ideation = json.pop('ideation')
    assert not json, list(json)

    ideation = Ideation(ideation)

    return ideation


def Property(
    property: str,
    *,
    ideation: str,
    api_url: str | None = None,
    api_key: str | None = None,
    cache: dict | None = None,
):
    if api_url is None:
        api_url = VAINL_API_URL()
    if api_key is None:
        api_key = VAINL_API_KEY()
    if cache is None:
        global __c7752b54
        try: __c7752b54
        except NameError: __c7752b54 = {}
        cache = __c7752b54

    ideation = Identity(ideation)
    property = Identity(property)

    url = api_url
    url = f'{url}linen_den/'
    url = f'{url}ideation/{auto.urllib.parse.quote(ideation)}/'
    url = f'{url}property/{auto.urllib.parse.quote(property)}/'

    ckey = url
    if ckey not in cache:
        with auto.requests.request(
            'GET',
            url,
            headers={
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
        ) as response:
            response.raise_for_status()
            json = response.json()

        cache[ckey] = json

    else:
        json = cache[ckey]

    def Property(json):
        json = json.copy()
        ideation = json.pop('ideation')
        identity = json.pop('identity')
        delegates = json.pop('delegates')
        metadata = json.pop('metadata')
        assert not json, list(json)

        ideation = Ideation(ideation)

        return auto.types.SimpleNamespace(
            identity=identity,
            delegates=delegates,
            metadata=metadata,
        )

    def Ideation(json):
        json = json.copy()
        identity = json.pop('identity')
        location = json.pop('location')
        assert not json, list(json)

        return auto.types.SimpleNamespace(
            identity=identity,
            location=location,
        )

    json = json.copy()
    property = json.pop('property')
    assert not json, list(json)

    property = Property(property)

    return property


def Material(
    properties: list[str],
    *,
    ideation: str,

    baseline: str | list[str] | None = None,
    neighbors: int | None = None,
    varietal: str | None = 'distance',

    verbose: bool | int = False,

    api_url: str | auto.typing.Literal[...] = ...,
    api_key: str | auto.typing.Literal[...] | None = ...,
    cache: dict | None = ...,
):
    if api_url is Ellipsis:
        api_url = VAINL_API_URL()
    if api_key is Ellipsis:
        api_key = VAINL_API_KEY()
    if cache is Ellipsis:
        global __62d2c2d6
        try: __62d2c2d6
        except NameError: __62d2c2d6 = auto.shelve.open(str(config.datadir / 'vainl.Material.cache'))
        cache = __62d2c2d6
    verbose = int(verbose)

    if baseline is not None:
        if isinstance(baseline, str):
            baseline = [baseline]

    ideation = Identity(ideation)

    it = properties
    properties = []
    for property in it:
        property = Identity(property)
        if ':' in property:
            property, _ = property.split(':', 1)
        properties.append(property)

    it = properties
    it = auto.more_itertools.chunked(it, 1)
    if verbose >= 1:
        it = auto.tqdm.auto.tqdm(it)

    ckey = {
        'properties': properties,
        'baseline': baseline,
        'neighbors': neighbors,
        'varietal': varietal,
        'ideation': ideation,
    }
    ckey1 = auto.json.dumps(ckey, sort_keys=True)
    if cache is None or ckey1 not in cache:
        dfs = []
        for properties in it:
            query = []
            if baseline is not None:
                for baseline in baseline:
                    query.append(('b', baseline))

                query.append(('k', neighbors))
                query.append(('varietal', varietal))

            for property in properties:
                query.append(('property', property))

            url = api_url
            url = f'{url}linen_den/'
            url = f'{url}ideation/{auto.urllib.parse.quote(ideation)}/'
            url = f'{url}material/'
            url = f'{url}?{auto.urllib.parse.urlencode(query)}'

            headers = {}
            headers['Accept'] = 'text/csv'
            if api_key is not None:
                headers['Authorization'] = f'Bearer {api_key}'

            ckey = {
                'url': url,
                'headers': headers,
            }
            ckey = auto.json.dumps(ckey, sort_keys=True)
            if cache is None or ckey not in cache:
                print(url)
                df = auto.pd.read_csv(
                    url,
                    dtype={
                        'where': str,
                    },
                    quoting=auto.csv.QUOTE_NONNUMERIC,
                    low_memory=False,
                    storage_options={
                        **headers,
                    },
                ).set_index('where')

                if 'geometry' in df.columns:
                    geometry = df['geometry']
                    geometry = geometry.apply(lambda s: (
                        auto.shapely.wkt.loads(s)
                    ) if not auto.pd.isna(s) else (
                        None
                    ))
                    df = df.assign(**{
                        'geometry': geometry,
                    })
                    df = auto.geopandas.GeoDataFrame(df, geometry='geometry')

                if cache is not None:
                    cache[ckey] = df

            else:
                df = cache[ckey]

            dfs.append(df)

        df = auto.pd.concat(dfs, axis=1)
        
        cache[ckey1] = df
    
    else:
        df = cache[ckey1]

    return df


def Bincount(
    *,
    ideation: str,

    xproperty: str,

    baseline: str | list[str] | None = None,
    neighbors: int | None = None,
    varietal: str | None = 'distance',

    width: int | str = 600,
    height: int | str = 400,

    api_url: str | None = None,
    api_key: str | None = None,
    cache: dict | None = None,
):
    if api_url is None:
        api_url = VAINL_API_URL()
    if api_key is None:
        api_key = VAINL_API_KEY()
    if cache is None:
        global __2722ed6a
        try: __2722ed6a
        except NameError: __2722ed6a = {}
        cache = __2722ed6a

    ideation = Identity(ideation)
    xproperty = Identity(xproperty)

    query = []
    query.append(('x', xproperty))
    if baseline is not None:
        if isinstance(baseline, str):
            baseline = [baseline]

        for baseline in baseline:
            query.append(('b', baseline))

        query.append(('k', neighbors))
        query.append(('v', varietal))
    query.append(('w', width))
    query.append(('h', height))

    url = f'{url}linen_den/'
    url = f'{url}ideation/{auto.urllib.parse.quote(ideation)}/'
    url = f'{url}bincount/'
    url = f'{url}?{auto.urllib.parse.urlencode(query)}'

    ckey = url
    if ckey not in cache:
        with auto.requests.request(
            'GET',
            url,
            headers={
                'Accept': 'vega',
                'Authorization': f'Bearer {api_key}',
            },
        ) as response:
            response.raise_for_status()
            json = response.json()

        cache[ckey] = json

    else:
        json = cache[ckey]

    return json


def Dotchart(
    *,
    ideation: str,

    xproperty: str,
    yproperty: str,

    baseline: str | None | list[str] = None,
    neighbors: int | None = None,
    varietal: str | None = 'distance',

    width: int | str = 600,
    height: int | str = 400,

    api_url: str | None = None,
    api_key: str | None = None,
    cache: dict | None = None,
):
    if api_url is None:
        api_url = VAINL_API_URL()
    if api_key is None:
        api_key = VAINL_API_KEY()
    if cache is None:
        global __3879fd46
        try: __3879fd46
        except NameError: __3879fd46 = {}
        cache = __3879fd46

    ideation = Identity(ideation)
    xproperty = Identity(xproperty)
    yproperty = Identity(yproperty)

    query = []
    query.append(('xproperty', xproperty))
    query.append(('yproperty', yproperty))
    if baseline is not None:
        if isinstance(baseline, str):
            baseline = [baseline]

        for baseline in baseline:
            query.append(('baseline', baseline))

        query.append(('neighbor', neighbors))
        query.append(('varietal', varietal))
    query.append(('width', width))
    query.append(('height', height))

    url = f'{url}linen_den/'
    url = f'{url}ideation/{auto.urllib.parse.quote(ideation)}/'
    url = f'{url}dotchart/'
    url = f'{url}?{auto.urllib.parse.urlencode(query)}'

    ckey = url
    if ckey not in cache:
        with auto.requests.request(
            'GET',
            url,
            headers={
                'Accept': 'vega',
                'Authorization': f'Bearer {api_key}',
            },
        ) as response:
            response.raise_for_status()
            json = response.json()

        cache[ckey] = json

    else:
        json = cache[ckey]

    return json



def scope():
    ideation = Ideation(
        'By Census Tract',
        cache=None,
    )
    print(auto.pprint.pformat(ideation, indent=2)[:10000])

    dotchart = Dotchart(
        ideation=ideation,
        xproperty='P0F38078',
        yproperty='P017296A',
    )
    print(auto.pprint.pformat(dotchart, indent=2)[:4000])

# /scope
