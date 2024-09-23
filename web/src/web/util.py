"""

"""

from __future__ import annotations
from ._auto import auto
from ._config import config


class LLM:
    """
    A class for interacting with a language model API to generate completions and embeddings.

    The LLM class provides methods to send prompts to a language model API and retrieve the
    generated completions or embeddings. It handles the details of the API request and response,
    and provides options for caching results to avoid redundant API calls.

    Parameters
    ----------
    model : str or None, default=None
        The name of the language model to use for generating completions and embeddings.
        If None, the model must be specified in each call to `complete` or `embed`.

    api_url : str
        The URL of the API endpoint to use for generating completions and embeddings.

    api_key : str or None, default=...
        The API key to use for authentication when making requests to the API. If not specified,
        the `api_key_name` parameter must be specified to retrieve the key from the Colab
        user data.

    api_key_name : str or None, default=None
        The name of the Colab user data key that stores the API key. If None, the `api_key`
        parameter must be specified directly.

    session : requests.Session or None, default=None
        The `requests.Session` object to use for making API requests. If None, a new session
        will be created.

    prompt_kwargs : dict or None, default=None
        A dictionary of default keyword arguments to use for the `complete` method. These
        arguments will be merged with any arguments specified in each call to `complete`.

    cache : dict or None, default=None
        A dictionary to use as a cache for storing API responses. If None, a new empty
        dictionary will be created.

    Methods
    -------
    complete(**prompt) -> dict
        Generate a completion for the given prompt using the language model API.

    embed(input) -> numpy.ndarray
        Generate embeddings for the given input text or texts using the language model API.

    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_url: str,
        api_key: str | None | auto.typing.Literal[Ellipsis] = ...,
        api_key_name: str | None = None,
        session: auto.requests.Session | None = None,
        prompt_kwargs: dict[str, auto.typing.Any] | None = None,
        cache: dict[str, auto.typing.Any] | None = None,
    ):
        if api_key is Ellipsis:
            assert api_key_name is not None, \
                "Either 'api_key' or 'api_key_name' must be specified."
            api_key = auto.google.colab.userdata.get(api_key_name)

        if session is None:
            session = auto.requests.Session()
        if prompt_kwargs is None:
            prompt_kwargs = {}
        if cache is None:
            cache = {}

        self.default_model = model
        self.default_api_url = api_url
        self.default_api_key = api_key
        self.default_session = session
        self.default_prompt_kwargs = prompt_kwargs
        self.default_cache = cache

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def complete(
        self,
        *,
        api_url: str | auto.typing.Literal[...] = ...,
        api_key: str | None | auto.typing.Literal[Ellipsis] = ...,
        session: auto.requests.Session | auto.typing.Literal[...] = ...,
        cache: dict[str, auto.typing.Any] | None | auto.typing.Literal[...] = ...,
        model: str | None | auto.typing.Literal[...] = ...,
        **prompt,
    ) -> dict[str, auto.typing.Any]:
        if api_url is Ellipsis:
            api_url = self.default_api_url
        if api_key is Ellipsis:
            api_key = self.default_api_key
        if session is Ellipsis:
            session = self.default_session
        if cache is Ellipsis:
            cache = self.default_cache
        if model is Ellipsis:
            model = self.default_model

        prompt = self.default_prompt_kwargs | prompt
        if model is not None:
            prompt = prompt | dict(
                model=model,
            )

        is_text = 'prompt' in prompt
        is_chat = 'messages' in prompt
        assert is_text != is_chat, \
            "Either 'prompt' or 'messages' must be specified."

        if is_text:
            url = f'{api_url}v1/completions'
        else:
            url = f'{api_url}v1/chat/completions'

        headers = {
            'Content-Type': 'application/json',
        }
        if api_key is not None:
            headers['Authorization'] = f'Bearer {api_key}'

        ckey = auto.json.dumps(prompt, sort_keys=True)
        if cache is None or ckey not in cache:
            with session.request(
                'POST',
                url,
                headers=headers,
                json=prompt,
            ) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    raise ValueError(f'API error: {response.text}') from e
                output = response.json()

            if cache is not None:
                cache[ckey] = output
            self.was_cached = False

        else:
            output = cache[ckey]
            self.was_cached = True

        return output

    def embed(
        self,
        input: str | list[str],
        *,
        api_url: str | None = None,
        api_key: str | None | auto.typing.Literal[Ellipsis] = ...,
        session: auto.requests.Session | None = None,
        cache: dict[str, auto.typing.Any] | None = None,
        model: str | None | auto.typing.Literal[Ellipsis] = ...,
        verbose: bool | int = False,
    ) -> auto.np.ndarray[float]:
        if api_url is None:
            api_url = self.default_api_url
        if api_key is Ellipsis:
            api_key = self.default_api_key
        if session is None:
            session = self.default_session
        if cache is None:
            cache = self.default_cache
        if model is Ellipsis:
            model = self.default_model
        verbose = int(verbose)

        if isinstance(input, str):
            input = [input]
            one = True
        else:
            one = False

        it = range(0, (N := len(input)), (K := 100))
        if verbose >= 1:
            it = list(it)
            it = auto.tqdm.auto.tqdm(it)

        embeds = []
        for i in it:
            json = dict(
                input=input[i:i+K],
            )
            if model is not None:
                json |= dict(
                    model=model,
                )

            ckey = auto.json.dumps(json, sort_keys=True)
            if ckey not in cache:
                with session.request(
                    'POST',
                    f'{api_url}v1/embeddings',
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {api_key}',
                    },
                    json=json,
                ) as response:
                    response.raise_for_status()
                    output = response.json()

                self.was_cached = False
                cache[ckey] = output

            else:
                self.was_cached = True
                output = cache[ckey]

            for data in output['data']:
                embed = data['embedding']
                embeds.append(embed)

        embeds = auto.np.array(embeds)

        if one:
            embeds = embeds[0]

        return embeds

    def tokenize(
        self,

        input: str,
        *,
        add_special: bool = False,

        api_url: str | auto.typing.Literal[...] = ...,
        api_key: str | None | auto.typing.Literal[...] = ...,
        session: auto.requests.Session | auto.typing.Literal[...] = ...,
        cache: dict[str, auto.typing.Any] | auto.typing.Literal[...] = ...,
        model: str | None | auto.typing.Literal[...] = ...,
    ) -> list[int]:
        if api_url is Ellipsis:
            api_url = self.default_api_url
        if api_key is Ellipsis:
            api_key = self.default_api_key
        if session is Ellipsis:
            session = self.default_session
        if cache is Ellipsis:
            cache = self.default_cache
        if model is Ellipsis:
            model = self.default_model

        url = api_url
        url = f'{url}tokenize'

        json = dict(
            content=input,
            add_special=add_special,
        )
        if model is not None:
            json |= dict(
                model=model,
            )

        ckey = auto.json.dumps(json, sort_keys=True)
        if ckey not in cache:
            with session.request(
                'POST',
                url,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                json=json,
            ) as response:
                response.raise_for_status()
                json = response.json()

            self.was_cached = False
            cache[ckey] = json

        else:
            self.was_cached = True
            json = cache[ckey]

        tokens = []
        for token in json['tokens']:
            tokens.append(token)

        return tokens

    def detokenize(
        self,

        tokints: list[int],
        *,
        api_url: str | auto.typing.Literal[...] = ...,
        api_key: str | None | auto.typing.Literal[...] = ...,
        session: auto.requests.Session | auto.typing.Literal[...] = ...,
        cache: dict[str, auto.typing.Any] | auto.typing.Literal[...] = ...,
    ) -> list[str]:
        if api_url is Ellipsis:
            api_url = self.default_api_url
        if api_key is Ellipsis:
            api_key = self.default_api_key
        if session is Ellipsis:
            session = self.default_session
        if cache is Ellipsis:
            cache = self.default_cache

        url = api_url
        url = f'{url}detokenize'

        tokens = []
        for tokint in tokints:
            json = dict(
                tokens=[tokint],
            )

            ckey = auto.json.dumps(json, sort_keys=True)
            if ckey not in cache:
                with session.request(
                    'POST',
                    url,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {api_key}',
                    },
                    json=json,
                ) as response:
                    response.raise_for_status()
                    json = response.json()

                self.was_cached = False
                cache[ckey] = json

            else:
                self.was_cached = True
                json = cache[ckey]

            token = json['content']

            tokens.append(token)

        return tokens


@auto.dataclasses.dataclass
class Chunk:
    offset: int
    length: int
    text: str
    
    def __lt__(self, other):
        return self.offset < other.offset or self.length < other.length or self.text < other.text
    
    def __eq__(self, other):
        return self.offset == other.offset and self.length == other.length and self.text == other.text


def Chunks(
    text: str,
    /,
    *,
    min_size: int | None = None,
    avg_size: int = 64,
    max_size: int | None = None,
) -> list[Chunk]:
    if min_size is None:
        min_size = avg_size // 2

    if max_size is None:
        max_size = avg_size * 2

    text: bytes = text.encode('ascii', errors='ignore')

    chunks = []

    for chunk in auto.fastcdc.fastcdc_py.chunk_generator(
        memview=memoryview(text),
        min_size=min_size,
        avg_size=avg_size,
        max_size=max_size,
        fat=False,
        hf=None,
    ):
        chunk = Chunk(
            offset=chunk.offset,
            length=chunk.length,
            text=text[chunk.offset:chunk.offset + chunk.length].decode('ascii', errors='ignore'),
        )

        chunks.append(chunk)

    return chunks


def Overlap(
    k: int,
    *,
    chunks: list[Chunk],
) -> list[Chunk]:
    rates = []
    left_empty = Chunk(offset=0, length=0, text = '')
    right_empty = Chunk(offset=max(c.offset + c.length for c in chunks), length=0, text = '')
    it = chunks
    it = [left_empty]*(k//2) + it + [right_empty]*(k//2)
    it = auto.more_itertools.windowed(it, k)
    it = (
        Chunk(
            offset=min(c.offset for c in window),
            length=sum(c.length for c in window),
            text=''.join(c.text for c in window),
        )
        for window in it
    )
    it = list(it)

    return it


#--- PROMPT: Create LLM user/assistant prompts using jinja2 templates

class PROMPT:
    """
    A class for creating and rendering prompt templates using the Jinja2 templating engine.

    The PROMPT class provides a way to define and render prompt templates that can be used
    to generate formatted prompts for language models. It uses the Jinja2 templating engine
    to allow for dynamic generation of prompts based on input variables.

    Methods
    -------
    register(name: str, template: str) -> None
        Register a new prompt template with the given name and template string.

    __new__(cls, s: str, /, **query) -> PROMPT
        Create a new PROMPT instance with the given template string and query variables.

    __call__(**query) -> dict
        Render the prompt template with the given query variables and return the resulting
        prompt dictionary.

    The rendered prompt dictionary can include the following keys:
    - "messages": A list of message dictionaries, where each dictionary represents a single
    message in the conversation, with keys for the role (e.g., "user" or "assistant") and
    the content of the message.
    - "prompt": A string representing the prompt text.
    - "grammar": A string representing a grammar for parsing the model's response.
    - "parser": A string representing a regular expression pattern for parsing the model's
    response.

    Exactly one of "messages" or "prompt" must be specified in the rendered prompt dictionary.

    """

    templates = {}
    environment = auto.jinja2.Environment(
        loader=auto.jinja2.DictLoader(templates),
        undefined=auto.jinja2.StrictUndefined,
    )
    environment.globals.update({
        'auto': auto,
    })

    @classmethod
    def register(PROMPT, name: str, template: str, /):
        PROMPT.templates[name] = template

    def __new__(PROMPT, s: str, /, **query):
        Prompt = super().__new__(PROMPT)
        Prompt.template = PROMPT.environment.from_string(s)

        if query:
            prompt = Prompt(**query)
            return prompt

        return Prompt

    def __call__(self, **query):
        template = self.template

        context = {}

        _messages = None
        def AddMessage(role: str, content: str):
            nonlocal _messages
            if _messages is None:
                _messages = []
            content = content.strip()
            _messages.append(dict(
                role=role,
                content=content,
            ))
            return f'<Message({role!r}, {content!r})>'
        context |= dict(
            user=lambda caller: AddMessage('user', caller()),
            assistant=lambda caller: AddMessage('assistant', caller()),
            system=lambda caller: AddMessage('system', caller()),
        )

        _prompt = None
        def SetPrompt(prompt: str):
            nonlocal _prompt
            _prompt = prompt
            return f'<Prompt({prompt!r})>'
        context |= dict(
            prompt=lambda caller: SetPrompt(caller()),
        )

        _grammar = None
        def SetGrammar(grammar: str):
            nonlocal _grammar
            _grammar = grammar
            return f'<Grammar({grammar!r})>'
        context |= dict(
            grammar=lambda caller: SetGrammar(caller()),
        )

        _parser = None
        def SetParser(parser: str):
            nonlocal _parser
            _parser = parser
            return f'<Parser({parser!r})>'
        context |= dict(
            parser=lambda caller: SetParser(caller()),
        )

        context |= query

        _ = template.render(
            **context,
        )

        prompt = auto.collections.UserDict(
        )

        assert (bool(_messages) != bool(_prompt)), \
            f"Exactly one of 'messages' or 'prompt' must be specified."
        if _messages is not None:
            prompt |= dict(
                messages=_messages,
            )
        elif _prompt is not None:
            prompt |= dict(
                prompt=_prompt,
            )
        else:
            assert False

        if _grammar is not None:
            prompt |= dict(
                grammar=_grammar,
            )

        if _parser is not None:
            prompt.parser = _parser
        return prompt


def __ICD10CM_a(
    year: int,
    *,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    tmp_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_name: str = '__ICD10CM.tmp',
    tmp_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    cache_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_name: str | auto.typing.Literal[...] = ...,
    cache_names: dict[int, str] = {
        2025: '_ICD10CM2025A.csv',
        2024: '_ICD10CM2024A.csv',
        2023: '_ICD10CM2023A.csv',
        2022: '_ICD10CM2022A.csv',
        2021: '_ICD10CM2021A.csv',
        2020: '_ICD10CM2020A.csv',
        2019: '_ICD10CM2019A.csv',
        2018: '_ICD10CM2018A.csv',
        2017: '_ICD10CM2017A.csv',
    },
    cache_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    zip_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    zip_name: str | auto.typing.Literal[...] = ...,
    zip_names: dict[int, str] = {
        2025: '__ICD10CM2025A.zip',
        2024: '__ICD10CM2024A.zip',
        2023: '__ICD10CM2023A.zip',
        2022: '__ICD10CM2022A.zip',
        2021: '__ICD10CM2021A.zip',
        2020: '__ICD10CM2020A.zip',
        2019: '__ICD10CM2019A.zip',
        2018: '__ICD10CM2018A.zip',
        2017: '__ICD10CM2017A.zip',
    },
    zip_path: auto.os.PathLike | auto.typing.Literal[...] = ...,
    zip_href: str | auto.typing.Literal[...] = ...,
    zip_hrefs: dict[int, str] = {
        2025: 'https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip',
        2024: 'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024-Update/icd10cm-Codes-Descriptions-April-2024.zip',
        2023: 'https://www.cms.gov/files/zip/2023-code-descriptions-tabular-order-updated-01/11/2023.zip',
        2022: 'https://www.cms.gov/files/zip/2022-code-descriptions-tabular-order-updated-02012022.zip',
        2021: 'https://www.cms.gov/files/zip/2021-code-descriptions-tabular-order-updated-12162020.zip',
        2020: 'https://www.cms.gov/medicare/coding/icd10/downloads/2020-icd-10-cm-codes.zip',
        2019: 'https://www.cms.gov/medicare/coding/icd10/downloads/2019-icd-10-cm-code-descriptions.zip',
        2018: 'https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-code-descriptions.zip',
        2017: 'https://www.cms.gov/medicare/coding/icd10/downloads/2017-icd10-code-descriptions.zip',
    },

    txt_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    txt_name: str | auto.typing.Literal[...] = ...,
    txt_names: dict[int, str] = {
        2025: 'icd10cm_codes_2025.txt',
        2024: 'icd10cm-codes-April-2024.txt',
        2023: 'icd10cm_codes_2023.txt',
        2022: 'Code Descriptions/icd10cm_codes_2022.txt',
        2021: '2021-code-descriptions-tabular-order/icd10cm_codes_2021.txt',
        2020: '2020 Code Descriptions/icd10cm_codes_2020.txt',
        2019: 'icd10cm_codes_2019.txt',
        2018: 'icd10cm_codes_2018.txt',
        2017: 'icd10cm_codes_2017.txt',
    },
    txt_path: auto.os.PathLike | auto.typing.Literal[...] = ...,
) -> auto.pd.DataFrame:
    if root is ...:
        root = config.datadir

    if cache_path is ...:
        if cache_root is ...:
            cache_root = root
        if cache_name is ...:
            cache_name = cache_names[year]
        cache_path = cache_root / cache_name

    if not cache_path.exists():
        if tmp_path is ...:
            if tmp_root is ...:
                tmp_root = root
            tmp_path = tmp_root / tmp_name

        if zip_path is ...:
            if zip_root is ...:
                zip_root = root
            if zip_name is ...:
                zip_name = zip_names[year]
            zip_path = zip_root / zip_name

        if not zip_path.exists():
            if zip_href is ...:
                zip_href = zip_hrefs[year]

            with auto.requests.request(
                'GET',
                zip_href,
                stream=True,
            ) as r:
                r.raise_for_status()
                with tmp_path.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            tmp_path.rename(zip_path)
        assert zip_path.exists()

        if txt_path is ...:
            if txt_root is ...:
                txt_root = auto.zipfile.Path(zip_path)
            if txt_name is ...:
                txt_name = txt_names[year]
            txt_path = txt_root / txt_name

        if not txt_path.exists():
            with auto.zipfile.ZipFile(zip_path) as arc:
                auto.pprint.pp(arc.infolist())
        assert txt_path.exists()

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
        with txt_path.open('r') as f:
            for line in f:
                code, desc = auto.re.split(r'\s+', line, maxsplit=1)
                code = code.strip()
                desc = desc.strip()
                df.append((code, desc))

        df = auto.pandas.DataFrame(
            df,
            columns=['dx10', 'desc'],
        )
        df.set_index([
            'dx10',
        ], inplace=True)
        df.sort_index(inplace=True)

        with tmp_path.open('w') as f:
            df.to_csv(
                f,
                index=True,
                header=True,
                quoting=auto.csv.QUOTE_NONNUMERIC,
            )

        tmp_path.rename(cache_path)
    assert cache_path.exists()

    with cache_path.open('r') as f:
        df = auto.pandas.read_csv(
            f,
            dtype=str,
            na_filter=False,
            quoting=auto.csv.QUOTE_NONNUMERIC,
        )

    df.set_index([
        'dx10',
    ], inplace=True)
    df.sort_index(inplace=True)

    return df


@auto.functools.cache
def __ICD10CM_b(
    year: int,
    *,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    cache_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_name: str | auto.typing.Literal[...] = ...,
    cache_names: dict[int, str] = {
        2025: '_ICD10CM2025B.csv',
        2024: '_ICD10CM2024B.csv',
        2023: '_ICD10CM2023B.csv',
        2022: '_ICD10CM2022B.csv',
        2021: '_ICD10CM2021B.csv',
        2020: '_ICD10CM2020B.csv',
        2019: '_ICD10CM2019B.csv',
        2018: '_ICD10CM2018B.csv',
        2017: '_ICD10CM2017B.csv',
    },
    cache_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    tmp_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_name: str = '__ICD10CM.tmp',
    tmp_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    zip_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    zip_name: str | auto.typing.Literal[...] = ...,
    zip_names: dict[int, str] = {
        2025: '__ICD10CM2025B.zip',
        2024: '__ICD10CM2024B.zip',
        2023: '__ICD10CM2023B.zip',
        2022: '__ICD10CM2022B.zip',
        2021: '__ICD10CM2021B.zip',
        2020: '__ICD10CM2020B.zip',
        2019: '__ICD10CM2019B.zip',
        2018: '__ICD10CM2018B.zip',
        2017: '__ICD10CM2017B.zip',
    },
    zip_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    zip_href: str | auto.typing.Literal[...] = ...,
    zip_hrefs: dict[int, str] = {
        2025: 'https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip',
        2024: 'https://www.cms.gov/files/zip/2024-code-tables-tabular-and-index-updated-02/01/2024.zip',
        2023: 'https://www.cms.gov/files/zip/2023-code-tables-tabular-and-index-updated-01/11/2023.zip',
        2022: 'https://www.cms.gov/files/zip/2022-code-tables-tabular-and-index-updated-02012022.zip',
        2021: 'https://www.cms.gov/files/zip/2021-code-tables-tabular-and-index-updated-12162020.zip',
        2020: 'https://www.cms.gov/medicare/coding/icd10/downloads/2020-icd-10-cm-code-tables.zip',
        2019: 'https://www.cms.gov/medicare/coding/icd10/downloads/2019-icd-10-cm-tables-and-index.zip',
        2018: 'https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-table-and-index.zip',
        2017: 'https://www.cms.gov/medicare/coding/icd10/downloads/2017-icd10-code-tables-index.zip',
    },

    xml_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    xml_name: str | auto.typing.Literal[...] = ...,
    xml_names: dict[int, str] = {
        2025: 'icd10cm_tabular_2025.xml',
        2024: 'icd10cm_tabular_2024.xml',
        2023: 'icd10cm_tabular_2023.xml',
        2022: 'Table and Index/icd10cm_tabular_2022.xml',
        2021: '2021-code-tables-and-index/icd10cm_tabular_2021.xml',
        2020: '2020 Table and Index/icd10cm_tabular_2020.xml',
        2019: 'icd10cm_tabular_2019.xml',
        2018: 'icd10cm_tabular_2018.xml',
        2017: 'icd10cm_tabular_2017.xml',
    },
    xml_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
) -> auto.pd.DataFrame:
    if root is ...:
        root = config.datadir

    if cache_path is ...:
        if cache_root is ...:
            cache_root = root
        if cache_name is ...:
            cache_name = cache_names[year]
        cache_path = cache_root / cache_name

    if not cache_path.exists():
        if tmp_path is ...:
            if tmp_root is ...:
                tmp_root = root
            tmp_path = tmp_root / tmp_name

        if zip_path is ...:
            if zip_root is ...:
                zip_root = root
            if zip_name is ...:
                zip_name = zip_names[year]
            zip_path = zip_root / zip_name

        if not zip_path.exists():
            if zip_href is ...:
                zip_href = zip_hrefs[year]

            with auto.requests.request(
                'GET',
                zip_href,
                stream=True,
            ) as r:
                r.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            tmp_path.rename(zip_path)
        assert zip_path.exists()

        # with auto.zipfile.ZipFile(zip_path) as arc:
        #     /auto.pprint.pp arc.infolist()

        if xml_path is ...:
            if xml_root is ...:
                xml_root = auto.zipfile.Path(zip_path)
            if xml_name is ...:
                xml_name = xml_names[year]
            xml_path = xml_root / xml_name

        if not xml_path.exists():
            with auto.zipfile.ZipFile(zip_path) as arc:
                auto.pprint.pp(arc.infolist())
        assert xml_path.exists()

        with xml_path.open('r') as f:
            soup = auto.bs4.BeautifulSoup(f, 'xml')


        # <chapter>
        #   <name>22</name>
        #   <desc>Codes for special purposes (U00-U85)</desc>
        #   <sectionIndex>
        #     <sectionRef first="U00" last="U49" id="U00-U49">
        #        Provisional assignment of new diseases of uncertain etiology or emergency use
        #     </sectionRef>
        #   </sectionIndex>
        #   <section id="U00-U49">
        #     <desc>Provisional assignment of new diseases of uncertain etiology or emergency use (U00-U49)</desc>
        #     <diag>
        #       <name>U07</name>
        #       <desc>Emergency use of U07</desc>
        #       <diag>
        #         <name>U07.0</name>
        #         <desc>Vaping-related disorder</desc>
        #         <inclusionTerm>
        #           <note>Dabbing related lung damage</note>
        #           <note>Dabbing related lung injury</note>
        #           <note>E-cigarette, or vaping, product use associated lung injury [EVALI]</note>
        #           <note>Electronic cigarette related lung damage</note>
        #           <note>Electronic cigarette related lung injury</note>
        #         </inclusionTerm>
        #         <useAdditionalCode>
        #           <note>code to identify manifestations, such as:</note>
        #           <note>abdominal pain (R10.84)</note>
        #           <note>acute respiratory distress syndrome (J80)</note>
        #           <note>diarrhea (R19.7)</note>
        #           <note>drug-induced interstitial lung disorder (J70.4)</note>
        #           <note>lipoid pneumonia (J69.1)</note>
        #           <note>weight loss (R63.4)</note>
        #         </useAdditionalCode>
        #       </diag>
        #       <diag>
        #         <name>U07.1</name>
        #         <desc>COVID-19</desc>
        #         <useAdditionalCode>
        #           <note>code to identify pneumonia or other manifestations, such as:</note>
        #           <note>pneumonia due to COVID-19 (J12.82)</note>
        #         </useAdditionalCode>
        #         <useAdditionalCode>
        #           <note>code, if applicable, for associated conditions such as:</note>
        #           <note>COVID-19 associated coagulopathy (D68.8)</note>
        #           <note>disseminated intravascular coagulation (D65)</note>
        #           <note>hypercoagulable states (D68.69)</note>
        #           <note>thrombophilia (D68.69)</note>
        #         </useAdditionalCode>
        #         <excludes2>
        #           <note>coronavirus as the cause of diseases classified elsewhere (B97.2-)</note>
        #           <note>pneumonia due to SARS-associated coronavirus (J12.81)</note>
        #         </excludes2>
        #       </diag>
        #     </diag>
        #     <diag>
        #       <name>U09</name>
        #       <desc>Post COVID-19 condition</desc>
        #       <diag>
        #         <name>U09.9</name>
        #         <desc>Post COVID-19 condition, unspecified</desc>
        #         <inclusionTerm>
        #           <note>Post-acute sequela of COVID-19</note>
        #         </inclusionTerm>
        #         <codeFirst>
        #           <note>the specific condition related to COVID-19 if known, such as:</note>
        #           <note>chronic respiratory failure (J96.1-)</note>
        #           <note>loss of smell (R43.8)</note>
        #           <note>loss of taste (R43.8)</note>
        #           <note>multisystem inflammatory syndrome (M35.81)</note>
        #           <note>pulmonary embolism (I26.-)</note>
        #           <note>pulmonary fibrosis (J84.10)</note>
        #         </codeFirst>
        #         <notes>
        #           <note>This code enables establishment of a link with COVID-19.</note>
        #           <note>This code is not to be used in cases that are still presenting with active COVID-19.  However, an exception is made in cases of re-infection with COVID-19, occurring with a condition related to prior COVID-19.</note>
        #         </notes>
        #       </diag>
        #     </diag>
        #   </section>
        # </chapter>

        # diags = []

        tabular = soup.find_all('ICD10CM.tabular', recursive=False)
        assert len(tabular) == 1, len(tabular)
        tabular ,= tabular

        chapters = tabular.find_all('chapter', recursive=False)
        assert len(chapters) > 0, len(chapters)

        df = []
        for chapter in chapters:
            chapter_name_ = chapter.find('name', recursive=False)
            assert chapter_name_ is not None
            chapter_name = chapter_name_.text

            chapter_desc = chapter.find('desc', recursive=False)
            assert chapter_desc is not None, chapter_name
            chapter_desc = chapter_desc.text

            sections = chapter.find_all('section', recursive=False)
            assert len(sections) > 0, len(sections)

            for section in sections:
                section_desc_ = section.find('desc', recursive=False)
                assert section_desc_ is not None, chapter_name
                section_desc = section_desc_.text

                def walk(diag, prev=[]):
                    diag_name_ = diag.find('name', recursive=False)
                    assert diag_name_ is not None, (chapter_name, chapter_desc, section_desc, prev)
                    diag_name = diag_name_.text

                    diag_desc_ = diag.find('desc', recursive=False)
                    assert diag_desc_ is not None, (chapter_name, chapter_desc, section_desc, prev, diag_name)
                    diag_desc = diag_desc_.text

                    full_name = diag_name.replace('.', '')

                    full_desc = ' > '.join([
                        chapter_desc,
                        section_desc,
                    ] + [
                        f'{prev_desc} ({prev_name})'
                        for prev_name, prev_desc in prev
                    ] + [
                        f'{diag_desc} ({diag_name})',
                    ])

                    yield (full_name, full_desc)

                    diags = diag.find_all('diag', recursive=False)
                    for diag in diags:
                        yield from walk(diag, prev=prev + [(diag_name, diag_desc)])

                diags = section.find_all('diag', recursive=False)
                # assert len(diags) > 0, (chapter_name, chapter_desc, section_desc, len(diags))

                for diag in diags:
                    for diag_name, diag_desc in walk(diag):
                        df.append((diag_name, diag_desc))

        df = auto.pd.DataFrame(
            df,
            columns=[
                'dx10',
                'desc',
            ],
        )
        df.set_index([
            'dx10',
        ], inplace=True)
        df.sort_index(inplace=True)

        with tmp_path.open('w') as f:
            df.to_csv(
                f,
                index=True,
                header=True,
                quoting=auto.csv.QUOTE_NONNUMERIC,
            )

        tmp_path.rename(cache_path)
    assert cache_path.exists()

    with cache_path.open('r') as f:
        df = auto.pd.read_csv(
            f,
            dtype=str,
            na_filter=False,
            quoting=auto.csv.QUOTE_NONNUMERIC,
        )

    df.set_index([
        'dx10',
    ], inplace=True)
    df.sort_index(inplace=True)

    return df


@auto.functools.cache
def ICD10CM(
    *,
    strict: bool = False,

    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    cache_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_name: str = 'ICD10CM.csv',
    cache_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_href: str | None = (
        None  # recompute
        # 'https://accona.eecs.utk.edu/ICD10CM.csv'  # bootstrap
    ),

    tmp_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_name: str = '__ICD10CM.tmp',
) -> auto.pd.DataFrame:
    if root is ...:
        root = config.datadir

    if cache_path is ...:
        if cache_root is ...:
            cache_root = root
        cache_path = cache_root / cache_name

    if not cache_path.exists():
        if tmp_path is ...:
            if tmp_root is ...:
                tmp_root = root
            tmp_path = tmp_root / tmp_name

        if cache_href is not None:
            with auto.requests.request(
                'GET',
                cache_href,
                stream=True,
            ) as r:
                r.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        else:
            def scope():
                dfs = []
                dfs.append(__ICD10CM_a(2017))
                dfs.append(__ICD10CM_a(2018))
                dfs.append(__ICD10CM_a(2019))
                dfs.append(__ICD10CM_a(2020))
                dfs.append(__ICD10CM_a(2021))
                dfs.append(__ICD10CM_a(2022))
                dfs.append(__ICD10CM_a(2023))
                dfs.append(__ICD10CM_a(2024))
                dfs.append(__ICD10CM_a(2025))

                df = auto.pd.concat(dfs)
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10cmA = scope()

            def scope():
                dfs = []
                dfs.append(__ICD10CM_b(2017))
                dfs.append(__ICD10CM_b(2018))
                dfs.append(__ICD10CM_b(2019))
                dfs.append(__ICD10CM_b(2020))
                dfs.append(__ICD10CM_b(2021))
                dfs.append(__ICD10CM_b(2022))
                dfs.append(__ICD10CM_b(2023))
                dfs.append(__ICD10CM_b(2024))
                dfs.append(__ICD10CM_b(2025))

                df = auto.pd.concat(dfs)
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10cmB = scope()

            def scope():
                names = icd10cmA.index.to_list()
                names.sort(key=len)

                df = []
                for name in names:
                    desc = icd10cmA.loc[name, 'desc']

                    if name in icd10cmB.index:
                        continue

                    prev_descs = []
                    __tested = []
                    for i in range(len(name)-1, 1, -1):
                        prev_name = name[:i]
                        __tested.append(prev_name)

                        if prev_name in icd10cmB.index:
                            prev_desc = icd10cmB.loc[prev_name, 'desc']
                            prev_descs.insert(0, prev_desc)
                            break

                        if len(prev_name) == 3:
                            prev_code = f'{prev_name[:3]}'
                        else:
                            prev_code = f'{prev_name[:3]}.{prev_name[3:]}'

                        if prev_name in icd10cmA.index:
                            prev_desc = icd10cmA.loc[prev_name, 'desc']
                            prev_descs.insert(0, f'{prev_desc} ({prev_code})')
                            continue

                    else:
                        if strict:
                            raise KeyError(f'{name!r}: {__tested!r}')

                    if len(prev_descs) == 0 and not strict:
                        continue

                    assert len(prev_descs) > 0, name
                    prev_desc = ' > '.join(prev_descs)

                    if len(name) == 3:
                        code = f'{name[:3]}'
                    else:
                        code = f'{name[:3]}.{name[3:]}'

                    desc = f'{prev_desc} > {desc} ({code})'

                    df.append((name, desc))

                df = auto.pd.DataFrame(
                    df,
                    columns=[
                        'dx10',
                        'desc',
                    ],
                )
                df.set_index([
                    'dx10',
                ], inplace=True)
                df.sort_index(inplace=True)

                return df
            icd10cmC = scope()

            def scope():
                df = auto.pd.concat([
                    icd10cmA,
                    icd10cmB,
                    icd10cmC,
                ])
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10cm = scope()

            def scope():
                df = []

                prevs = {}
                for dx10 in icd10cm.index:
                    desc = icd10cm.loc[dx10, 'desc']

                    for n in range(len(dx10)-1, 0, -1):
                        prev = dx10[:n]

                        if prev in icd10cm.index:
                            continue

                        prevs.setdefault(prev, set()).add(desc)

                for prev, descs in prevs.items():
                    desc = auto.os.path.commonprefix(sorted(descs))
                    desc = desc[:desc.rfind(' > ')]

                    df.append((prev, desc))

                df = auto.pd.DataFrame(
                    df,
                    columns=[
                        'dx10',
                        'desc',
                    ],
                )
                df.set_index([
                    'dx10',
                ], inplace=True)
                df.sort_index(inplace=True)

                return df
            icd10cmD = scope()

            def scope():
                df = auto.pd.concat([
                    icd10cm,
                    icd10cmD,
                ])
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10cm = scope()

            def scope():
                dx10s = set(icd10cm.index.to_list())
                assert len(dx10s) == len(icd10cm)

                prevs = auto.pd.Series(
                    None,
                    index=icd10cm.index,
                    dtype=str,
                )

                for dx10 in dx10s:
                    if len(dx10) == 1:
                        prevs[dx10] = ''
                        continue

                    for n in range(len(dx10)-1, 0, -1):
                        prev_name = dx10[:n]

                        if prev_name in dx10s:
                            prevs[dx10] = prev_name
                            break

                    else:
                        raise ValueError(f'No prev found for {dx10!r}')

                return prevs
            icd10cm['prev'] = scope()

            def scope():
                nexts = auto.pd.Series(
                    [set() for _ in icd10cm.index],
                    index=icd10cm.index,
                )

                for dx10 in icd10cm.index:
                    prev = icd10cm.loc[dx10, 'prev']

                    assert prev is not None, dx10
                    if prev == '':
                        continue

                    nexts[prev].add(dx10)

                for dx10, next in nexts.items():
                    assert not any(' ' in s for s in next), (dx10, next)

                    nexts[dx10] = ' '.join(sorted(next))

                return nexts
            icd10cm['nexts'] = scope()

            def scope():
                disps = auto.pd.Series(
                    None,
                    index=icd10cm.index,
                    dtype=str,
                )

                for dx10 in icd10cm.index:
                    if len(dx10) <= 3:
                        disp = dx10
                    else:
                        disp = dx10[:3] + '.' + dx10[3:]

                    disps[dx10] = disp

                return disps
            icd10cm['disp'] = scope()

            with tmp_path.open('w') as f:
                icd10cm.to_csv(
                    f,
                    index=True,
                    header=True,
                    quoting=auto.csv.QUOTE_NONNUMERIC,
                )

        assert tmp_path.exists()
        tmp_path.rename(cache_path)
    assert cache_path.exists()

    with cache_path.open('r') as f:
        icd10cm = auto.pd.read_csv(
            f,
            dtype=str,
            na_filter=False,
            quoting=auto.csv.QUOTE_NONNUMERIC,
        )

    icd10cm.set_index([
        'dx10',
    ], inplace=True)
    icd10cm.sort_index(inplace=True)

    return icd10cm


#@title ICD10PCS
def __ICD10PCS(
    year: int,
    *,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    tmp_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_name: str = '__ICD10PCS.tmp',

    cache_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_name: str | auto.typing.Literal[...] = ...,
    cache_names: dict[int, str] = {
        2025: '_ICD10PCS2025.csv',
        2024: '_ICD10PCS2024.csv',
        2023: '_ICD10PCS2023.csv',
        2022: '_ICD10PCS2022.csv',
        2021: '_ICD10PCS2021.csv',
        2020: '_ICD10PCS2020.csv',
        2019: '_ICD10PCS2019.csv',
        2018: '_ICD10PCS2018.csv',
        2017: '_ICD10PCS2017.csv',
    },
    cache_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    zip_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    zip_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    zip_name: str | auto.typing.Literal[...] = ...,
    zip_names: dict[int, str] = {
        2025: '__ICD10PCS2025.zip',
        2024: '__ICD10PCS2024.zip',
        2023: '__ICD10PCS2023.zip',
        2022: '__ICD10PCS2022.zip',
        2021: '__ICD10PCS2021.zip',
        2020: '__ICD10PCS2020.zip',
        2019: '__ICD10PCS2019.zip',
        2018: '__ICD10PCS2018.zip',
        2017: '__ICD10PCS2017.zip',
    },
    zip_href: str | auto.typing.Literal[...] = ...,
    zip_hrefs: dict[int, str] = {
        2025: 'https://www.cms.gov/files/zip/2025-icd-10-pcs-code-tables-and-index-updated-07/09/2024.zip',
        2024: 'https://www.cms.gov/files/zip/2024-icd-10-pcs-code-tables-and-index-updated-12/19/2023.zip',
        2023: 'https://www.cms.gov/files/zip/2023-icd-10-pcs-code-tables-and-index-updated-01/11/2023.zip',
        2022: 'https://www.cms.gov/files/zip/2022-icd-10-pcs-code-tables-and-index-updated-december-1-2021.zip',
        2021: 'https://www.cms.gov/files/zip/2021-icd-10-pcs-code-tables-and-index-updated-december-1-2020.zip',
        2020: 'https://www.cms.gov/medicare/coding/icd10/downloads/2020-icd-10-pcs-code-tables.zip',
        2019: 'https://www.cms.gov/medicare/coding/icd10/downloads/2019-icd-10-pcs-tables-and-index.zip',
        2018: 'https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-pcs-tables-and-index.zip',
        2017: 'https://www.cms.gov/medicare/coding/icd10/downloads/2017-pcs-code-tables.zip',
    },

    xml_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    xml_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    xml_name: str | auto.typing.Literal[...] = ...,
    xml_names: dict[int, str] = {
        2025: 'Zip File 2 2025 Code Tables and Index/icd10pcs_tables_2025.xml',
        2024: 'icd10pcs_tables_2024.xml',
        2023: 'icd10pcs_tables_2023.xml',
        2022: '2 2022 Code Tables and Index/icd10pcs_tables_2022.xml',
        2021: '2021 Code Tables and Index/icd10pcs_tables_2021.xml',
        2020: 'PCS_2020/icd10pcs_tables_2020.xml',
        2019: 'icd10pcs_tables_2019.xml',
        2018: 'icd10pcs_tables_2018.xml',
        2017: 'icd10pcs_tables_2017.xml',
    },
):
    if root is ...:
        root = config.datadir

    if tmp_path is ...:
        if tmp_root is ...:
            tmp_root = root
        tmp_path = tmp_root / tmp_name

    if cache_path is ...:
        if cache_root is ...:
            cache_root = root
        if cache_name is ...:
            cache_name = cache_names[year]
        cache_path = cache_root / cache_name

    if not cache_path.exists():
        if zip_path is ...:
            if zip_root is ...:
                zip_root = root
            if zip_name is ...:
                zip_name = zip_names[year]
            zip_path = zip_root / zip_name

        if not zip_path.exists():
            if zip_href is ...:
                zip_href = zip_hrefs[year]

            with auto.requests.request(
                'GET',
                zip_href,
                stream=True,
            ) as r:
                r.raise_for_status()

                with tmp_path.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            tmp_path.rename(zip_path)
        assert zip_path.exists(), zip_path

        if xml_path is ...:
            if xml_root is ...:
                xml_root = auto.zipfile.Path(zip_path)
            if xml_name is ...:
                xml_name = xml_names[year]
            xml_path = xml_root / xml_name

        if not xml_path.exists():
            with auto.zipfile.ZipFile(zip_path, 'r') as arc:
                raise ValueError(repr(arc.infolist()))

        with xml_path.open('r') as f:
            soup = auto.bs4.BeautifulSoup(f, 'xml')

        soup ,= soup('ICD10PCS.tabular')

        it = soup('pcsTable', recursive=False)

        df = {}
        for table in it:
            it = table('axis', recursive=False)
            assert len(it) == 3, len(it)

            TABLE_POS = iter(auto.itertools.count(1))
            table_data: dict[int, list] = {
                pos: []
                for pos in range(1, 3+1)
            }

            for table_axis in it:
                table_pos = next(TABLE_POS)
                assert table_axis['pos'] == str(table_pos), (table_axis['pos'], table_pos)

                assert table_axis['values'] == '1', table_axis['values']

                table_title ,= table_axis('title', recursive=False)
                table_title_text = table_title.text
                table_label ,= table_axis('label', recursive=False)

                label_code = table_label['code']
                label_text = table_label.text

                table_data[table_pos].append(auto.types.SimpleNamespace(
                    text=f'{table_title_text}: {label_text}',
                    code=label_code,
                ))

            table_npos = next(TABLE_POS)
            assert table_npos == 3+1, table_npos

            assert all(len(v) > 0 for v in table_data.values()), table_data

            it = table('pcsRow', recursive=False)
            assert len(it) > 0, len(it)

            for row in it:
                it = row('axis', recursive=False)
                assert len(it) == 4, len(it)

                ROW_POS = iter(auto.itertools.count(table_npos))
                row_data: dict[int, list] = {
                    pos: []
                    for pos in range(4, 7+1)
                }

                for row_axis in it:
                    row_pos = next(ROW_POS)
                    assert row_axis['pos'] == str(row_pos), (row_axis['pos'], row_pos)

                    row_title ,= row_axis('title', recursive=False)
                    row_title_text = row_title.text

                    it = row_axis('label', recursive=False)
                    assert len(it) > 0, len(it)

                    for row_label in it:
                        row_code = row_label['code']
                        row_text = row_label.text

                        row_data[row_pos].append(auto.types.SimpleNamespace(
                            text=f'{row_title_text}: {row_text}',
                            code=row_code,
                        ))

                row_npos = next(ROW_POS)
                assert row_npos == 7+1, row_npos

                for n in range(1, 7+1):
                    it = auto.itertools.product(*[
                        (
                            table_data[i]
                        ) if 1 <= i <= 3 else (
                            row_data[i]
                        )
                        for i in range(1, n+1)
                    ])

                    for parts in it:
                        code = ''.join(part.code for part in parts)
                        text = '; '.join(part.text for part in parts)

                        df[code] = text

        df = auto.pd.DataFrame(
            df.items(),
            columns=[
                'pd10',
                'desc',
            ],
        )
        df.set_index([
            'pd10',
        ], inplace=True)
        df.sort_index(inplace=True)

        with tmp_path.open('w') as f:
            df.to_csv(
                f,
                index=True,
                header=True,
                quoting=auto.csv.QUOTE_NONNUMERIC,
            )
        tmp_path.rename(cache_path)
    assert cache_path.exists()

    with cache_path.open('r') as f:
        df = auto.pd.read_csv(
            f,
            dtype=str,
            na_filter=False,
            quoting=auto.csv.QUOTE_NONNUMERIC,
        )

    df.set_index([
        'pd10',
    ], inplace=True)
    df.sort_index(inplace=True)

    return df


@auto.functools.cache
def ICD10PCS(
    *,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,

    cache_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_name: str = 'ICD10PCS.csv',
    cache_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    cache_href: str | None = (
        None  # recompute
        # 'https://accona.eecs.utk.edu/ICD10PCS.csv'  # bootstrap
    ),

    tmp_root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    tmp_name: str = '__ICD10PCS.tmp',
):
    if root is ...:
        root = config.datadir

    if cache_path is ...:
        if cache_root is ...:
            cache_root = root
        cache_path = cache_root / cache_name

    if not cache_path.exists():
        if tmp_path is ...:
            if tmp_root is ...:
                tmp_root = root
            tmp_path = tmp_root / tmp_name

        if cache_href is not None:
            with auto.requests.request(
                'GET',
                cache_href,
                stream=True,
            ) as r:
                r.raise_for_status()
                with tmp_path.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        else:
            def scope():
                dfs = []
                dfs.append(__ICD10PCS(2017))
                dfs.append(__ICD10PCS(2018))
                dfs.append(__ICD10PCS(2019))
                dfs.append(__ICD10PCS(2020))
                dfs.append(__ICD10PCS(2021))
                dfs.append(__ICD10PCS(2022))
                dfs.append(__ICD10PCS(2023))
                dfs.append(__ICD10PCS(2024))
                dfs.append(__ICD10PCS(2025))

                df = auto.pd.concat(dfs)
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10pcs = scope()

            def scope():
                prevs = {}
                for pd10 in icd10pcs.index:
                    desc = icd10pcs.loc[pd10, 'desc']

                    for n in range(len(pd10)-1, 0, -1):
                        prev = pd10[:n]
                        if prev in icd10pcs.index:
                            continue

                        prevs.setdefault(prev, set()).add(desc)

                df = []
                for prev, descs in prevs.items():
                    desc = auto.os.path.commonprefix(sorted(descs))
                    desc = desc[:desc.rfind('; ')]

                    df.append((prev, desc))

                df = auto.pd.DataFrame(
                    df,
                    columns=[
                        'pd10',
                        'desc',
                    ],
                )
                df.set_index([
                    'pd10',
                ], inplace=True)
                df.sort_index(inplace=True)

                return df
            icd10pcsA = scope()

            def scope():
                df = auto.pd.concat([
                    icd10pcs,
                    icd10pcsA,
                ])
                df = df[~df.index.duplicated(keep='last')]
                df.sort_index(inplace=True)

                return df
            icd10pcs = scope()

            def scope():
                prevs = auto.pd.Series(
                    None,
                    index=icd10pcs.index,
                    dtype=str,
                )

                for pd10 in icd10pcs.index:
                    if len(pd10) == 1:
                        prevs[pd10] = ''
                        continue

                    __tested = []
                    for n in range(len(pd10)-1, 0, -1):
                        short = pd10[:n]

                        __tested.append(short)
                        if short in icd10pcs.index:
                            prevs[pd10] = short
                            break

                    else:
                        raise ValueError(f'No previous found for {pd10!r}: ({__tested!r})')

                return prevs
            icd10pcs['prev'] = scope()

            def scope():
                nexts = auto.pd.Series(
                    [set() for _ in icd10pcs.index],
                    index=icd10pcs.index,
                )

                for pd10, prev in icd10pcs['prev'].items():
                    if prev == '':
                        continue

                    nexts[prev].add(pd10)

                for pd10, nexts_ in nexts.items():
                    nexts[pd10] = ' '.join(nexts_)

                return nexts
            icd10pcs['nexts'] = scope()

            def scope():
                disps = auto.pd.Series(
                    None,
                    index=icd10pcs.index,
                    dtype=str,
                )

                for pd10 in icd10pcs.index:
                    disp = pd10

                    disps[pd10] = disp

                return disps
            icd10pcs['disp'] = scope()

            with tmp_path.open('w') as f:
                icd10pcs.to_csv(
                    f,
                    index=True,
                    header=True,
                    quoting=auto.csv.QUOTE_NONNUMERIC,
                )

        assert tmp_path.exists()
        tmp_path.rename(cache_path)
    assert cache_path.exists()

    with cache_path.open('r') as f:
        icd10pcs = auto.pd.read_csv(
            f,
            dtype=str,
            na_filter=False,
            quoting=auto.csv.QUOTE_NONNUMERIC,
        )

    icd10pcs.set_index([
        'pd10',
    ], inplace=True)
    icd10pcs.sort_index(inplace=True)

    return icd10pcs


#@title Score
@auto.functools.cache
def Matrix(
    *,
    what: auto.typing.Literal['Positive', 'Negative'],
    pred: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4'],
    prod: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4'],
    
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
) -> auto.pd.DataFrame:
    assert pred in ['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4'], pred
    assert prod in ['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4'], prod

    if root is ...:
        root = config.datadir

    if pred.startswith('DX') and prod.startswith('DX'):
        a, b = sorted([pred, prod])
    elif pred.startswith('PD') and prod.startswith('PD'):
        a, b = sorted([pred, prod])
    elif pred.startswith('DX') and prod.startswith('PD'):
        a, b = prod, pred
    elif pred.startswith('PD') and prod.startswith('DX'):
        a, b = pred, prod

    path = root / f'{what}{a}-{b}.npz'
    assert path.exists(), path
    assert path.stat().st_size > 0, path
    with path.open('rb') as f:
        npz = auto.np.load(f)
        arr = npz['arr']
        rowids = npz['rowids'].tolist()
        colids = npz['colids'].tolist()
    
    if a == prod:
        assert b == pred
        arr = arr.T
        rowids, colids = colids, rowids
    
    df = auto.pd.DataFrame(
        arr,
        index=rowids,
        columns=colids,
        # dtype='f4',
    )

    # df /= 1e-9 + df.sum(axis=1)

    return df


def Score(
    *,
    what: auto.typing.Literal['Positive', 'Negative'],
    dxs: list[str],
    pds: list[str],
    top: int = 0,

    verbose: bool | int = False,
    icd10cm: auto.pd.DataFrame | None = None,
    icd10pcs: auto.pd.DataFrame | None = None,
):
    verbose = int(verbose)
    
    Item = auto.typing.NamedTuple('Item', [
        ('kind', auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3', 'PD4']),
        ('name', str),
        ('full', str),
    ])

    It = auto.typing.NamedTuple('It', [
        ('kind', auto.typing.Literal['dx-dx', 'pd-pd', 'dx-pd']),
        ('pred', Item),
        ('prod', Item),
    ])

    itA = [
        It('dx-dx', Item(f'DX{an}', adx[:an], adx), Item(f'DX{bn}', bdx[:bn], bdx))
        for adx in dxs
        for bdx in dxs
        # if adx != bdx
        for an in range(1, 1+min(3, len(adx)))
        for bn in range(1, 1+min(3, len(bdx)))
    ]

    itB = [
        It('pd-pd', Item(f'PD{an}', apd[:an], apd), Item(f'PD{bn}', bpd[:bn], bpd))
        for apd in pds
        for bpd in pds
        # if apd != bpd
        for an in range(1, 1+min(4, len(apd)))
        for bn in range(1, 1+min(4, len(bpd)))
    ]

    itC = [
        It('dx-pd', Item(f'DX{an}', adx[:an], adx), Item(f'PD{bn}', bpd[:bn], bpd))
        for adx in dxs
        for bpd in pds
        for an in range(1, 1+min(3, len(adx)))
        for bn in range(1, 1+min(4, len(bpd)))
    ]
    
    # itD = [
    #     It('pd-dx', Item(f'PD{an}', apd[:an], apd), Item(f'DX{bn}', bdx[:bn], bdx))
    #     for apd in pds
    #     for bdx in dxs
    #     for an in range(1, 1+min(4, len(apd)))
    #     for bn in range(1, 1+min(3, len(bdx)))
    # ]

    it = itA + itB + itC
    it = sorted(it)  # optimize data loading
    if verbose >= 1:
        it = auto.tqdm.auto.tqdm(it, total=len(it))

    totals = {
        'dx-dx': auto.pd.DataFrame(
            0,
            index=dxs,
            columns=dxs,
            dtype=float,
        ),
        'pd-pd': auto.pd.DataFrame(
            0,
            index=pds,
            columns=pds,
            dtype=float,
        ),
        'dx-pd': auto.pd.DataFrame(
            0,
            index=dxs,
            columns=pds,
            dtype=float,
        ),
        'pd-dx': auto.pd.DataFrame(
            0,
            index=pds,
            columns=dxs,
            dtype=float,
        ),
    }
    counts = {
        'dx-dx': 0,
        'pd-pd': 0,
        'dx-pd': 0,
        'pd-dx': 0,
    }

    for it in it:
        df = Matrix(what=what, pred=it.pred.kind, prod=it.prod.kind)

        # For each predictor, we want to create vectors vA and vB.
        # vA is the probabilities of all products, given the predictor. vB is +1
        # where the product is present in our person, and -1 where it isn't.
        #
        # Iteratively, we can imagine doing this one at a time: for an item iA
        # and an item iB, we add up the probability of cooccurrence of iA and iB,
        # which we can call X, and subtract all the other probabilities, Y.
        #
        # There is a trick. Because the probabilities add to 1, then Y = 1 - X.
        # So instead of doing "+= X - Y" then it's the same as "+= X - (1 - X)"
        # or just "+= 2X - 1"

        probs = df.loc[it.pred.name]
        prob = (
            probs[it.prod.name] / (1e-9 + probs.sum())
            # probs[it.prod.name]
        )

        value = (
            # 2 * prob - 1
            prob
            # prob / len(it.pred.full) / len(it.prod.full)
        )

        totals[it.kind].loc[it.pred.full, it.prod.full] += value
        counts[it.kind] += 1

    total = 0
    for kind in totals:
        totals[kind] /= (
            counts[kind]
            # auto.np.sqrt(counts[kind])
        )

        total += totals[kind].sum().sum()
    
    if top == 0:
        pass

    elif top == 1:
        Best = auto.typing.NamedTuple('Best', [
            ('kind', auto.typing.Literal['dx-dx', 'pd-pd', 'dx-pd']),
            ('pred', auto.typing.Literal['DX', 'PD']),
            ('prod', auto.typing.Literal['DX', 'PD']),
            ('value', float),
            ('pred_name', str),
            ('prod_name', str),
        ])
        best = None
        
        def scope(kind, PRED, PROD):
            nonlocal best
            for pred, row in totals[kind].iterrows():
                for prod, value in row.items():
                    if best is None or value > best.value:
                        best = Best(kind, PRED, PROD, value, pred, prod)
        scope('dx-dx', 'DX', 'DX')
        scope('pd-pd', 'PD', 'PD')
        scope('dx-pd', 'DX', 'PD')
        
        foo = { 'DX': set(), 'PD': set() }
        foo[best.pred].add(best.pred_name)
        foo[best.prod].add(best.prod_name)

        dxs = sorted(foo['DX'])
        pds = sorted(foo['PD'])

    else:
        raise ValueError(f'Invalid top: {top!r}')

    if verbose >= 2:
        assert icd10cm is not None
        assert icd10pcs is not None
        print(f'Total: {total:.3f}')
        print(f'  DX:')
        for dx in dxs:
            print(f'    {dx}: {auto.textwrap.fill(icd10cm.loc[dx, "desc"], initial_indent="", subsequent_indent="      ")}')
        print(f'  PD:')
        for pd in pds:
            print(f'    {pd}: {auto.textwrap.fill(icd10pcs.loc[pd, "desc"], initial_indent="", subsequent_indent="      ")}')
        display(totals['dx-dx'])
        display(totals['pd-pd'])
        display(totals['dx-pd'])
    
    if top > 0:
        return total, dxs, pds
    else:
        return total


def SQLQuery(s: str, /, **kwargs):
    environment = auto.jinja2.Environment()
    environment.filters['tosqlref'] = lambda x: '"' + str(x).replace('"', '""') + '"'
    environment.filters['tosqlstr'] = lambda x: "'" + str(x).replace("'", "''") + "'"
    environment.globals['auto'] = auto

    template = environment.from_string(s)

    return template.render(**kwargs)


def SQLQuery_verbose(s: str, /, **kwargs):
    s = SQLQuery(s, **kwargs)

    with auto.mediocreatbest.Textarea():
        print(s)

    return s

SQLQuery.verbose = SQLQuery_verbose


def starstarmap(function, iterable):
    for kwargs in iterable:
        yield function(**kwargs)


__UnderpaymentModel = auto.typing.NamedTuple('Model', [
    ('N', auto.pd.DataFrame),
    ('μ', auto.pd.DataFrame),
    ('σ', auto.pd.DataFrame),
])

@auto.functools.cache
def UnderpaymentModel(
    *,
    path: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    root: auto.pathlib.Path | auto.typing.Literal[...] = ...,
    name: str | auto.typing.Literal[...] = ...,

    kind: auto.typing.Literal['mult', 'cost'] | None = None,
    pred: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3'] | None = None,
    prod: auto.typing.Literal['DX1', 'DX2', 'DX3', 'PD1', 'PD2', 'PD3'] | None = None,
) -> __UnderpaymentModel:
    if path is ...:
        if root is ...:
            root = config.datadir
        if name is ...:
            assert kind is not None
            assert pred is not None
            assert prod is not None
            name = f'Underpayment.{kind}.{pred}-{prod}.zip'
        path = root / name

    root = auto.zipfile.Path(path)

    with (root / 'N.feather').open('rb') as f:
        N = auto.pd.read_feather(f)
    
    with (root / 'μ.feather').open('rb') as f:
        μ = auto.pd.read_feather(f)
    
    with (root / 'σ.feather').open('rb') as f:
        σ = auto.pd.read_feather(f)

    with auto.warnings.catch_warnings():
        auto.warnings.simplefilter('ignore', FutureWarning)

        N = N.replace({ None: auto.np.nan })
        μ = μ.replace({ None: auto.np.nan })
        σ = σ.replace({ None: auto.np.nan })

    return __UnderpaymentModel(
        N = N,
        μ = μ,
        σ = σ,
    )


__UnderpaymentItemItemItem = auto.typing.TypedDict('UnderpaymentItem', {
    'avg': float,
    'std': float,
})
__UnderpaymentItemItem = auto.typing.TypedDict('UnderpaymentItem', {
    'tight': __UnderpaymentItemItemItem,
    'loose': __UnderpaymentItemItemItem,
})
__UnderpaymentItem = auto.typing.TypedDict('UnderpaymentItem', {
    'lo': __UnderpaymentItemItem,
    'hi': __UnderpaymentItemItem,
})
__Underpayment = auto.typing.TypedDict('Underpayment', {
    'cost': __UnderpaymentItem,
    'mult': __UnderpaymentItem,
})

def Underpayment(
    *,
    icd10cm: auto.pd.DataFrame,
    icd10pcs: auto.pd.DataFrame,

    dxs: list[str],
    pds: list[str],
    ndx: auto.typing.Literal[1, 2, 3],
    npd: auto.typing.Literal[1, 2, 3],
) -> __Underpayment:
    icd10cm = icd10cm[icd10cm.index.str.len() == ndx]
    icd10pcs = icd10pcs[icd10pcs.index.str.len() == npd]

    mult = UnderpaymentModel(
        kind = 'mult',
        pred = f'DX{ndx}',
        prod = f'PD{npd}',
    )
    cost = UnderpaymentModel(
        kind = 'cost',
        pred = f'DX{ndx}',
        prod = f'PD{npd}',
    )

    dxs = [dx[:ndx] for dx in dxs if dx[:ndx] in icd10cm.index]
    pds = [pd[:npd] for pd in pds if pd[:npd] in icd10pcs.index]

    cµs = []
    cσs = []
    mµs = []
    mσs = []
    for dx, pd in auto.itertools.product(dxs, pds):
        cµ = cost.μ.loc[dx, pd]
        cσ = cost.σ.loc[dx, pd]
        mµ = mult.μ.loc[dx, pd]
        mσ = mult.σ.loc[dx, pd]

        if auto.pd.isna(cµ): print('a'); continue
        if auto.pd.isna(cσ): print('b'); continue
        if auto.pd.isna(mµ): print('c'); continue
        if auto.pd.isna(mσ): print('d'); continue

        cµs.append(cµ)
        cσs.append(cσ)
        mµs.append(mµ)
        mσs.append(mσ)

    # /auto.pprint.pp cµs
    # /auto.pprint.pp cσs
    # /auto.pprint.pp mµs
    # /auto.pprint.pp mσs

    cµlo = auto.np.min(cµs)
    cµhi = auto.np.max(cµs)
    mµlo = auto.np.min(mµs)
    mµhi = auto.np.max(mµs)

    cσlo = auto.np.min(cσs)
    cσhi = auto.np.max(cσs)
    mσlo = auto.np.min(mσs)
    mσhi = auto.np.max(mσs)

    return __Underpayment(
        cost = __UnderpaymentItem(
            lo = __UnderpaymentItemItem(
                tight = __UnderpaymentItemItemItem(
                    avg = cµlo,
                    std = cσlo,
                ),
                loose = __UnderpaymentItemItemItem(
                    avg = cµlo,
                    std = cσhi,
                ),
            ),
            hi = __UnderpaymentItemItem(
                tight = __UnderpaymentItemItemItem(
                    avg = cµhi,
                    std = cσlo,
                ),
                loose = __UnderpaymentItemItemItem(
                    avg = cµhi,
                    std = cσhi,
                ),
            ),
        ),
        mult = __UnderpaymentItem(
            lo = __UnderpaymentItemItem(
                tight = __UnderpaymentItemItemItem(
                    avg = mµlo,
                    std = mσlo,
                ),
                loose = __UnderpaymentItemItemItem(
                    avg = mµlo,
                    std = mσhi,
                ),
            ),
            hi = __UnderpaymentItemItem(
                tight = __UnderpaymentItemItemItem(
                    avg = mµhi,
                    std = mσlo,
                ),
                loose = __UnderpaymentItemItemItem(
                    avg = mµhi,
                    std = mσhi,
                ),
            ),
        ),
    )
