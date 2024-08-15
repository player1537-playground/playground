"""

"""

from __future__ import annotations
from ._auto import auto


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
