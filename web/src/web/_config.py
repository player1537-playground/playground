"""

"""

from __future__ import annotations
from ._auto import auto


config = auto.types.SimpleNamespace()

config.rootdir = auto.pathlib.Path(auto.os.environ['SCRIBE_ROOT_DIR'])
config.datadir = auto.pathlib.Path(
    auto.os.environ.get(
        'SCRIBE_DATA_DIR',
        config.rootdir / 'data',
    ),
)

config.llama = auto.types.SimpleNamespace()
config.llama.api_key = auto.os.environ['LLAMA_API_KEY']

config.nomic = auto.types.SimpleNamespace()
config.nomic.api_key = auto.os.environ['NOMIC_API_KEY']

config.vainl = auto.types.SimpleNamespace()
config.vainl.api_url = auto.os.environ['VAINL_API_URL']
config.vainl.api_key = auto.os.environ['VAINL_API_KEY']
