"""

"""

from __future__ import annotations
from ._auto import auto


config = auto.types.SimpleNamespace()

config.rootdir = auto.pathlib.Path(auto.os.environ['UPLOAD_ROOT_DIR'])
config.datadir = auto.pathlib.Path(
    auto.os.environ.get(
        'UPLOAD_DATA_DIR',
        config.rootdir / 'data',
    ),
)
config.codedir = auto.pathlib.Path(__file__).parent

config.encryption_key_name = 'UPLOAD_ENCRYPTION_KEY'
assert config.encryption_key_name in auto.os.environ, config.encryption_key_name

config.login = auto.types.SimpleNamespace()
config.login.username = auto.os.environ['UPLOAD_LOGIN_USERNAME']
config.login.password = auto.os.environ['UPLOAD_LOGIN_PASSWORD']
