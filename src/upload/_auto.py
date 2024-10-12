"""

"""

from functools import cached_property

__all__ = [
    'auto',
]

class AutoImport:
    @cached_property
    def importlib(self):
        import importlib
        return importlib

    @cached_property
    def uvicorn(self):
        import uvicorn
        return uvicorn

    @cached_property
    def fastapi(self):
        import fastapi
        import fastapi.templating
        import fastapi.staticfiles
        import fastapi.middleware.cors
        return fastapi

    def __getattr__(auto, name: str):
        return auto.importlib.import_module(name)


auto = AutoImport()
