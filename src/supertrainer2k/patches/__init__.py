import importlib
import pkgutil

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module('.' + module_name, __name__)
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(module, attr_name)
