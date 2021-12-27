import inspect
from pathlib import Path

_path_shaft_module = Path(inspect.getsourcefile(lambda: 0)).resolve().parent