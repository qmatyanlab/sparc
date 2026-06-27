import os
from pathlib import Path
from typing import Any


class Calculator:
    def __init__(
        self,
        root_dir: str,
        task: str,
    ) -> None:
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path
        self.root_dir = str(root_path)
        self.task = task
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

    def calc(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
