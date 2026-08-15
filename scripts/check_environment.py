from __future__ import annotations

import json

from cardiogb.utils.device import resolve_device


if __name__ == "__main__":
    print(json.dumps(resolve_device("auto").to_dict(), indent=2))

