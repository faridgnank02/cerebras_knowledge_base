from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Row:
    source: str
    source_id: str
    document: str
    raw_content: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    embedding: np.ndarray | None = None


class Connector(Protocol):
    name: str

    def fetch(self, since: str | None) -> Iterator[Row]: ...

    def watermark(self) -> str | None: ...
