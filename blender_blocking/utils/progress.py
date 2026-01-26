"""Optional progress helpers with a custom unicode bar."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

_BAR_CHARS = "⣀⣄⣆⣇⣧⣶⣷⣿"
_DEFAULT_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
_ANSI_RESET = "\x1b[0m"
_ANSI_BG_BLACK = "\x1b[40m"
_BRIGHT_COLORS = [
    "\x1b[92m",  # bright green
    "\x1b[93m",  # bright yellow
    "\x1b[96m",  # bright cyan
    "\x1b[95m",  # bright magenta
    "\x1b[94m",  # bright blue
    "\x1b[91m",  # bright red
]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


@dataclass
class _DummyBar:
    total: Optional[int] = None

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    enabled: bool = True,
    mininterval: Optional[float] = None,
    miniters: Optional[int] = None,
) -> Iterable[T]:
    if not enabled or tqdm is None:
        return iterable
    color = random.choice(_BRIGHT_COLORS)
    bar_format = f"{_ANSI_BG_BLACK}{color}{_DEFAULT_BAR_FORMAT}{_ANSI_RESET}"
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        ascii=_BAR_CHARS,
        bar_format=bar_format,
        position=0,
        leave=True,
        dynamic_ncols=True,
        mininterval=mininterval,
        miniters=miniters,
    )


def progress_bar(
    total: int,
    *,
    desc: Optional[str] = None,
    enabled: bool = True,
    mininterval: Optional[float] = None,
    miniters: Optional[int] = None,
) -> _DummyBar:
    if not enabled or tqdm is None:
        return _DummyBar(total=total)
    color = random.choice(_BRIGHT_COLORS)
    bar_format = f"{_ANSI_BG_BLACK}{color}{_DEFAULT_BAR_FORMAT}{_ANSI_RESET}"
    return tqdm(
        total=total,
        desc=desc,
        ascii=_BAR_CHARS,
        bar_format=bar_format,
        position=0,
        leave=True,
        dynamic_ncols=True,
        mininterval=mininterval,
        miniters=miniters,
    )


def progress_print(*args: object, enabled: bool = True, **kwargs: object) -> None:
    """Print without disrupting an active tqdm bar."""
    if tqdm is not None and enabled:
        tqdm.write(" ".join(str(arg) for arg in args))
        return
    print(*args, **kwargs)
