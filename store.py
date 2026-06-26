import queue
from io import BytesIO
from typing import Any

# Files dict — written from main thread only, safe without a lock
_files: dict[str, Any] = {}

# Thread-safe queues: tools run in worker threads, app.py drains from main thread
_plot_q: queue.Queue = queue.Queue()
_code_q: queue.Queue = queue.Queue()


def put(name: str, df: Any) -> None:
    _files[name] = df


def get(name: str) -> Any:
    if name not in _files:
        raise KeyError(
            f"'{name}' not found. Available: {list(_files.keys()) or 'none uploaded yet'}."
        )
    return _files[name]


def all_files() -> dict[str, Any]:
    return _files


def queue_plot(title: str, buf: BytesIO) -> None:
    _plot_q.put((title, buf))


def pop_plots() -> list[tuple[str, BytesIO]]:
    plots = []
    while not _plot_q.empty():
        try:
            plots.append(_plot_q.get_nowait())
        except queue.Empty:
            break
    return plots


def queue_code(label: str, code: str) -> None:
    _code_q.put((label, code))


def pop_code() -> list[tuple[str, str]]:
    snippets = []
    while not _code_q.empty():
        try:
            snippets.append(_code_q.get_nowait())
        except queue.Empty:
            break
    return snippets


# DataFrames queued by tools — rendered as st.dataframe() in the UI
_df_q: queue.Queue = queue.Queue()


def queue_dataframe(label: str, df: Any) -> None:
    _df_q.put((label, df))


def pop_dataframes() -> list[tuple[str, "Any"]]:
    dfs = []
    while not _df_q.empty():
        try:
            dfs.append(_df_q.get_nowait())
        except queue.Empty:
            break
    return dfs
