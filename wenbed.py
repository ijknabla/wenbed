from __future__ import annotations

import sys

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("wenbed requires python>=3.7")

import argparse
import enum
import re
from asyncio import create_subprocess_exec, gather, run, set_event_loop
from asyncio.subprocess import Process
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from functools import total_ordering, wraps
from io import BytesIO
from pathlib import Path
from shutil import which
from subprocess import PIPE, CalledProcessError
from tempfile import gettempdir
from typing import TYPE_CHECKING, Any, NamedTuple, NewType, TypeVar, cast
from urllib.request import urlopen
from warnings import warn
from zipfile import ZipFile

if TYPE_CHECKING:
    from typing import Final, ParamSpec, Protocol


if TYPE_CHECKING:
    _P = ParamSpec("_P")
_T = TypeVar("_T")
Platform = NewType("Platform", str)
URI = NewType("URI", str)

if sys.platform == "win32":
    from asyncio import ProactorEventLoop

    set_event_loop(ProactorEventLoop())


def run_coroutine(f_co: Callable[_P, Coroutine[Any, Any, _T]]) -> Callable[_P, _T]:
    @wraps(f_co)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return run(f_co(*args, **kwargs))

    return wrapped


@run_coroutine
async def main() -> None:
    args = parse_args()

    results = await gather(
        *(
            setup_python_embed(Path(args.output), v, a, args.pip_argument)
            for v, a in set(iter_platform(args.platform))
        ),
        return_exceptions=True,
    )
    exceptions = [(i, exc) for i, exc in enumerate(results) if isinstance(exc, BaseException)]

    if exceptions:
        warning_lines = [
            f"{len(exceptions)} exception(s) raised!",
            *(f"[{i}/{len(exceptions)}] {type(exc).__name__}" for i, exc in exceptions),
            "Raise first exception!",
        ]
        warn(Warning("\n".join(warning_lines)))
        (_, exc), *_ = exceptions
        raise exc


# arguments & options

if TYPE_CHECKING:

    class Namespace(Protocol):
        platform: Platform
        output: str
        verbose: int
        pip_argument: Sequence[str]


def parse_args() -> "Namespace":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", nargs="?", default=".")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("platform")
    parser.add_argument("pip_argument", nargs="+")

    return cast("Namespace", parser.parse_args())


class Version(NamedTuple):
    major: int
    minor: int
    micro: int

    def __str__(self) -> str:
        return ".".join(map(str, self))


@total_ordering
class Architecture(enum.Enum):
    win32 = enum.auto()
    amd64 = enum.auto()
    arm64 = enum.auto()

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: "Architecture") -> bool:
        return self.value < other.value


def iter_platform(
    platform: Platform,
) -> Iterator[tuple[Version, Architecture]]:
    version_architecture = re.compile(
        rf"(\d+)\.(\d+)\.(\d+)\-({'|'.join(a.name for a in Architecture)})"
    )
    for word in platform.split(","):
        matched = version_architecture.fullmatch(word)
        if matched is None:
            raise ValueError(f"{word!r} does not match {version_architecture.pattern}")
        major = int(matched.group(1))
        minor = int(matched.group(2))
        micro = int(matched.group(3))
        architecture = Architecture[matched.group(4)]
        yield Version(major=major, minor=minor, micro=micro), architecture


# python package installation


async def setup_python_embed(
    root: Path, version: Version, architecture: Architecture, pip_argument: Sequence[str]
) -> None:
    directory = root / get_embed_name(version, architecture)

    try:
        python = get_python_embed_executable(directory)
    except Exception:
        embed_uri = get_embed_uri(version, architecture)
        with ZipFile(BytesIO(download(embed_uri)), mode="r") as archive:
            archive.extractall(directory)

        python = get_python_embed_executable(directory)

    await run_subprocess(python, "-V")

    if await run_subprocess(python, "-m", "pip", check=False) != 0:
        await get_pip(python)

    await run_subprocess(python, "-m", "pip", "install", "--upgrade", "pip")
    await run_subprocess(python, "-m", "pip", *pip_argument)


def get_embed_name(version: Version, architecture: Architecture) -> str:
    return f"python-{version}-embed-{architecture}"


def get_embed_uri(version: Version, architecture: Architecture) -> URI:
    return URI(
        f"https://www.python.org/ftp/python/{version}/{get_embed_name(version, architecture)}.zip"
    )


def download(uri: URI) -> bytes:
    with urlopen(uri) as response:
        return cast(bytes, response.read())


def get_python_embed_executable(directory: Path) -> str:
    (python,) = directory.glob("python.exe")
    return str(python)


async def get_pip(python: str) -> None:
    pattern = re.compile(r"^#\s*(import\s+site)", re.MULTILINE)
    for pth in Path(python).parent.rglob("*._pth"):
        text = pth.read_text(encoding="utf-8")
        substituted = pattern.sub(r"\1", text)
        if text != substituted:
            pth.write_text(substituted, encoding="utf-8")

    async with aopen_subprocess(python, "-", stdin=PIPE) as process:
        await process.communicate(download(URI("https://bootstrap.pypa.io/get-pip.py")))


async def run_subprocess(program: str, *args: str, check: bool = True) -> int:
    async with AsyncExitStack() as stack:
        if not check:
            stack.enter_context(suppress(CalledProcessError))
        process = await stack.enter_async_context(aopen_subprocess(program, *args))
        await process.wait()

    assert process.returncode is not None
    return process.returncode


@asynccontextmanager
@wraps(create_subprocess_exec)
async def aopen_subprocess(
    program: str, *args: str, **kwargs: Any
) -> AsyncGenerator[Process, None]:
    process = await create_subprocess_exec(program, *args, **kwargs)
    try:
        yield process
    except Exception:
        returncode = process.returncode
        if returncode is None:
            process.terminate()
            await process.wait()
            raise
        elif returncode != 0:
            raise CalledProcessError(returncode, [program, *args])

    returncode = process.returncode
    if returncode is None:
        process.terminate()
        await process.wait()
    elif returncode != 0:
        raise CalledProcessError(returncode, [program, *args])


# unused utilities for windows & wsl


async def _get_temp_path() -> Path:
    chcp = which("chcp.com")
    cmd = which("cmd.exe")
    wslpath = which("wslpath")
    if chcp is not None and cmd is not None and wslpath is not None:
        return await _get_windows_temp_path(chcp=chcp, cmd=cmd, wslpath=wslpath)
    else:
        return Path(gettempdir())


async def _get_windows_temp_path(chcp: str, cmd: str, wslpath: str) -> Path:
    encoding = await _get_windows_encoding(chcp)
    tmps = await gather(
        *(_get_windows_environ(cmd, key, encoding) for key in ["TMPDIR", "TEMP", "TMP"])
    )
    for tmp in tmps:
        if tmp is None:
            continue
        path = await _windows2posix(wslpath, tmp, encoding)
        if path.exists():
            return path
    raise RuntimeError("Can't find tempdir")


async def _get_windows_encoding(chcp: str) -> str:
    command = [chcp]
    process = await create_subprocess_exec(*command, stdout=PIPE)
    bstdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)

    matched = re.search(rb"\d+", bstdout)
    if matched is None:
        raise ValueError(bstdout)
    codepage = int(matched.group(0))
    encoding = CODEPAGE2ENCODING[codepage]
    bstdout.decode(encoding)
    return encoding


async def _get_windows_environ(cmd: str, key: str, encoding: str) -> "str | None":
    expr = f"%{key}%"
    command = [cmd, "/C", "echo", expr]
    process = await create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)
    bstdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)

    stdout = bstdout.rstrip(b"\r\n").decode(encoding)
    return None if stdout == expr else stdout


async def _windows2posix(wslpath: str, path: str, encoding: str) -> Path:
    command = [wslpath, "-u", path]
    process = await create_subprocess_exec(*command, stdout=PIPE)
    bstdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)

    stdout = bstdout.rstrip(b"\n").decode(encoding)
    return Path(stdout)


CODEPAGE2ENCODING: Final[dict[int, str]] = {
    37: "cp037",
    437: "cp437",
    500: "cp500",
    720: "cp720",
    737: "cp737",
    775: "cp775",
    850: "cp850",
    852: "cp852",
    855: "cp855",
    857: "cp857",
    858: "cp858",
    860: "cp860",
    861: "cp861",
    862: "cp862",
    863: "cp863",
    864: "cp864",
    865: "cp865",
    866: "cp866",
    869: "cp869",
    874: "cp874",
    875: "cp875",
    932: "cp932",
    936: "gb2312",
    949: "cp949",
    950: "cp950",
    1026: "cp1026",
    1140: "cp1140",
    1200: "utf_16",
    1250: "cp1250",
    1251: "cp1251",
    1252: "cp1252",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    1361: "johab",
    12000: "utf_32",
    12001: "utf_32_be",
    20866: "koi8_r",
    20932: "euc_jp",
    21866: "koi8_u",
    28592: "iso8859_2",
    28593: "iso8859_3",
    28594: "iso8859_4",
    28595: "iso8859_5",
    28596: "iso8859_6",
    28597: "iso8859_7",
    28598: "iso8859_8",
    28599: "iso8859_9",
    28603: "iso8859_13",
    28605: "iso8859_15",
    50220: "iso2022_jp",
    50222: "iso2022_jp",
    50225: "iso2022_kr",
    51932: "euc_jp",
    51949: "euc_kr",
    54936: "gb18030",
    65000: "utf_7",
    65001: "utf_8",
}


if __name__ == "__main__":
    main()
