import argparse
import enum
import os
import re
import sys
from asyncio import create_subprocess_exec, gather, get_event_loop
from contextlib import ExitStack
from functools import total_ordering
from io import BytesIO
from pathlib import Path
from shutil import which
from subprocess import PIPE, CalledProcessError
from tempfile import TemporaryDirectory, gettempdir
from typing import TYPE_CHECKING, NamedTuple, NewType, TypeVar, cast
from urllib.request import urlopen
from zipfile import ZipFile

if sys.version_info < (3, 6, 1):
    raise RuntimeError("wenbed requires python>=3.6.1")


T = TypeVar("T")
Platform = NewType("Platform", str)
URI = NewType("URI", str)


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


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Final, Protocol

    class Namespace(Protocol):
        platform: Platform
        output: str
        verbose: int
        pip_argument: Sequence[str]


async def main() -> None:
    args = parse_args()

    await gather(
        *(build(v, a) for v, a in set(iter_platform(args.platform))),
        # return_exceptions=True,
    )


def parse_args() -> "Namespace":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", nargs="?", default=".")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("platform")
    parser.add_argument("pip-argument", nargs="+")

    return cast("Namespace", parser.parse_args())


def iter_platform(
    platform: Platform,
) -> "Iterator[tuple[Version, Architecture]]":
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


async def build(version: Version, architecture: Architecture) -> None:
    temp_root = await get_temp_path()
    cache = temp_root / f"wenbed/versions/{version}/{get_embed_name(version, architecture)}"
    if not cache.exists():
        embed_uri = get_embed_uri(version, architecture)
        with TemporaryDirectory(dir=temp_root) as temp:
            with ZipFile(BytesIO(download(embed_uri)), mode="r") as archive:
                archive.extractall(temp)

            await ensure_pip(Path(temp))

            os.makedirs(cache.parent, exist_ok=True)
            with ExitStack() as stack:
                enter = stack.enter_context
                archive = enter(ZipFile(enter(cache.open("wb")), mode="w"))
                for path in Path(temp).rglob("*"):
                    archive.write(path, path.relative_to(temp))

    with ExitStack() as stack:
        enter = stack.enter_context
        archive = enter(ZipFile(enter(cache.open("rb")), mode="r"))
        print(archive)


async def get_temp_path() -> Path:
    chcp = which("chcp.com")
    cmd = which("cmd.exe")
    wslpath = which("wslpath")
    if chcp is not None and cmd is not None and wslpath is not None:
        return await get_windows_temp_path(chcp=chcp, cmd=cmd, wslpath=wslpath)
    else:
        return Path(gettempdir())


async def get_windows_temp_path(chcp: str, cmd: str, wslpath: str) -> Path:
    encoding = await get_windows_encoding(chcp)
    tmps = await gather(
        *(get_windows_environ(cmd, key, encoding) for key in ["TMPDIR", "TEMP", "TMP"])
    )
    for tmp in tmps:
        if tmp is None:
            continue
        path = await windows2posix(wslpath, tmp, encoding)
        if path.exists():
            return path
    raise RuntimeError("Can't find tempdir")


async def get_windows_encoding(chcp: str) -> str:
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


async def get_windows_environ(cmd: str, key: str, encoding: str) -> "str | None":
    expr = f"%{key}%"
    command = [cmd, "/C", "echo", expr]
    process = await create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)
    bstdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)

    stdout = bstdout.rstrip(b"\r\n").decode(encoding)
    return None if stdout == expr else stdout


async def windows2posix(wslpath: str, path: str, encoding: str) -> Path:
    command = [wslpath, "-u", path]
    process = await create_subprocess_exec(*command, stdout=PIPE)
    bstdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)

    stdout = bstdout.rstrip(b"\n").decode(encoding)
    return Path(stdout)


def get_embed_name(version: Version, architecture: Architecture) -> str:
    return f"python-{version}-embed-{architecture}.zip"


def get_embed_uri(version: Version, architecture: Architecture) -> URI:
    return URI(
        f"https://www.python.org/ftp/python/{version}/{get_embed_name(version, architecture)}"
    )


def download(uri: URI) -> bytes:
    with urlopen(uri) as response:
        return cast(bytes, response.read())


async def ensure_pip(directory: Path) -> None:
    (executable,) = map(str, directory.rglob("python.exe"))

    pip_check = await create_subprocess_exec(executable, "-m", "pip")
    await pip_check.wait()
    if pip_check.returncode:
        pattern = re.compile(r"^#\s*(import\s+site)", re.MULTILINE)
        for pth in directory.rglob("*._pth"):
            text = pth.read_text(encoding="utf-8")
            substituted = pattern.sub(r"\1", text)
            if text != substituted:
                pth.write_text(substituted, encoding="utf-8")

        command = [executable]
        python = await create_subprocess_exec(*command, stdin=PIPE)
        await python.communicate(download(URI("https://bootstrap.pypa.io/get-pip.py")))
        if python.returncode:
            raise CalledProcessError(python.returncode, command)

    command = [executable, "-m", "pip", "install", "--upgrade", "pip"]
    pip_install = await create_subprocess_exec(*command)
    await pip_install.wait()
    if pip_install.returncode:
        raise CalledProcessError(pip_install.returncode, command)


CODEPAGE2ENCODING: "Final[dict[int, str]]" = {
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
    949: "cp949",
    950: "cp950",
    1026: "cp1026",
    1140: "cp1140",
    1250: "cp1250",
    1251: "cp1251",
    1252: "cp1252",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    936: "gb2312",
    1200: "utf_16",
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
    loop = get_event_loop()
    loop.run_until_complete(main())
