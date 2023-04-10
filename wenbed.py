import argparse
import enum
import re
from functools import total_ordering
from subprocess import PIPE, run
from typing import TYPE_CHECKING, NamedTuple, NewType, cast

Platform = NewType("Platform", str)


class Version(NamedTuple):
    major: int
    minor: int
    micro: int


@total_ordering
class Architecture(enum.Enum):
    win32 = enum.auto()
    amd64 = enum.auto()
    arm64 = enum.auto()

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


def main() -> None:
    args = parse_args()
    sorted(set(iter_platform(args.platform)))


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
        r"(\d+)\.(\d+)\.(\d+)\-" rf"({'|'.join(a.name for a in Architecture)})"
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


def detect_windows_encoding(chcp: str) -> str:
    process = run([chcp], stdout=PIPE, check=True)
    matched = re.search(rb"\d+", process.stdout)
    if matched is None:
        raise ValueError(process.stdout)
    codepage = int(matched.group(0))
    encoding = CODEPAGE2ENCODING[codepage]
    process.stdout.decode(encoding)
    return encoding


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
    main()
