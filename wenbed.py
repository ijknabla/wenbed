import argparse
import asyncio
import enum
import os
import re
import sys
from asyncio import create_subprocess_exec, gather, get_event_loop, set_event_loop
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterator, Sequence
from contextlib import ExitStack
from functools import total_ordering, wraps
from io import BytesIO
from pathlib import Path
from shutil import which
from subprocess import PIPE, CalledProcessError
from tempfile import TemporaryDirectory, gettempdir
from types import MethodType, TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    ContextManager,
    Generic,
    NamedTuple,
    NewType,
    Type,
    TypeVar,
    cast,
)
from urllib.request import urlopen
from zipfile import ZipFile

if TYPE_CHECKING:
    from typing import Final, ParamSpec, Protocol

if sys.version_info < (3, 6, 1):
    raise RuntimeError("wenbed requires python>=3.6.1")


if TYPE_CHECKING:
    _P = ParamSpec("_P")
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
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

    class Namespace(Protocol):
        platform: Platform
        output: str
        verbose: int
        pip_argument: Sequence[str]


def asynccontextmanager(
    func: "Callable[_P, AsyncGenerator[_T, Any]]",
) -> "Callable[_P, AsyncContextManager[_T]]":
    @wraps(func)
    def wrapped(*args: "_P.args", **kwargs: "_P.kwargs") -> "AsyncContextManager[_T]":
        return _AsyncGeneratorContextManager[_T](func(*args, **kwargs))

    return wrapped


class _AsyncGeneratorContextManager(Generic[_T]):
    """Helper for @asynccontextmanager decorator."""

    gen: "AsyncGenerator[_T, Any]"

    def __init__(self, gen: "AsyncGenerator[_T, Any]") -> None:
        self.gen = gen

    async def __aenter__(self) -> _T:
        try:
            return await self.gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None

    async def __aexit__(
        self,
        typ: "Type[BaseException] | None",
        value: "BaseException | None",
        traceback: "TracebackType | None",
    ) -> "bool | None":
        if typ is None:
            try:
                await self.gen.__anext__()
            except StopAsyncIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                # Need to force instantiation so we can reliably
                # tell if we get the same exception back
                value = typ()
            try:
                await self.gen.athrow(typ, value, traceback)
            except StopAsyncIteration as exc:
                # Suppress StopIteration *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed.
                return exc is not value
            except RuntimeError as exc:
                # Don't re-raise the passed in exception. (issue27122)
                if exc is value:
                    return False
                # Avoid suppressing if a Stop(Async)Iteration exception
                # was passed to athrow() and later wrapped into a RuntimeError
                # (see PEP 479 for sync generators; async generators also
                # have this behavior). But do this only if the exception wrapped
                # by the RuntimeError is actully Stop(Async)Iteration (see
                # issue29692).
                if (
                    isinstance(value, (StopIteration, StopAsyncIteration))
                    and exc.__cause__ is value
                ):
                    return False
                raise
            except BaseException as exc:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if exc is not value:
                    raise
                return False
            raise RuntimeError("generator didn't stop after athrow()")


class _BaseExitStack:
    """A base class for ExitStack and AsyncExitStack."""

    @staticmethod
    def _create_exit_wrapper(cm, cm_exit):  # type: ignore
        return MethodType(cm_exit, cm)

    @staticmethod
    def _create_cb_wrapper(callback, *args, **kwds):  # type: ignore
        def _exit_wrapper(exc_type, exc, tb):  # type: ignore
            callback(*args, **kwds)

        return _exit_wrapper

    def __init__(self) -> None:
        self._exit_callbacks = deque()  # type: ignore

    def pop_all(self):  # type: ignore
        """Preserve the context stack by transferring it to a new instance."""
        new_stack = type(self)()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        return new_stack

    def push(self, exit):  # type: ignore
        """Registers a callback with the standard __exit__ method signature.

        Can suppress exceptions the same way __exit__ method can.
        Also accepts any object with an __exit__ method (registering a call
        to the method instead of the object itself).
        """
        # We use an unbound method rather than a bound method to follow
        # the standard lookup behaviour for special methods.
        _cb_type = type(exit)

        try:
            exit_method = _cb_type.__exit__
        except AttributeError:
            # Not a context manager, so assume it's a callable.
            self._push_exit_callback(exit)  # type: ignore
        else:
            self._push_cm_exit(exit, exit_method)  # type: ignore
        return exit  # Allow use as a decorator.

    def enter_context(self, cm: ContextManager[_T_co]) -> _T_co:
        """Enters the supplied context manager.

        If successful, also pushes its __exit__ method as a callback and
        returns the result of the __enter__ method.
        """
        # We look up the special methods on the type to match the with
        # statement.
        _cm_type = type(cm)
        _exit = _cm_type.__exit__
        result: _T_co
        result = _cm_type.__enter__(cm)  # type: ignore
        self._push_cm_exit(cm, _exit)  # type: ignore
        return result

    def callback(self, callback, *args, **kwds):  # type: ignore
        """Registers an arbitrary callback and arguments.

        Cannot suppress exceptions.
        """
        _exit_wrapper = self._create_cb_wrapper(callback, *args, **kwds)  # type: ignore

        # We changed the signature, so using @wraps is not appropriate, but
        # setting __wrapped__ may still help with introspection.
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper)  # type: ignore
        return callback  # Allow use as a decorator

    def _push_cm_exit(self, cm, cm_exit):  # type: ignore
        """Helper to correctly register callbacks to __exit__ methods."""
        _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)  # type: ignore
        self._push_exit_callback(_exit_wrapper, True)  # type: ignore

    def _push_exit_callback(self, callback, is_sync=True):  # type: ignore
        self._exit_callbacks.append((is_sync, callback))


# Inspired by discussions on https://bugs.python.org/issue29302
class AsyncExitStack(_BaseExitStack):
    """Async context manager for dynamic management of a stack of exit
    callbacks.

    For example:
        async with AsyncExitStack() as stack:
            connections = [await stack.enter_async_context(get_connection())
                for i in range(5)]
            # All opened connections will automatically be released at the
            # end of the async with statement, even if attempts to open a
            # connection later in the list raise an exception.
    """

    @staticmethod
    def _create_async_exit_wrapper(cm, cm_exit):  # type: ignore
        return MethodType(cm_exit, cm)

    @staticmethod
    def _create_async_cb_wrapper(callback, *args, **kwds):  # type: ignore
        async def _exit_wrapper(exc_type, exc, tb):  # type: ignore
            await callback(*args, **kwds)

        return _exit_wrapper

    async def enter_async_context(self, cm: AsyncContextManager[_T_co]) -> _T_co:
        """Enters the supplied async context manager.

        If successful, also pushes its __aexit__ method as a callback and
        returns the result of the __aenter__ method.
        """
        _cm_type = type(cm)
        _exit = _cm_type.__aexit__
        result: _T_co
        result = await _cm_type.__aenter__(cm)  # type: ignore
        self._push_async_cm_exit(cm, _exit)  # type: ignore
        return result

    def push_async_exit(self, exit):  # type: ignore
        """Registers a coroutine function with the standard __aexit__ method
        signature.

        Can suppress exceptions the same way __aexit__ method can.
        Also accepts any object with an __aexit__ method (registering a call
        to the method instead of the object itself).
        """
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__aexit__
        except AttributeError:
            # Not an async context manager, so assume it's a coroutine function
            self._push_exit_callback(exit, False)  # type: ignore
        else:
            self._push_async_cm_exit(exit, exit_method)  # type: ignore
        return exit  # Allow use as a decorator

    def push_async_callback(self, callback, *args, **kwds):  # type: ignore
        """Registers an arbitrary coroutine function and arguments.

        Cannot suppress exceptions.
        """
        _exit_wrapper = self._create_async_cb_wrapper(callback, *args, **kwds)  # type: ignore

        # We changed the signature, so using @wraps is not appropriate, but
        # setting __wrapped__ may still help with introspection.
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper, False)  # type: ignore
        return callback  # Allow use as a decorator

    async def aclose(self) -> None:
        """Immediately unwind the context stack."""
        await self.__aexit__(None, None, None)

    def _push_async_cm_exit(self, cm, cm_exit):  # type: ignore
        """Helper to correctly register coroutine function to __aexit__
        method."""
        _exit_wrapper = self._create_async_exit_wrapper(cm, cm_exit)  # type: ignore
        self._push_exit_callback(_exit_wrapper, False)  # type: ignore

    async def __aenter__(self) -> "AsyncExitStack":
        return self

    async def __aexit__(
        self,
        typ: "Type[BaseException] | None",
        value: "BaseException | None",
        traceback: "TracebackType | None",
    ) -> "bool | None":
        exc_details = typ, value, traceback
        received_exc = exc_details[0] is not None

        # We manipulate the exception state so it behaves as though
        # we were actually nesting multiple with statements
        frame_exc = sys.exc_info()[1]

        def _fix_exception_context(new_exc, old_exc):  # type: ignore
            # Context may not be correct, so find the end of the chain
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    # Context is already set correctly (see issue 20317)
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            # Change the end of the chain to point to the exception
            # we expect it to reference
            new_exc.__context__ = old_exc

        # Callbacks are invoked in LIFO order to match the behaviour of
        # nested context managers
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            try:
                if is_sync:
                    cb_suppress = cb(*exc_details)
                else:
                    cb_suppress = await cb(*exc_details)

                if cb_suppress:
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:  # noqa: E722
                new_exc_details = sys.exc_info()
                # simulate the stack of exceptions by setting the context
                _fix_exception_context(new_exc_details[1], exc_details[1])  # type: ignore
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                # bare "raise exc_details[1]" replaces our carefully
                # set-up context
                fixed_ctx = exc_details[1].__context__  # type: ignore
                raise exc_details[1]  # type: ignore
            except BaseException:
                exc_details[1].__context__ = fixed_ctx  # type: ignore
                raise
        return received_exc and suppressed_exc


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
    if sys.platform == "win32":
        set_event_loop(asyncio.ProactorEventLoop())
    loop = get_event_loop()
    loop.run_until_complete(main())
