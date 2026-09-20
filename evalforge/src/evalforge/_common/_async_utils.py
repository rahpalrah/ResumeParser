# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Helpers for exposing async internals through a synchronous public API."""

import asyncio
import concurrent.futures
from typing import Any, Awaitable, Callable, TypeVar

__all__ = ["run_allowing_running_loop", "async_wrap"]

T = TypeVar("T")


def run_allowing_running_loop(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine even when an event loop is already running.

    :func:`asyncio.run` raises if called from inside a running loop (notebooks,
    async web handlers). In that case the coroutine is executed on a worker
    thread with its own loop, so a synchronous ``__call__`` works everywhere.

    :param coro_factory: Zero-argument callable returning a fresh coroutine.
        A factory rather than a coroutine object, so the coroutine is only
        created on the loop that will await it.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


async def async_wrap(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Await a blocking callable on the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
