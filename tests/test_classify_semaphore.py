from __future__ import annotations

import asyncio

from formuloom.classify import get_llm_semaphore, reset_llm_semaphores


def test_semaphore_survives_sequential_event_loops() -> None:
    reset_llm_semaphores()

    async def acquire_once() -> None:
        sem = get_llm_semaphore(2)
        async with sem:
            pass

    asyncio.run(acquire_once())
    asyncio.run(acquire_once())


def test_same_loop_reuses_one_semaphore() -> None:
    reset_llm_semaphores()

    async def two_lookups() -> None:
        assert get_llm_semaphore(3) is get_llm_semaphore(3)
        assert get_llm_semaphore(3) is not get_llm_semaphore(4)

    asyncio.run(two_lookups())
