from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from inference_lab.metrics import BenchmarkReport, RequestMeasurement, aggregate_measurements
from inference_lab.prompts import PromptCase, load_prompt_suite


@dataclass(frozen=True)
class SGLangBenchmarkOptions:
    base_url: str
    model: str
    prompt_file: Path
    concurrency: list[int]
    max_tokens: int
    requests_per_concurrency: int
    warmup_requests: int
    timeout_s: float = 120.0
    temperature: float = 0.2
    stream: bool = True


async def run_sglang_benchmark(options: SGLangBenchmarkOptions) -> BenchmarkReport:
    prompts = load_prompt_suite(options.prompt_file)
    measurements: list[RequestMeasurement] = []

    async with httpx.AsyncClient(timeout=options.timeout_s) as client:
        for concurrency in options.concurrency:
            warmups = _cycle_prompts(prompts, options.warmup_requests)
            await _run_batch(client, options, warmups, concurrency=1, record=False)

            prompt_cases = _cycle_prompts(prompts, options.requests_per_concurrency)
            batch_measurements = await _run_batch(
                client,
                options,
                prompt_cases,
                concurrency=concurrency,
                record=True,
            )
            measurements.extend(batch_measurements)

    return BenchmarkReport(
        name="sglang",
        metadata={
            "base_url": options.base_url,
            "model": options.model,
            "max_tokens": options.max_tokens,
            "stream": options.stream,
        },
        measurements=measurements,
        summaries=aggregate_measurements(measurements),
    )


async def _run_batch(
    client: httpx.AsyncClient,
    options: SGLangBenchmarkOptions,
    prompts: list[PromptCase],
    *,
    concurrency: int,
    record: bool,
) -> list[RequestMeasurement]:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(prompt: PromptCase) -> RequestMeasurement:
        async with semaphore:
            return await _measure_completion(client, options, prompt, concurrency=concurrency)

    results = await asyncio.gather(*(worker(prompt) for prompt in prompts))
    return results if record else []


async def _measure_completion(
    client: httpx.AsyncClient,
    options: SGLangBenchmarkOptions,
    prompt: PromptCase,
    *,
    concurrency: int,
) -> RequestMeasurement:
    payload = {
        "model": options.model,
        "messages": prompt.messages,
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "stream": options.stream,
    }
    started = time.perf_counter()
    try:
        if options.stream:
            ttft_s, output_tokens = await _stream_completion(client, options.base_url, payload, started)
        else:
            ttft_s, output_tokens = await _completion(client, options.base_url, payload, started)
        latency_s = time.perf_counter() - started
        return RequestMeasurement(
            scenario=prompt.scenario,
            prompt_name=prompt.name,
            concurrency=concurrency,
            latency_s=latency_s,
            ttft_s=ttft_s,
            output_tokens=output_tokens,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark reports should capture backend failures.
        return RequestMeasurement(
            scenario=prompt.scenario,
            prompt_name=prompt.name,
            concurrency=concurrency,
            latency_s=time.perf_counter() - started,
            ttft_s=None,
            output_tokens=0,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    started: float,
) -> tuple[float | None, int]:
    ttft_s: float | None = None
    text_chunks = 0
    async with client.stream("POST", _chat_url(base_url), json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if raw == "[DONE]":
                break
            chunk = json.loads(raw)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if (content or reasoning) and ttft_s is None:
                ttft_s = time.perf_counter() - started
            if content:
                text_chunks += _estimate_tokens(str(content))
            if reasoning:
                text_chunks += _estimate_tokens(str(reasoning))
    return ttft_s, max(text_chunks, 1)


async def _completion(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    started: float,
) -> tuple[float, int]:
    response = await client.post(_chat_url(base_url), json=payload)
    response.raise_for_status()
    body = response.json()
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage_tokens = body.get("usage", {}).get("completion_tokens")
    output_tokens = int(usage_tokens) if usage_tokens is not None else _estimate_tokens(str(text))
    return time.perf_counter() - started, max(output_tokens, 1)


def _chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _cycle_prompts(prompts: list[PromptCase], count: int) -> list[PromptCase]:
    if count <= 0:
        return []
    return [prompts[index % len(prompts)] for index in range(count)]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) or len(text) // 4)
