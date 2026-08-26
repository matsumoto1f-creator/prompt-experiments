"""LLM providers.

Same shape as the other projects in this family: a protocol, a deterministic offline
implementation, and a real one behind a deferred import so the vendor SDK stays
optional and the demo needs no key.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Protocol

from prompt_experiments.models import PromptVersion

# Published per-1M-token rates, snapshotted 2026-06-24. Drives every cost figure, so
# treat as data that goes stale.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return input_tokens / 1e6 * rates[0] + output_tokens / 1e6 * rates[1]


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    errored: bool = False
    error: str = ""


class Provider(Protocol):
    name: str
    def complete(self, version: PromptVersion, user_input: str, unit_id: str) -> Completion: ...


class MockProvider:
    """Deterministic offline provider.

    A SIMULATION FIXTURE, not a model. Quality is drawn from a per-version true rate
    supplied by the caller, seeded on (version, unit) so the same unit always gets the
    same outcome. That determinism is what makes the demo reproducible and the tests
    meaningful: an effect the platform reports has to come from the configured ground
    truth, never from sampling noise in the fixture.
    """

    name = "mock"

    def __init__(self, true_rates: dict[int, float] | None = None,
                 latency_ms: dict[int, float] | None = None,
                 error_rates: dict[int, float] | None = None) -> None:
        self.true_rates = true_rates or {}
        self.latency_ms = latency_ms or {}
        self.error_rates = error_rates or {}

    @staticmethod
    def _unit_draw(*parts: str) -> float:
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def complete(self, version: PromptVersion, user_input: str, unit_id: str) -> Completion:
        started = time.perf_counter()
        rate = self.true_rates.get(version.version, 0.7)
        error_rate = self.error_rates.get(version.version, 0.0)

        if self._unit_draw("err", str(version.version), unit_id) < error_rate:
            return Completion("", 0, 0, 0.0, 0.0, errored=True, error="simulated upstream failure")

        good = self._unit_draw("q", str(version.version), unit_id) < rate
        text = "PASS" if good else "FAIL"

        in_tok = max(1, len(version.system) // 4 + len(user_input) // 4)
        out_tok = 40 + int(self._unit_draw("t", str(version.version), unit_id) * 60)
        base = self.latency_ms.get(version.version, 500.0)
        # Right-skewed, like real latency: a long tail rather than a symmetric spread.
        jitter = self._unit_draw("l", str(version.version), unit_id)
        latency = base * (0.6 + 2.4 * jitter ** 3)

        return Completion(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency + (time.perf_counter() - started) * 1000,
            cost_usd=price(version.model, in_tok, out_tok),
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[provider] ANTHROPIC_API_KEY unset; falling back to SDK credential resolution")
        self._client = anthropic.Anthropic()

    def complete(self, version: PromptVersion, user_input: str, unit_id: str) -> Completion:
        messages: list[dict] = []
        for shot in version.few_shot:
            messages.append({"role": "user", "content": shot.input})
            messages.append({"role": "assistant", "content": shot.output})
        messages.append({"role": "user", "content": user_input})

        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=version.model,
                max_tokens=version.max_tokens,
                system=version.system,
                messages=messages,
                output_config={"effort": version.effort},
            )
        except Exception as exc:  # noqa: BLE001 - an errored call is data, not a crash
            return Completion("", 0, 0, (time.perf_counter() - started) * 1000, 0.0,
                              errored=True, error=f"{type(exc).__name__}: {exc}")

        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=price(version.model, response.usage.input_tokens, response.usage.output_tokens),
        )


def get_provider(name: str, **kwargs) -> Provider:
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown provider {name!r} (known: mock, anthropic)")
