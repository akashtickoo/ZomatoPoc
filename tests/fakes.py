"""Test doubles shared by the API and UI tests."""

import re

from app.llm.client import LLMRanking, LLMRecommendation, LLMResult

CANDIDATE_ID = re.compile(r'"restaurant_id":"([0-9a-f]{12})"')


class FakeLLM:
    """Ranks the candidates in reverse pre-rank order, so tests can tell its order apart.

    Set `error` to make every call fail, or `explanation` to control the text it writes.
    """

    def __init__(self, is_configured: bool = True):
        self.is_configured = is_configured
        self.key_rejected = False
        self.error: Exception | None = None
        self.explanation = "Picked {rid}."
        self.usage = {
            "prompt_tokens": 4000,
            "cached_tokens": 1000,
            "completion_tokens": 1500,
            "reasoning_tokens": 900,
        }
        self.calls = 0

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        ids = CANDIDATE_ID.findall(user_message)[::-1]
        recs = [
            LLMRecommendation(
                restaurant_id=rid, rank=i, explanation=self.explanation.format(rid=rid)
            )
            for i, rid in enumerate(ids, start=1)
        ]
        return LLMResult(
            ranking=LLMRanking(recommendations=recs),
            model="fake-model",
            request_id="req_fake",
            stop_reason="stop",
            latency_ms=1,
            usage=dict(self.usage),
        )
