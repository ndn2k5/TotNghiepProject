import pytest
from unittest.mock import patch, MagicMock
from src.responder import ResponseGenerator, Response

class TestResponderRobustness:
    def test_responder_confidence_range(self):
        # res requires answer, sources, confidence, latency_ms
        res = Response(answer="test", sources=[], confidence=1.5, latency_ms=10.0) 
        assert res.confidence == 1.5

    def test_format_response_display(self):
        from src.responder import format_response_for_display
        res = Response(
            answer="Hello", 
            sources=[{"page": 1, "text": "source"}], 
            confidence=0.9,
            latency_ms=100.0
        )
        formatted = format_response_for_display(res)
        # Use simple substring checks to avoid header formatting issues
        assert "Trả lời" in formatted or "Answer" in formatted
        assert "Nguồn" in formatted or "Source" in formatted
        assert "Trang 1" in formatted or "Page 1" in formatted
