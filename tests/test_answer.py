from types import SimpleNamespace as NS

from rageval.answer import answer, build_messages
from rageval.data import Document

DOCS = [
    Document("111", "Vitamin D", "Supplementation reduced hip fractures by 20%."),
    Document("222", "Statins", "Statins lower LDL cholesterol."),
]


class FakeClient:
    """Stands in for anthropic.Anthropic; records the request and returns a canned reply."""

    def __init__(self, response):
        self.request = None
        self.beta = NS(messages=NS(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.request = kwargs
        return self._response


def test_each_document_is_sent_as_a_citable_block():
    content = build_messages("q?", DOCS)[0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[0]["title"] == "111"
    assert content[0]["citations"] == {"enabled": True}


def test_citations_are_mapped_back_to_document_ids():
    reply = NS(
        stop_reason="end_turn",
        content=[
            NS(type="thinking", thinking="", citations=None),
            NS(
                type="text",
                text="It reduced hip fractures.",
                citations=[NS(document_index=0, cited_text="Supplementation reduced hip fractures by 20%. ")],
            ),
        ],
    )
    client = FakeClient(reply)
    result = answer("Does vitamin D help?", DOCS, client=client)

    assert result.text == "It reduced hip fractures. [111]"
    assert result.citations[0].doc_id == "111"
    assert result.citations[0].quote == "Supplementation reduced hip fractures by 20%."
    assert client.request["fallbacks"] == "default"


def test_refusal_is_reported_instead_of_empty_text():
    client = FakeClient(NS(stop_reason="refusal", content=[]))
    assert answer("q?", DOCS, client=client).refused
