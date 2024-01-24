"""Integration tests for basic OLS REST API endpoints."""

from unittest.mock import patch

import requests
import pytest
from fastapi.testclient import TestClient

from ols import constants
from ols.app.main import app
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader

from tests.mock_classes.llm_loader import mock_llm_loader
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.app.models.config import LLMConfig, ProviderConfig, OLSConfig, ModelConfig

from ols.utils import config

client = TestClient(app)


def test_liveness() -> None:
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_readiness() -> None:
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptins will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    response = client.post(
        "/v1/debug/query", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": "1234",
        "query": "test query",
        "response": "test response",
    }


def test_post_question_on_unexpected_payload() -> None:
    """Check the REST API /v1/query with POST HTTP method when unexpected payload is posted."""
    response = client.post("/v1/query", json="this is really not proper payload")
    print(response)
    assert response.status_code == requests.codes.unprocessable
    assert response.json() == {
        "detail": [
            {
                "input": "this is really not proper payload",
                "loc": ["body"],
                "msg": "Input should be a valid dictionary or object to extract fields from",
                "type": "model_attributes_type",
                "url": "https://errors.pydantic.dev/2.5/v/model_attributes_type",
            }
        ],
    }


def test_post_question_on_invalid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    answer = (constants.INVALID, "anything")
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        response = client.post(
            "/v1/query", json={"conversation_id": "1234", "query": "test query"}
        )
        assert response.status_code == requests.codes.unprocessable
        assert response.json() == {
            "detail": {
                "response": "Sorry, I can only answer questions about OpenShift "
                "and Kubernetes. This does not look like something I "
                "know how to handle."
            }
        }


def test_post_question_on_unknown_response_type() -> None:
    """Check the REST API /v1/query with POST HTTP method when unknown response type is returned."""
    # let's pretend the question is valid, but there's an error, without even asking LLM
    answer = (constants.VALID, constants.SOME_FAILURE)
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        response = client.post(
            "/v1/query", json={"conversation_id": "1234", "query": "test query"}
        )
        assert response.status_code == requests.codes.internal_server_error
        assert response.json() == {
            "detail": {"response": "Internal server error. Please try again."}
        }

llm_cfgs = [
    [constants.PROVIDER_OPENAI, constants.GPT35_TURBO_1106],
]

@pytest.mark.parametrize("provider, model", llm_cfgs)
def test_post_question_on_noyaml_response_type(monkeypatch, provider, model) -> None:
    """Check the REST API /ols/ with POST HTTP method when unknown response type is returned."""

    config.load_empty_config()
    config.llm_config = LLMConfig()

    providerConfig = ProviderConfig()
    modelConfig = ModelConfig()
    modelConfig.name = model
    providerConfig.models = {model: modelConfig}
    config.llm_config.providers = {provider: providerConfig}

    config.ols_config = OLSConfig()
    config.ols_config.summarizer_model = model
    config.ols_config.summarizer_provider = provider

    def dummy_validator(
        self, conversation: str, query: str, verbose: bool = False
    ) -> list[str]:
        return constants.VALID, constants.NOYAML

    # let's pretend the question is valid without even asking LLM
    # but the question type is unknown
    monkeypatch.setattr(QuestionValidator, "validate_question", dummy_validator)

    from tests.mock_classes.langchain_interface import mock_langchain_interface

    ml = mock_langchain_interface("test response")
    with patch("ols.src.docs.docs_summarizer.LLMLoader", new=mock_llm_loader(ml())):
        with patch("ols.src.docs.docs_summarizer.ServiceContext.from_defaults"):
            response = client.post(
                "/v1/query", json={"conversation_id": "1234", "query": "test query"}
            )
            print(response)
            assert response.status_code == requests.codes.ok
            print(response.json())
            summary = f""" The following response was generated without access to RAG content:

                        success
                      """
            assert response.json() == {
                "query": "test query",
                "conversation_id": "1234",
                "response": summary,
            }
