"""Tests for OpenAI Chat Model."""
import os
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from squirro_chatbot.models.openai_chat_model import (OpenAIChatModel,
                                                      OpenAIChatModelConfig)


class TestOpenAIChatModel(absltest.TestCase):
    """Tests for OpenAIChatModel class."""

    @patch("squirro_chatbot.models.openai_chat_model.openai.OpenAI")
    @patch("squirro_chatbot.models.openai_chat_model.tiktoken.encoding_for_model")
    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_ORGANIZATION": "test_org"}
    )
    def test_initialization(self, mock_encoding_for_model, mock_openai):
        """Test the initialization of the OpenAIChatModel."""
        config = OpenAIChatModelConfig()
        model = OpenAIChatModel(config)
        self.assertEqual(model.chat_model_name, "gpt-3.5-turbo")
        mock_openai.assert_called_with(api_key="test_key", organization="test_org")

    @patch("squirro_chatbot.models.openai_chat_model.openai.OpenAI")
    @patch("squirro_chatbot.models.openai_chat_model.tiktoken.encoding_for_model")
    def test_generate(self, mock_encoding_for_model, mock_openai):
        """Test the generate function for OpenAIChatModel."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mocked response"))]
        )

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))  # Example token list
        mock_encoder.decode.return_value = "Mocked Question"
        mock_encoding_for_model.return_value = mock_encoder

        config = OpenAIChatModelConfig()
        model = OpenAIChatModel(config)
        response = model.generate("Test prompt")

        self.assertEqual(response, "Mocked response")
        mock_encoder.encode.assert_called_once_with("Test prompt")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Mocked Question"}],
            temperature=0,
            max_tokens=config.generation_length_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )


if __name__ == "__main__":
    absltest.main()
