//! Remote translation backend: an OpenAI-compatible chat/completions
//! endpoint (e.g. a llama.cpp server, vLLM, OpenAI itself).
//!
//! Reuses the direction-aware prompt built by prompt.rs and returns the
//! reply content verbatim, so the existing post-processing
//! (improve_formatting) applies unchanged.
//! Motivation: LibreTranslate/LTEngine#3 ("External model support").

use std::time::Duration;

use anyhow::{Result, anyhow};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use reqwest::Client;

use crate::prompt::Prompt;

#[derive(Debug, thiserror::Error)]
pub enum OpenAIBackendError {
    #[error("Translation endpoint returned an error: {0}")]
    Remote(String),
}

pub struct OpenAIBackend {
    client: async_openai::Client<OpenAIConfig>,
    model: String,
}

impl OpenAIBackend {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(normalize_base_url(base_url))
            .with_api_key(api_key);

        // LLM answers can take minutes on long inputs; without a timeout a
        // hung endpoint would hold the request open forever.
        let http_client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("building HTTP client for the OpenAI backend failed");

        OpenAIBackend {
            client: async_openai::Client::build(http_client, config),
            model: model.to_string(),
        }
    }

    /// Non-streaming completion with the Prompt{system, user} messages.
    /// The first choice's content is returned verbatim. Thinking models
    /// deliver their reasoning in a separate reasoning_content field, which
    /// async-openai does not map into content and is therefore discarded.
    pub async fn translate(&self, prompt: Prompt) -> Result<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(prompt.system)
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt.user)
                    .build()?
                    .into(),
            ])
            .temperature(0.3_f32)
            .build()?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| OpenAIBackendError::Remote(e.to_string()))?;

        response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| anyhow!("Model produced empty output"))
    }
}

/// async-openai appends "/chat/completions" to the base URL; a trailing
/// slash would turn that into "//chat/completions".
fn normalize_base_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::normalize_base_url;

    #[test]
    fn base_url_trailing_slash_is_trimmed() {
        assert_eq!(
            normalize_base_url("http://llama.local:8080/v1/"),
            "http://llama.local:8080/v1"
        );
        assert_eq!(
            normalize_base_url("  http://llama.local:8080/v1  "),
            "http://llama.local:8080/v1"
        );
        assert_eq!(
            normalize_base_url("http://llama.local:8080/v1"),
            "http://llama.local:8080/v1"
        );
    }
}
