pub struct PromptBuilder {
    source_language: &'static str,
    target_language: &'static str,
    format: String,
}

pub struct Prompt {
    pub system: String,
    pub user: String,
}

impl PromptBuilder {
    pub fn new() -> PromptBuilder {
        PromptBuilder {
            source_language: "auto",
            target_language: "English",
            format: "text".to_string(),
        }
    }

    pub fn set_format(&mut self, format: &str) -> &mut PromptBuilder {
        self.format = format.to_string();
        self
    }

    pub fn set_source_language(&mut self, s: &'static str) -> &mut PromptBuilder {
        self.source_language = s;
        self
    }

    pub fn set_target_language(&mut self, t: &'static str) -> &mut PromptBuilder {
        self.target_language = t;
        self
    }

    pub fn build(&self, q: &String) -> Prompt {
        let system = if self.format == "html" {
            "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You must preserve all HTML tags and elements in the translation. You always answer with the translated text and nothing else."
        } else {
            "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You always answer with the translated text and nothing else."
        }.to_string();

        let user = (if self.source_language == "auto" {
            format!(
                "Translate the text below to {}.\n\nText: {}\n\n{}:\n",
                self.target_language, q, self.target_language
            )
        } else {
            format!(
                "Translate the text below from {} to {}.\n\n{}: {}\n\n{}:\n",
                self.source_language,
                self.target_language,
                self.source_language,
                q,
                self.target_language
            )
        })
        .to_string();

        Prompt { system, user }
    }
}
