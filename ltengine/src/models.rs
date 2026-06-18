use std::collections::HashMap;
use std::path::PathBuf;
use hf_hub::api::sync::ApiBuilder;
use anyhow::{anyhow, Context, Result};

#[derive(Clone, Debug)]
pub struct HuggingFace {
    pub repo: &'static str,
    pub model: &'static str,
}

#[derive(Debug)]
pub enum Model {
    Local {
        path: PathBuf,
    },
    Remote {
        hf: HuggingFace,
    },
}

pub static MODELS: once_cell::sync::Lazy<HashMap<&'static str, HuggingFace>> = once_cell::sync::Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gemma3-1b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-1b-it-q4_0.gguf" });
    m.insert("gemma3-4b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-4b-it-q4_0.gguf" });
    m.insert("gemma3-12b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-12b-it-q4_0.gguf" });
    m.insert("gemma3-27b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-27b-it-q4_0.gguf" });
    m.insert("gemma4-e4b", HuggingFace { repo: "bartowski/google_gemma-4-E4B-it-GGUF", model: "google_gemma-4-E4B-it-Q4_0.gguf" });
    m
});


impl Model {
    fn load(&self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => {
                if path.exists() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                    Ok(path.clone())
                } else {
                    Err(anyhow!(format!("Invalid path or not a .gguf file: {}", path.display())))
                }
            },
            Model::Remote { hf } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "Unable to create HF API")?
                .model(hf.repo.to_string())
                .get(&hf.model)
                .with_context(|| "Unable to download model"),
        }
    }
}

pub fn load_model(model_id: &String, model_file: &String) -> Result<PathBuf> {
    let model = if !model_file.is_empty() {
        Model::Local {
            path: PathBuf::from(model_file),
        }
    } else {
        Model::Remote {
            hf: MODELS[model_id.as_str()].clone(),
        }
    };

    match model.load() {
        Ok(path) => Ok(path),
        Err(e) => Err(e),
    }
}