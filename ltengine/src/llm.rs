use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, LlamaChatMessage};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;
use parking_lot::Mutex;
use std::time::Duration;
use anyhow::{Result, Context};

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Server busy, please try again later")]
    Busy,
}

/// Query (free_mib, total_mib) on device 0. Works for CUDA and Vulkan; None for CPU builds.
#[allow(unused_mut)]
fn vram_mib() -> Option<(u64, u64)> {
    let mut free = 0usize;
    let mut total = 0usize;

    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn ggml_backend_cuda_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_cuda_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some((free as u64 / (1024 * 1024), total as u64 / (1024 * 1024)));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        unsafe extern "C" {
            fn ggml_backend_vk_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_vk_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some((free as u64 / (1024 * 1024), total as u64 / (1024 * 1024)));
        }
    }

    let _ = (free, total);
    None
}

/// Pick n_ubatch based on total VRAM of the primary GPU device.
///
/// On cards with limited VRAM the default n_ubatch can exhaust memory when
/// combined with KV cache and GPU compute buffers, causing an OOM crash.
/// We use n_ubatch=128 on cards with less than 6 GB of total VRAM.
fn pick_n_ubatch(use_gpu: bool) -> u32 {
    let default = LlamaContextParams::default().n_ubatch();
    if use_gpu {
        if let Some((_, total_mib)) = vram_mib() {
            let n = if total_mib >= 6 * 1024 { default } else { 128 };
            eprintln!("ltengine: {} MiB total VRAM, n_ubatch={}", total_mib, n);
            return n;
        }
    }
    default
}

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
    prompt_lock: Mutex<()>,
    n_ubatch: u32,
}

pub struct LLMContext<'a>{
    llm: &'a LLM,
    ctx: LlamaContext<'a>,
    ctx_size: i32
}

impl LLM {
    pub fn new(model_path: PathBuf, cpu: bool, verbose: bool) -> Result<Self> {
        if !verbose {
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        }

        let backend = LlamaBackend::init()?;
        let use_gpu = !cpu && cfg!(any(feature = "cuda", feature = "vulkan"));

        let (model, gpu_layers) = if use_gpu {
            let mut n_gpu = 9999u32;
            let model = loop {
                let model = LlamaModel::load_from_file(
                    &backend, &model_path,
                    &LlamaModelParams::default().with_n_gpu_layers(n_gpu),
                ).with_context(|| "Unable to load model")?;

                // Probe: create a minimal context and decode one token to confirm
                // the GPU has enough VRAM for compute scratch buffers.
                let probe_ok = model.new_context(
                    &backend,
                    LlamaContextParams::default()
                        .with_n_ctx(Some(NonZeroU32::new(8).unwrap()))
                        .with_n_ubatch(1),
                ).ok().and_then(|mut ctx| {
                    let mut batch = LlamaBatch::new(8, 1);
                    batch.add(LlamaToken(0), 0, &[0], true).ok()?;
                    ctx.decode(&mut batch).ok()
                }).is_some();

                if probe_ok {
                    break model;
                }

                let actual = model.n_layer() as u32;
                let current = n_gpu.min(actual);
                let next = current.saturating_sub((current / 10).max(1));
                eprintln!("ltengine: GPU probe failed at {} layers, retrying with {}", current, next);
                n_gpu = next;
                drop(model);

                if n_gpu == 0 {
                    return Err(anyhow::anyhow!("GPU inference failed even with 0 layers"));
                }
            };

            let actual = model.n_layer() as u32;
            let on_gpu = n_gpu.min(actual);
            let gpu_layers = if on_gpu < actual { Some(on_gpu) } else { None };
            (model, gpu_layers)
        } else {
            let model = LlamaModel::load_from_file(
                &backend, model_path,
                &LlamaModelParams::default().with_n_gpu_layers(0),
            ).with_context(|| "Unable to load model")?;
            (model, None)
        };

        let n_ubatch = pick_n_ubatch(use_gpu);

        match (use_gpu, gpu_layers) {
            (false, _) => eprintln!("ltengine: {} model layers, CPU only", model.n_layer()),
            (true, None) => eprintln!("ltengine: {} model layers, all offloaded to GPU", model.n_layer()),
            (true, Some(n)) => eprintln!("ltengine: {}/{} model layers on GPU, rest on CPU", n, model.n_layer()),
        }

        Ok(LLM { backend, model, prompt_lock: Mutex::new(()), n_ubatch })
    }

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext<'_>>{
        let ctx_params =
            LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(ctx_size as u32).unwrap()))
                .with_n_ubatch(self.n_ubatch);

        // Use all threads
        // ctx_params = ctx_params.with_n_threads(threads);
        // ctx_params = ctx_params.with_n_threads_batch(threads_batch);

        let ctx = self.model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "Unable to create the llama context")?;
        Ok(LLMContext{ llm: self, ctx, ctx_size })
    }

    pub fn run_prompt(&self, system: String, user: String) -> Result<String>{
        let messages = [
            LlamaChatMessage::new("user".to_string(), format!("{system}\n\n{user}"))
                .context("Failed to build chat message")?
        ];

        // Use the model's embedded chat template when llama.cpp can detect it.
        // Falls back to hardcoded Gemma format when detection fails (e.g. Gemma 4
        // until llama-cpp-sys picks up the upstream Gemma 4 template detection fix).
        let llm_input = self.model
            .chat_template(None)
            .ok()
            .and_then(|tmpl| self.model.apply_chat_template(&tmpl, &messages, true).ok())
            .unwrap_or_else(|| format!(
                "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
            ));

        // BOS is not added by apply_chat_template — str_to_token handles it.
        let tokens_list = self.model
            .str_to_token(&llm_input, AddBos::Always)
            .with_context(|| "Failed to tokenize prompt")?;
        // for token in &tokens_list {
        //     eprint!("{} {} | ", self.model.token_to_str(*token, Special::Tokenize)?, token);
        // }
        let ctx_size: i32 = tokens_list.len() as i32 * 3;
        // Lock before create_context: context allocation uses GPU resources and
        // two concurrent allocations corrupt each other even before inference starts.
        // TODO: The llama bindings (or llama itself?) do not appear to be totally thread-safe
        // as garbage starts to come out when we run inference in parallel
        // this might need to be investigated and fixed. For now we lock and process requests
        // one at a time.
        let _lock = self.prompt_lock.try_lock_for(Duration::from_secs(120))
            .ok_or(LLMError::Busy)?;
        let mut ctx = self.create_context(ctx_size)?;
        ctx.process(tokens_list)
    }
}

impl LLMContext<'_>{
    pub fn process(&mut self, tokens_list: Vec<LlamaToken>) -> Result<String>{
        // let ctx_size: i32 = tokens_list.len() as i32 * 3;
        
        // We use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(self.ctx_size.try_into()?, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        self.ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let seq_breakers = vec![b"\n", b":", b"\"", b"*"];

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.0, 0.0, 0.0),
            LlamaSampler::dry(&self.llm.model, 0.0, 1.75, 2, -1, seq_breakers),
            LlamaSampler::top_k(40),
            LlamaSampler::typical(1.0, 0),
            LlamaSampler::top_p(0.95, 0),
            LlamaSampler::min_p(0.05, 0),
            LlamaSampler::xtc(0.0, 0.1, 0, 42),
            LlamaSampler::temp_ext(0.0, 0.0, 1.0),
            LlamaSampler::dist(42)
        ]);

        let mut output = String::new();

        while n_cur <= self.ctx_size {

            // sample the next token
            {
                let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.llm.model.is_eog_token(token) {
                    break;
                }
                    
                let output_string = self.llm.model.token_to_piece(token, &mut decoder, true, None)?;
                output.push_str(&output_string);

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch).with_context(|| "Failed to eval")?;
        }

        // Gemma 4 thinking mode emits thinking content before the actual response in two forms:
        // 1. <|channel>thought\n...<channel|>answer  (full block with closing tag)
        // 2. <|channel>thought answer                (no closing tag, space-separated)
        let output = if let Some(pos) = output.find("<channel|>") {
            output[pos + "<channel|>".len()..].to_owned()
        } else if let Some(rest) = output.strip_prefix("<|channel>thought") {
            rest.trim_start_matches(['\n', ' ']).to_owned()
        } else {
            output
        };

        // Gemma may emit <end_of_turn> as literal text when it cannot translate
        // (e.g. unsupported language/format combination) instead of the special
        // EOG token caught above. Strip it and treat empty output as an error.
        let output = output.replace("<end_of_turn>", "");
        let output = output.trim().to_owned();
        if output.is_empty() {
            return Err(anyhow::anyhow!("Model produced empty output"));
        }

        Ok(output)
    }
}
