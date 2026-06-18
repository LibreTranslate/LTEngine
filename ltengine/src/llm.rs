use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaModel, LlamaChatMessage};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Mutex;
use anyhow::{Result, Context};

/// Query total VRAM (MiB) on device 0 using llama.cpp's backend API.
/// Works for both CUDA and Vulkan builds; returns None for CPU builds.
fn total_vram_mib() -> Option<u64> {
    let mut free = 0usize;
    let mut total = 0usize;

    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn ggml_backend_cuda_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_cuda_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some(total as u64 / (1024 * 1024));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        unsafe extern "C" {
            fn ggml_backend_vk_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_vk_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some(total as u64 / (1024 * 1024));
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
        if let Some(total_mib) = total_vram_mib() {
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
    prompt_lock: Mutex<bool>,
    n_ubatch: u32,
}

pub struct LLMContext<'a>{
    llm: &'a LLM,
    ctx: LlamaContext<'a>,
    ctx_size: i32
}

impl LLM {
    pub fn new(model_path: PathBuf, cpu: bool, verbose: bool) -> Result<Self> {
        if !verbose{
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        }
        
        let backend = LlamaBackend::init()?;

        let model_params = {
            if !cpu && cfg!(any(feature = "cuda", feature = "vulkan")) {
                LlamaModelParams::default().with_n_gpu_layers(9999)
            } else {
                LlamaModelParams::default()
            }
        };

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| "Unable to load model")?;

        let use_gpu = !cpu && cfg!(any(feature = "cuda", feature = "vulkan"));
        let n_ubatch = pick_n_ubatch(use_gpu);

        Ok(LLM { backend, model, prompt_lock: Mutex::new(true), n_ubatch })
    }

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext>{
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
        let tmpl = self.model.chat_template(None)?;
        let llm_input = self.model.apply_chat_template(&tmpl, &[
            LlamaChatMessage::new("system".to_string(), system)?,
            LlamaChatMessage::new("user".to_string(), user)?
        ], true)?;

        let tokens_list = self.model
            .str_to_token(&llm_input
            , AddBos::Always)
            .with_context(|| format!("Failed to tokenize {llm_input}"))?;
        // for token in &tokens_list {
        //     eprint!("{} {} | ", self.model.token_to_str(*token, Special::Tokenize)?, token);
        // }
        let ctx_size: i32 = tokens_list.len() as i32 * 3;
        // Lock before create_context: context allocation uses GPU resources and
        // two concurrent allocations corrupt each other even before inference starts.
        let _lock = self.prompt_lock.lock();
        // TODO: The llama bindings (or llama itself?) do not appear to be totally thread-safe
        // as garbage starts to come out when we run inference in parallel
        // this might need to be investigated and fixed. For now we lock and process requests
        // one at a time.
        // TODO: consider locking with a timeout: https://docs.rs/parking_lot/latest/parking_lot/type.Mutex.html#method.try_lock_for
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
                    
                let output_bytes = self.llm.model.token_to_bytes(token, Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
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

        Ok(output)
    }
}
