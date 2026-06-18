# LTEngine

Free and Open Source Local AI Machine Translation API, written in Rust, entirely self-hosted and compatible with [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Its translation capabilities are powered by large language models (LLMs) that run locally on your machine via [llama.cpp](https://github.com/ggml-org/llama.cpp). 

![Translation](https://github.com/user-attachments/assets/37dd4e20-382b-459d-bcc1-5de3ed4b4c18)

The LLMs in LTEngine are much larger than the lightweight transformer models in [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Thus memory usage and speed are traded off for quality of outputs, which for some languages has been reported as being [on par or better than DeepL](https://community.libretranslate.com/t/ltengine-llm-powered-local-machine-translation/1862/5).

It is possible to run LTEngine entirely on the CPU, but an accelerator will greatly improve performance. Supported accelerators currently include:

 * CUDA
 * Metal (macOS)
 * Vulkan

 The largest model (`gemma3-27b`) can fit on a single consumer RTX 3090 with 24G of VRAM.

> ⚠️ LTEngine is in active development. Check the [Roadmap](#roadmap) for current limitations.


## Requirements

 * [Rust](https://www.rust-lang.org/)
 * [clang](https://clang.llvm.org/)
 * [CMake](https://cmake.org/)
 * A C++ compiler (g++, MSVC) for building the llama.cpp bindings

## Build

```bash
git clone https://github.com/LibreTranslate/LTEngine --recursive
cd LTEngine
cargo build [--features cuda,vulkan,metal] --release
```

## Run

```bash
./target/release/ltengine
```

To run different LLM models:

```bash
./target/release/ltengine -m gemma3-12b [--model-file /path/to/model.gguf]
```

## Models

LTEngine supports any GGUF language model supported by [llama.cpp](https://github.com/ggml-org/llama.cpp). You can pass a path to load a custom .gguf model using the `--model-file` parameter. Otherwise LTEngine will download one of the Gemma3 models based on the `-m` parameter: 

| Model      | RAM Usage | GPU Usage | Notes                               | Default            |
| ---------- | --------- | --------- | ----------------------------------- | ------------------ |
| gemma3-1b  | 1G        | 2G        | Good for testing, poor translations |                    |
| gemma3-4b  | 4G        | 4G        |                                     | :heavy_check_mark: |
| gemma3-12b | 8G        | 10G       |                                     |                    |
| gemma3-27b | 16G       | 18G       | Best translation quality, slowest   |                    |

Memory usage numbers are approximate.

### Simple

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Hello!",
    source: "en",
    target: "es",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "translatedText": "¡Hola!"
}
```

List of language codes: https://0.0.0.0:5000/languages

### Auto Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5000/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Ciao!",
    source: "auto",
    target: "en",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "detectedLanguage": {
        "confidence": 83,
        "language": "it"
    },
    "translatedText": "Bye!"
}
```

## Language Bindings

You can use the LTEngine API using the following bindings:

- Rust: <https://github.com/DefunctLizard/libretranslate-rs>
- Node.js: <https://github.com/franciscop/translate>
- TypeScript: <https://github.com/tderflinger/libretranslate-ts>
- .Net: <https://github.com/sigaloid/LibreTranslate.Net>
- Go: <https://github.com/SnakeSel/libretranslate>
- Python: <https://github.com/argosopentech/LibreTranslate-py>
- PHP: <https://github.com/jefs42/libretranslate>
- C++: <https://github.com/argosopentech/LibreTranslate-cpp>
- Swift: <https://github.com/wacumov/libretranslate>
- Unix: <https://github.com/argosopentech/LibreTranslate-sh>
- Shell: <https://github.com/Hayao0819/Hayao-Tools/tree/master/libretranslate-sh>
- Java: <https://github.com/suuft/libretranslate-java>
- Ruby: <https://github.com/noesya/libretranslate>
- R: <https://github.com/myanesp/libretranslateR>

## Roadmap

 - [ ] Remove mutex block that currently limits the software to process one single translation request at a time due to a possible bug in llama.cpp. 
 - [ ] Cancel inference (stop generating tokens) when HTTP connections are aborted by clients. I'm unsure how this could done with actix-web.
 - [ ] Add support for `/translate_file` (ability to translate files).
 - [ ] Add support for sentence splitting. Currently text is sent to the LLM as-is, but longer texts (like documents) should be split into chunks, translated and merged back.
 - [ ] Better language detection for short texts (port [LexiLang](https://github.com/LibreTranslate/LexiLang) to Rust)
 - [ ] Test/add more LLM models aside from Gemma3
 - [ ] Create comparative benchmarks between LTEngine and proprietary software.
 - [ ] Add support for command line inference (run `./ltengine translate` as a command line app separate from `./ltengine server`)
 - [ ] Make ltengine available as a library, possibly creating bindings for other languages like Python.
 - [x] Automated builds / CI
 - [ ] Your ideas? We welcome contributions.

## Contributing

We welcome contributions! Just open a pull request.

## Credits

This work is largely possible thanks [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) which provide the Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Trademark

See [Trademark Guidelines](https://github.com/LibreTranslate/LibreTranslate/blob/main/TRADEMARK.md)
