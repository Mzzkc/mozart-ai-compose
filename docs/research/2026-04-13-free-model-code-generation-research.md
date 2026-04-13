# Free Model Code Generation Research

**Date:** 2026-04-13
**Purpose:** Evaluate code generation capabilities of free models on OpenRouter for use as Marianne musician instruments
**Use Case:** Model receives Python class stubs (typed interfaces with docstrings, ~500 tokens) and must generate executable Python code that calls methods on those interfaces. Generated code runs in a sandbox with real implementations behind the stubs.

---

## Executive Summary

Five free-tier models on OpenRouter were evaluated for code generation against typed Python interfaces. All five have free variants available. The strongest candidates for the typed-interface code generation use case are **MiniMax M2.5** and **Gemma 4 31B**, with **Nemotron 3 Super** as a strong third option for agentic workflows. **GLM-4.5-Air** is serviceable but older. **Llama 4 Maverick** has the weakest code generation reliability despite its large parameter count.

### Quick Comparison

| Model | OpenRouter ID | Context | Max Output | Free? | Tool Use | Code Reliability |
|-------|--------------|---------|------------|-------|----------|-----------------|
| MiniMax M2.5 | `minimax/minimax-m2.5:free` | 196K | 8,192 | Yes | Yes (BFCL 76.9%) | High |
| Gemma 4 31B | `google/gemma-4-31b-it:free` | 262K | 32,768 | Yes | Yes (native) | High |
| Nemotron 3 Super | `nvidia/nemotron-3-super-120b-a12b:free` | 262K | 262K | Yes | Yes | High |
| GLM-4.5-Air | `z-ai/glm-4.5-air:free` | 131K | 96,000 | Yes | Yes (BFCL 76.4%) | Medium-High |
| Llama 4 Maverick | `meta-llama/llama-4-maverick:free` | 1M | 16,384 | Yes | Unreliable | Medium |

### Free Tier Rate Limits (all models)

- 20 requests/minute
- 50 requests/day (without credits)
- 1,000 requests/day (with $10+ credits purchased)

---

## 1. MiniMax M2.5

**OpenRouter ID:** `minimax/minimax-m2.5:free`
**Architecture:** Sparse MoE, 456B total parameters
**Context Window:** 196,608 tokens
**Max Output:** 8,192 tokens
**Release:** February 2026
**Provider on Free Tier:** OpenInference

### 1.1 Code Generation Reliability

MiniMax M2.5 is currently one of the strongest open models for code generation. Key benchmark scores:

- **SWE-Bench Verified:** 80.2% (within 0.6% of Claude Opus 4.6)
- **Multi-SWE-Bench:** 51.3% (first place among all models)
- **HumanEval:** 89.6%
- **LiveCodeBench:** 65.0%
- **Terminal-Bench:** 42.2%

Trained on 10+ programming languages across 200,000+ real-world environments. Python is a primary training language. The model can generate 1200+ lines of structured code in a single response.

**Known failure modes:**
- Slower per-round due to mandatory thinking overhead (~2.2s thinking latency on simple tasks)
- Can ignore system prompt constraints when its thinking process decides "completeness is important"
- HTTP 400 errors with structured-output requests in Anthropic-compatible format (relevant for API integration)
- Output limited to 8,192 tokens on free tier -- this is a hard constraint for complex code generation

### 1.2 Typed Interface Support

Strong. The model was trained extensively on typed code across multiple languages. It decomposes projects structurally before generating code, which aligns well with receiving typed stubs as input. No specific reports of interface contract violations found, but the system-prompt-override behavior means explicit constraints about interface adherence should be strongly worded.

### 1.3 Prompt Patterns

- Uses mandatory `<think>...</think>` reasoning blocks before generating code
- Anthropic-compatible API is the officially recommended interface
- Start prompts with clear, specific task descriptions
- The model plans before coding -- providing typed stubs as context works with its natural decomposition pattern
- **Recommendation:** Provide the typed interface stubs in the system prompt or early in the user message. Be explicit about constraints ("You MUST only call methods defined in the provided interface"). The thinking overhead means simple tasks get overthought.

### 1.4 Context Window Utilization

196K context is generous for the use case (~500 token stubs). No specific reports of degradation within the supported window. The model was trained for real-world productivity tasks that involve large document contexts.

### 1.5 Fill-in-the-Middle

No documented FIM support. MiniMax M2.5 is an instruction-tuned model, not a code-completion model. FIM would need to be simulated via prompting.

### 1.6 Function Calling / Tool Use

Yes, native support. BFCL v3 score: 76.9%. Supports OpenAI-format tool definitions. However, there is a documented issue where the thinking model's reasoning can interfere with tool call efficiency in agent scenarios -- it may overthink simple tool selections.

### 1.7 Key Risks for Use Case

- **8K output limit on free tier** severely constrains generated code length
- Mandatory thinking adds latency
- May override explicit constraints if reasoning decides otherwise

---

## 2. Google Gemma 4 31B

**OpenRouter ID:** `google/gemma-4-31b-it:free`
**Architecture:** Dense transformer, 30.7B parameters
**Context Window:** 262,144 tokens
**Max Output:** 32,768 tokens
**Release:** April 2026
**License:** Apache 2.0

### 2.1 Code Generation Reliability

Gemma 4 represents a massive leap from Gemma 3 in code generation:

- **LiveCodeBench v6:** 80.0% (up from 29.1% for Gemma 3 27B)
- **Codeforces Elo:** 2150 ("Candidate Master" level)
- **HumanEval:** ~+8.7% improvement over Gemma 3 (exact score not published, estimated ~85-88%)
- **MMLU Pro:** 85.2%

Supports Python, JavaScript/TypeScript, Java, C++, Rust, Go, SQL natively.

**Known failure modes:**
- Performance degrades on abstract counterfactuals and tasks requiring consistent state tracking across many reasoning steps
- Multi-step formal proofs and advanced symbolic manipulation need explicit step-by-step prompting
- Reliability decreases with more than 3 parallel function calls per turn
- Slow inference on some hardware configurations (vLLM FlashAttention fallback issues, though this is provider-side)

### 2.2 Typed Interface Support

Excellent. Gemma 4 has specific built-in support for defining tools via raw Python functions with type hints -- the system parses function signatures, type hints, and docstrings automatically. Google recommends docstrings adhere to Google Python Style Guide for best results. This means the model is specifically trained to understand and respect typed Python interfaces.

Supports structured output generation via guided decoding, constraining output to valid JSON matching a provided schema.

### 2.3 Prompt Patterns

- Configurable thinking/reasoning mode (not mandatory, unlike MiniMax)
- Two methods for tool definition: JSON schema or raw Python function signatures with type hints
- Use JSON/XML schemas, function calling, and explicit instruction templates for reliability
- **Recommendation:** This model is ideal for the typed-stub use case. Provide Python class stubs directly -- the model is trained to parse type hints and docstrings. Use the system prompt to set the code generation context, and provide stubs as tool/function definitions or as code blocks in the user message.

### 2.4 Context Window Utilization

262K token context is the largest among the dense models evaluated. The model was designed for processing entire codebases in a single prompt. Native support for 140+ languages. Video input support suggests strong multi-modal context handling.

### 2.5 Fill-in-the-Middle

Gemma 4 itself does not support FIM natively. However, the CodeGemma variant (based on earlier Gemma architecture) has full FIM support with `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` tokens. These are **not** available on the Gemma 4 31B instruct model on OpenRouter.

### 2.6 Function Calling / Tool Use

Yes, native support. This is one of Gemma 4's headline features. The model can:
- Accept tool definitions as JSON schema or Python function signatures
- Generate multiple tool calls per turn (reliability drops past 3)
- Produce structured JSON output matching schemas

### 2.7 Key Risks for Use Case

- No FIM support
- Newer model (April 2026) -- less community battle-testing than MiniMax M2.5
- The 26B MoE variant (`google/gemma-4-26b-a4b-it:free`) is also free but scores lower on LiveCodeBench (77.1% vs 80.0%)

---

## 3. NVIDIA Nemotron 3 Super

**OpenRouter ID:** `nvidia/nemotron-3-super-120b-a12b:free`
**Architecture:** Hybrid Mamba-Transformer MoE, 120B total / 12B active
**Context Window:** 262,144 tokens
**Max Output:** 262,144 tokens
**Release:** March 2026
**License:** NVIDIA Open License

### 3.1 Code Generation Reliability

Nemotron 3 Super is the strongest open-weight model on SWE-bench and shows excellent code generation:

- **SWE-Bench Verified:** 60.47% (highest open-weight score at release)
- **LiveCodeBench:** 81.19%
- **HumanEval (Nano variant):** 78.05% (Super variant expected higher, exact score not published)

Multi-Token Prediction (MTP) provides 2-3x wall-clock speedup on structured generation like code. The model activates distinct expert pathways for different programming languages (e.g., Python syntax vs SQL logic).

**Known failure modes:**
- Not pre-configured for native function calling like Claude/GPT-4o -- requires JSON schema patterns and testing before production
- MTP does not work with NVFP4/FP8 on some inference engines
- Poor performance on creative/image generation tasks (27% on image tooling)

### 3.2 Typed Interface Support

Good but requires more scaffolding. The model excels at structured outputs (JSON, code, formatted documents) and can handle typed interfaces, but it was primarily trained for agentic reasoning rather than interface-contract code generation specifically. The expert routing means it can specialize its attention for Python typing constructs.

### 3.3 Prompt Patterns

- Uses `<think>...</think>` reasoning tokens
- Excels when given entire repository contexts in a single prompt
- Handles structured output prompts well -- JSON schemas, typed interfaces
- **Recommendation:** Pack the typed stubs along with execution context into a single structured prompt. The model works best with detailed prompts that include clear output format expectations. Explicitly request Python code that implements the given interface.

### 3.4 Context Window Utilization

262K on OpenRouter (1M claimed in some contexts). The Mamba-Transformer hybrid architecture is specifically designed for efficient long-context processing. Testing shows strong performance loading entire codebases into context. Outperforms GPT-OSS-120B and Qwen3.5-122B on RULER at 1M context length.

### 3.5 Fill-in-the-Middle

No documented FIM support. The model is instruction-tuned for agentic reasoning, not code completion.

### 3.6 Function Calling / Tool Use

Supported but not native. The model was post-trained across 15 environments in NeMo Gym covering multi-step tool use, code execution, and structured output. Tool calling uses the `qwen3_coder` parser in deployment. JSON schema format for tool definitions. Reliability in multi-step tool-calling scenarios is strong but requires proper scaffolding.

### 3.7 Key Risks for Use Case

- **Free tier logs all prompts and outputs** to improve NVIDIA's models -- privacy concern for sensitive code
- Tool calling requires more setup than models with native support
- The massive output window (262K) is unusual and may be a documentation error or provider-specific setting

---

## 4. GLM-4.5-Air

**OpenRouter ID:** `z-ai/glm-4.5-air:free`
**Architecture:** MoE, 106B total / 12B active
**Context Window:** 131,072 tokens
**Max Output:** 96,000 tokens
**Release:** July 2025
**Knowledge Cutoff:** December 2024
**Provider:** Z.ai (direct)

### 4.1 Code Generation Reliability

GLM-4.5-Air is the older model in this group but shows solid coding performance:

- **LiveCodeBench:** 72.9% (top performer at release)
- **BFCL v3:** 76.4% (outperforms Gemini Pro 2.5)
- **SWE-Bench:** 64.2%
- **HumanEval:** Not published for Air variant; GLM-4.6 scores 84.2% (Air expected lower)
- **AIME 2024:** 89.4%

Optimized for tool invocation, web browsing, and software engineering.

**Known failure modes:**
- Multiple sequence generation (n>1) causes interwoven, degenerate output
- Streaming with reasoning enabled always starts with a spurious newline
- Can enter repetitive thinking loops on complex prompts
- Overthinking behavior -- may repeat itself extensively before producing output

### 4.2 Typed Interface Support

Good. GLM-4.5-Air generates "well-structured, scalable, high-quality code based on natural language instructions" and supports OpenAI-compatible function calling with JSON schema parameter definitions. The model is specifically noted as being integrable into code-centric agents like Claude Code and Roo Code.

### 4.3 Prompt Patterns

- Dual mode: "thinking mode" for complex reasoning, "non-thinking mode" for immediate responses
- OpenAI-compatible API format
- Supports `tools`, `tool_choice` parameters directly
- **Recommendation:** Use thinking mode for complex code generation, non-thinking mode for simple stub implementations. Provide typed stubs as function definitions or code context. The model's tendency to overthink means simpler interfaces may benefit from non-thinking mode.

### 4.4 Context Window Utilization

131K is the smallest window in this group but still generous for the use case. Hybrid inference modes allow trading context processing depth for speed.

### 4.5 Fill-in-the-Middle

No documented FIM support. GLM architecture historically uses autoregressive blank infilling with 2D positional encodings, but this is not exposed as a FIM API in the instruction-tuned model.

### 4.6 Function Calling / Tool Use

Yes, native support on OpenRouter. Parameters `tools` and `tool_choice` are directly supported. Strong performance on BFCL (76.4%). The model can be used as a drop-in replacement for OpenAI function calling workflows.

### 4.7 Key Risks for Use Case

- Oldest model in the evaluation (July 2025), superseded by GLM-4.6 and GLM-4.7
- 131K context is adequate but smallest in the group
- Repetitive output loops are a real risk with complex prompts
- Knowledge cutoff (Dec 2024) means it may not know newer Python typing features

---

## 5. Meta Llama 4 Maverick

**OpenRouter ID:** `meta-llama/llama-4-maverick:free`
**Architecture:** MoE, 400B total / 17B active, 128 experts
**Context Window:** 1,048,576 tokens (1M)
**Max Output:** 16,384 tokens
**Release:** April 2025
**Knowledge Cutoff:** August 2024

### 5.1 Code Generation Reliability

Llama 4 Maverick has the weakest code generation scores in this group despite having the most total parameters:

- **MBPP:** 77.6
- **LiveCodeBench:** 43.4 (from 2025 evaluation) to 70.4 (Meta's claimed number -- discrepancy noted)
- **HumanEval:** 62% (independent) vs "matches GPT-4" (Meta's claim) -- **significant discrepancy**
- **SWE-Bench:** Not a top performer

**Known failure modes -- this is where Maverick is most concerning:**
- Performance frequently degrades mid-execution in multi-step tasks
- Malformed tool calls mid-conversation
- Loss of JSON structure in output
- Forgets earlier decisions during extended generation
- Only marginally better than 32B models in some uncertainty-driven tasks
- Correct initial reasoning followed by execution breakdown

Academic evaluation (KAMI v0.1 benchmark) describes Maverick as having "a mixture of high-ceiling successes and unusually fragile failure modes."

### 5.2 Typed Interface Support

Questionable. The model can generate code against typed interfaces when the task is straightforward, but the documented mid-execution degradation means it may start correctly following an interface contract and then drift. The 128-expert MoE architecture provides specialized routing, but execution stability is the bottleneck, not comprehension.

### 5.3 Prompt Patterns

- Standard Llama instruction format
- Supports multilingual text and code output
- 12 language support
- **Recommendation:** Keep code generation requests simple and atomic. Do not ask Maverick to generate long, multi-method implementations in one pass. Break complex interface implementations into individual method-level requests. Validate output structure rigorously.

### 5.4 Context Window Utilization

1M token context is the largest in this evaluation. Testing shows the model maintains better cross-module analysis at 1M vs 128K. However, the knowledge cutoff (August 2024) is the oldest, which means less exposure to modern Python patterns.

Independent testing confirmed: the 1M context model provides "more complete analysis with fewer disconnects between high-level modules and deeply nested functions" compared to truncated contexts.

### 5.5 Fill-in-the-Middle

No documented FIM support for the instruct variant.

### 5.6 Function Calling / Tool Use

Supported but unreliable. Multiple sources document:
- Malformed tool calls mid-conversation
- JSON structure loss during extended tool-calling chains
- The model begins with correct tool selections but degrades

Tool calling is available on OpenRouter (standard JSON schema format), but should not be relied upon for multi-step workflows.

### 5.7 Key Risks for Use Case

- **Execution instability is the primary concern** -- code may start correct and degrade
- Oldest knowledge cutoff (Aug 2024) in the group
- Discrepancy between Meta's claimed benchmarks and independent evaluations raises trust concerns
- Tool calling unreliability makes agentic integration risky

---

## Recommendations for Typed Interface Code Generation

### Tier 1: Recommended

1. **Gemma 4 31B** (`google/gemma-4-31b-it:free`) -- Best for the typed-stub use case specifically. Native function calling via Python type hints, 32K output window, configurable reasoning mode. The model is literally trained to parse typed Python function signatures.

2. **MiniMax M2.5** (`minimax/minimax-m2.5:free`) -- Strongest overall code generation benchmarks, but **8K output limit on free tier** is a hard constraint. If the generated code fits in 8K tokens, this is the highest-reliability option.

### Tier 2: Viable

3. **Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b:free`) -- Strong code generation, but tool calling requires scaffolding and **free tier logs all inputs/outputs to NVIDIA**. Best for non-sensitive code generation where privacy is not a concern.

4. **GLM-4.5-Air** (`z-ai/glm-4.5-air:free`) -- Solid function calling support, good code generation, but older model with repetitive output risks. The 96K max output is generous.

### Tier 3: Use with Caution

5. **Llama 4 Maverick** (`meta-llama/llama-4-maverick:free`) -- Not recommended for reliable code generation against typed interfaces. Mid-execution degradation and tool-call instability make it unsuitable for sandbox execution where correctness is required.

### Prompt Pattern Recommendations (Cross-Model)

For the specific use case of generating code against typed Python stubs:

```
System prompt:
  "You are a code generator. You receive Python class stubs with type hints
   and docstrings. Generate executable Python code that uses these interfaces.
   Only call methods defined in the provided stubs. Do not import external
   libraries unless specified. Output only the Python code, no explanation."

User message:
  "Given the following interface:\n\n```python\n{stub_code}\n```\n\n
   Generate Python code that {task_description}."
```

- For Gemma 4: Provide stubs as raw Python function signatures; the model parses type hints natively
- For MiniMax M2.5: Be very explicit about constraints; the thinking process may override brevity
- For Nemotron 3: Include structured output format expectations in the prompt
- For GLM-4.5-Air: Use non-thinking mode for simple implementations to avoid repetitive loops
- For Llama 4 Maverick: Break into atomic, single-method requests; validate JSON structure of every response

### FIM Support Summary

**None of the five models support FIM natively in their instruct variants on OpenRouter.** FIM is a feature of base/code-specific model variants (CodeGemma, Code Llama, etc.), not the instruction-tuned chat models available on OpenRouter's free tier. Code generation must use the standard chat completion API with explicit prompting.

---

## Uncertainties and Caveats

1. **Benchmark score discrepancies** -- Llama 4 Maverick has significant gaps between Meta's claimed scores and independent evaluations. MiniMax M2.5's HumanEval score (89.6%) comes from aggregator sites, not the official technical report.

2. **Free tier behavior** -- Free variants may have different quantization, lower priority, or additional latency compared to paid variants. Nemotron 3 Super explicitly logs all free-tier traffic.

3. **Context window claims** -- Nemotron 3 Super claims 1M context in some sources but OpenRouter lists 262K for the free variant. Llama 4 Maverick's 1M context may be truncated by some OpenRouter providers.

4. **Evolving landscape** -- MiniMax M2.7 and GLM-4.7 exist as successors. These evaluations are a snapshot as of 2026-04-13.

5. **No independent typed-interface testing** -- All assessments are inferred from general code benchmarks and documented capabilities. No evaluation specifically tested the "receive stubs, generate implementation" pattern.

---

## Sources

- [MiniMax M2.5 Official](https://www.minimax.io/news/minimax-m25)
- [MiniMax M2.5 on HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMax M2.5 Tool Calling Guide](https://github.com/MiniMax-AI/MiniMax-M2.5/blob/main/docs/tool_calling_guide.md)
- [MiniMax M2.5 Function Calling Issues](https://github.com/MiniMax-AI/MiniMax-M2/issues/77)
- [Gemma 4 Official](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 Function Calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 Limitations](https://gemmai4.com/limitations/)
- [Gemma 4 Benchmarks](https://gemma4all.com/blog/gemma-4-benchmarks-performance)
- [Nemotron 3 Super Technical Blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)
- [Nemotron 3 Super on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
- [Nemotron 3 Super Agentic Coding](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html)
- [GLM-4.5 Official Documentation](https://docs.z.ai/guides/llm/glm-4.5)
- [GLM-4.5 on HuggingFace](https://huggingface.co/zai-org/GLM-4.5)
- [GLM-4.5 Known Issues (vLLM)](https://github.com/vllm-project/vllm/issues/23251)
- [Llama 4 Official](https://www.llama.com/models/llama-4/)
- [Llama 4 Maverick on HuggingFace](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)
- [Llama 4 Agentic Failure Analysis (KAMI)](https://arxiv.org/html/2512.07497v2)
- [OpenRouter Free Models](https://openrouter.ai/collections/free-models)
- [OpenRouter Free Model List (Apr 2026)](https://costgoat.com/pricing/openrouter-free-models)
- [OpenRouter Rate Limits](https://openrouter.ai/docs/api/reference/limits)
- [Best Open-Source Coding Models 2026](https://www.morphllm.com/best-open-source-coding-model-2026)
- [LiveCodeBench Leaderboard](https://artificialanalysis.ai/evaluations/livecodebench)
