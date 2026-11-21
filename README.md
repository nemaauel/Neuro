<div align="center">

# Neuro

A statically-checked programming language and tensor runtime for neural networks.

[Building from Source](#building-from-source) | [CLI Usage](#cli-usage) | [Architecture](#architecture) | [Language Guide](#language-guide) | [Standard Library](#standard-library)

</div>

This repository contains the source code for the **Neuro** interpreter, static analyzer, reverse-mode automatic differentiation engine, and CPU/GPU compute backends.

Neuro is designed around an indentation-sensitive syntax with explicit control over tensor memory placement (`@ram`, `@vram`), execution targets (`@cpu`, `@gpu`, `@simd`, `@fuse`), and symbolic dimension verification (`Tensor[B, T, D]`). Programs are written in `.nr` source files and executed directly by the `neuro` runtime without external Python or C++ framework dependencies.

## Building from Source

### Prerequisites

- **Rust toolchain** (edition 2021 or later), installable via [rustup][rustup].
- **GPU drivers** supporting Vulkan or DirectX 12 (optional; if no hardware adapter is found, matrix operations fall back to the multi-threaded CPU backend automatically).

[rustup]: https://rustup.rs/

### Compilation

Clone the repository and build the optimized release binary using Cargo:

```sh
git clone https://github.com/nemaauel/Neuro.git
cd Neuro
cargo build --release
```

The resulting executable is written to `target\release\neuro.exe` on Windows or `target/release/neuro` on Unix-like systems. To install the binary globally into your Cargo bin directory:

```sh
cargo install --path .
```

### Editor Support

A Visual Studio Code extension package (`neuro-lang.vsix`) is included at the root of the repository for `.nr` syntax highlighting:

```sh
code --install-extension neuro-lang.vsix
```

## CLI Usage

All source files passed to the CLI must use the `.nr` extension.

```sh
# Execute a Neuro script
neuro run main.nr
neuro main.nr

# Run static type, mutability, and symbolic shape checks without executing
neuro check main.nr

# Print detected CPU worker threads and active Vulkan GPU adapter
neuro info

# Print version or usage summary
neuro --version
neuro --help
```

## Architecture

The codebase under `src/` is organized into five stages mirroring the execution pipeline of a `.nr` program:

| Subsystem | Modules | Responsibility |
| :--- | :--- | :--- |
| **Driver** | [`main.rs`](src/main.rs), [`lib.rs`](src/lib.rs) | Command-line interface (`run`, `check`, `info`), file validation, and crate exports. |
| **Frontend** | [`lexer.rs`](src/lexer.rs), [`parser.rs`](src/parser.rs), [`ast.rs`](src/ast.rs) | Indentation-aware tokenization (`INDENT`/`DEDENT`), recursive-descent parsing, AST nodes, and execution directives. |
| **Static Analysis** | [`checker.rs`](src/checker.rs) | Pre-execution validation of variable scopes, mutability (`let` vs `let mut`), reference types, and symbolic tensor shapes. |
| **Runtime & Autodiff** | [`interp.rs`](src/interp.rs), [`autograd.rs`](src/autograd.rs), [`value.rs`](src/value.rs) | Tree-walking evaluation, directive scoping, tape-based reverse-mode backpropagation, and reference-counted runtime values. |
| **Compute Backends** | [`tensor.rs`](src/tensor.rs), [`gpu.rs`](src/gpu.rs) | Dense tensor storage, broadcasting, 8-wide SIMD CPU GEMM, and `wgpu` (Vulkan/DX12) WGSL compute shaders with buffer caching. |
| **Stdlib & Storage** | [`builtins.rs`](src/builtins.rs), [`data.rs`](src/data.rs), [`io.rs`](src/io.rs) | Built-in layers and optimizers, tokenizers, streaming readers, `NRBN` dataset I/O, and `NRMD` checkpoint serialization. |

## Language Guide

### Example

Save the following program as `main.nr` and run it with `neuro run main.nr`:

```python
seed(42)

let w = linear(4, 2)
let x = ones([3, 4])
let target = zeros([3, 2])

# Fuse linear projection and GELU activation into a single kernel
@fuse fn forward(inp: Tensor[B, 4]) -> Tensor[B, 2]:
    return inp |> w |> gelu

for step in 0..10:
    let pred = forward(x)
    let loss = mse(pred, target)
    let mut grads = grad(loss, [w])
    sgd(&mut [w], grads, 0.05)
    zero_grad(&mut grads)
    print("step:", step, "loss:", item(loss))
```

### Variables, Types & Control Flow

- **Bindings**: `let x = 1` defines an immutable binding; `let mut x = 1` (or `mut x = 1`) defines a mutable binding.
- **Type Annotations**: Optional on `let` bindings and function parameters/returns. Supported types are `Tensor`, `Tensor[D1, D2, ...]`, `f32`, `f16`, `i32`, `usize`, `bool`, `str`, `List`, `Layer`, `Tokenizer`, `Any`, and references (`&T`, `&mut T`).
- **Symbolic Dimensions**: Named dimensions in function signatures (e.g., `fn step(x: Tensor[B, D], w: Tensor[D, H]) -> Tensor[B, H]`) are bound on first use and checked for consistency across arguments and calls. Use `_` as a wildcard dimension.
- **Control Flow**: Indentation-delimited blocks for `fn`, `if` / `elif` / `else`, `for x in a..b`, `for x in collection`, and `no_grad:` (disables autograd tape recording during inference).

### Operators

| Operator | Semantics | Example |
| :--- | :--- | :--- |
| `><` | Matrix multiplication (supports 1D vectors, 2D matrices, and batched N-D tensors) | `let c = a >< b` |
| `\|>` | Pipeline application (passes left expression as the first argument to the right call) | `x \|> norm \|> fc1 \|> gelu` |
| `>>` | Moves a tensor, layer, or list of parameters to a target memory space (`ram`, `vram`) | `let g_x = x >> vram` |
| `+`, `-`, `*`, `/`, `%`, `**` | Element-wise arithmetic with NumPy-style broadcasting, or scalar arithmetic | `let z = (x - mean) / std` |
| `+=`, `-=`, `*=`, `/=` | Compound in-place assignment on mutable variables or indexed elements | `total_loss += item(loss)` |
| `&x`, `&mut x` | Immutable and mutable references required by in-place optimizers and gradient helpers | `adam(&mut model, grads, 0.001)` |
| `a..b` | Half-open slice indexing (`..b`, `a..`, `..`, and negative indices supported) | `let m = mask[0..t, 0..t]` |
| `.T`, `.shape`, `.ndim` | Tensor properties: transpose of last two axes, dimension list, and rank | `let xt = x.T` |

### Directives

Directives can prefix a function definition (`@fuse fn ...`), an indented block (`@gpu:`), or a single statement (`@ram let ...`). Multiple directives may be stacked.

| Directive | Behavior |
| :--- | :--- |
| `@cpu` | Routes tensor execution to the multi-threaded CPU backend. |
| `@gpu` | Dispatches matrix multiplications (`><`) to the Vulkan/DX12 compute pipeline. |
| `@npu` | Targets NPU execution (falls back to CPU with a one-time diagnostic notice). |
| `@ram` | Allocates newly created tensors in host system RAM. |
| `@vram` | Allocates newly created tensors in logical VRAM (tracked via `vram_used()`). |
| `@simd` | Enables 8-wide SIMD-vectorized inner loops on CPU GEMM kernels. |
| `@f16_auto` | Quantizes matmul and linear operations using round-to-nearest-even FP16 precision. |
| `@fuse` | Fuses `linear` followed by `relu`, `gelu`, or `sigmoid` into a single forward/backward op. |
| `@sync` | Inserts a full memory synchronization barrier after the decorated statement or block. |

## Standard Library

### Tensor Creation & Manipulation

| Function | Signature / Behavior |
| :--- | :--- |
| `zeros(shape...)` | Allocates a zero-filled `f32` tensor (`zeros([2, 3])` or `zeros(2, 3)`). |
| `ones(shape...)` | Allocates a `f32` tensor filled with `1.0`. |
| `randn(shape...)` | Samples a tensor from a standard normal distribution $\mathcal{N}(0, 1)$. |
| `reshape(x, shape...)` | Returns a tensor sharing `x`'s underlying buffer with a new compatible shape. |
| `concat(list, axis)` | Concatenates a non-empty list of tensors along `axis`. |
| `split(x, parts, [axis])` | Splits `x` into `parts` equal slices along `axis` (defaults to `0`). |
| `sum(x, [axis])` | Sums all elements into a scalar or reduces along `axis`. |
| `mean(x, [axis])` | Computes the arithmetic mean across all elements or along `axis`. |
| `argmax(x, [axis])` | Returns `i32` indices of maximum values along `axis` (defaults to the last axis). |
| `sample(logits, [temp])` | Multinomial sampling from unnormalized logits or probabilities at temperature `temp` (default `1.0`). |
| `free(target)` | Immediately releases the backing buffer of a `Tensor`, `Layer`, or `List` of them. |

### Neural Layers, Activations & Attention

| Function | Signature / Behavior |
| :--- | :--- |
| `linear(in_dim, out_dim)` | Trainable affine layer ($y = xW + b$) initialized with scaled normal weights. |
| `conv2d(in_ch, out_ch, k, [stride], [pad])` | Trainable 2D convolution layer (`k` accepts an integer or `[kh, kw]`). |
| `embed(vocab_size, dim)` | Trainable token embedding lookup table of shape `[vocab_size, dim]`. |
| `embed(ids, weight)` | Functional embedding lookup mapping integer token IDs against `weight`. |
| `pos_embed(max_len, dim)` | Trainable learned positional embedding layer added to sequence inputs. |
| `attention(q, k, v, [mask])` | Scaled dot-product attention $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$. |
| `causal_mask(seq_len)` | Returns a `[seq_len, seq_len]` upper-triangular mask filled with `-1e9`. |
| `relu(x)`, `gelu(x)`, `sigmoid(x)`, `tanh(x)` | Element-wise neural activation functions with analytical backward passes. |
| `exp(x)`, `log(x)`, `sqrt(x)` | Element-wise exponential, natural logarithm, and square root. |
| `softmax(x)` | Numerically stable softmax normalized over the last axis. |
| `norm(x)` | Layer normalization across the last axis ($\epsilon = 10^{-5}$). |
| `dropout(x, rate)` | Inverted dropout with drop probability `rate` in `[0.0, 1.0)` when gradient tracking is active. |

### Losses, Autograd & Optimizers

| Function | Signature / Behavior |
| :--- | :--- |
| `mse(pred, target)` | Element-wise squared error $(pred - target)^2$. |
| `cross_entropy(logits, targets)` | Fused log-softmax and negative log-likelihood loss over unnormalized logits. |
| `grad(loss, params)` | Runs reverse-mode backpropagation from scalar `loss` and returns gradients aligned with `params`. |
| `detach(x)` | Returns a leaf copy of `x` detached from the autograd graph. |
| `clip_grad(&mut grads, max_norm)` | Scales gradients in place if their global $L_2$ norm exceeds `max_norm`; returns the pre-clip norm. |
| `zero_grad(&mut grads)` | Resets all gradient tensors in place to zero. |
| `sgd(&mut params, grads, lr)` | Updates parameters in place via stochastic gradient descent ($\theta \leftarrow \theta - \eta g$). |
| `adam(&mut params, grads, lr, [b1], [b2], [eps])` | Updates parameters in place using Adam (defaults: `b1 = 0.9`, `b2 = 0.999`, `eps = 1e-8`). |

### Data, Tokenizers & Serialization

| Function | Signature / Behavior |
| :--- | :--- |
| `load_text(path)` | Reads a UTF-8 text file into a `str`. |
| `load_bin(path)` | Loads an `NRBN` dataset (or raw little-endian `i32` array) into a 1D tensor. |
| `save_bin(tensor, path)` | Serializes a 1D tensor to disk in `NRBN` format. |
| `stream_data(path, batch_size)` | Returns an iterable stream yielding sequential chunks from a text or `NRBN` file. |
| `tokenizer([path])` | Loads a line-delimited vocabulary file, or returns a byte-level (256-token) tokenizer if omitted. |
| `build_vocab(text, path)` | Writes sorted unique characters from `text` to `path` and returns the vocabulary size. |
| `vocab_size([tok])` | Returns the token count of `tok` (or the active default tokenizer). |
| `encode([tok], text)` | Encodes a `str` into a 1D `i32` tensor of token IDs. |
| `decode([tok], ids)` | Decodes a tensor, list, or integer ID back into a `str`. |
| `batch(data, [b_size], seq_len)` | Slices a 1D token tensor into `[N, seq_len]` or `[N, b_size, seq_len]` batches. |
| `save(model, path)` | Writes a tensor, layer, or nested list of layers to an `NRMD` checkpoint file. |
| `load_model(path)` | Restores a model or tensor checkpoint from `path` into the active memory space. |

### Diagnostics & Utilities

| Function | Signature / Behavior |
| :--- | :--- |
| `print(args...)` | Formats and prints space-separated values to standard output. |
| `item(x)` | Extracts a 1-element `Tensor` into a pure scalar (`i64` or `f64`) detached from autograd. |
| `vram_used()` | Returns currently allocated logical VRAM in megabytes (`MB`). |
| `len(x)` | Returns the element count of a `List`, character count of a `str`, or outer dimension of a `Tensor`. |
| `shape(x)` | Returns the dimensions of tensor `x` as a `List` of integers. |
| `elapsed()` | Returns elapsed wall-clock seconds since interpreter startup. |
| `seed(s)` | Seeds the deterministic `xorshift64*` pseudo-random number generator. |

## Binary Formats

- **`NRBN` (Neuro Binary Data)**: Dataset container starting with a 13-byte header (`"NRBN"` magic bytes, 1-byte dtype tag, little-endian `u64` element count) followed by packed `i32`, `u16`, or `f32` values.
- **`NRMD` (Neuro Model Checkpoint)**: Self-describing binary checkpoint beginning with `"NRMD"` and a format version byte, encoding layer kinds, hyperparameters, tensor shapes, and raw `f32` weights so checkpoints can be restored without reconstructing layer definitions manually.