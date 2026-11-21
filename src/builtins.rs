//! Built-in functions for Neuro: tensor creation, layers, losses, autograd, optimizers and I/O.

use crate::autograd as ag;
use crate::autograd::Var;
use crate::data::{build_vocab, load_bin, save_bin, Stream, Tokenizer};
use crate::interp::{
    const_val, flatten_params, norm_axis, to_f64, to_int, to_list, to_str, to_tensor, to_usize,
    AdamState, Interp,
};
use crate::io::{load_value, save_value};
use crate::tensor::{numel, DType, Mem, Res, Tensor, Un, VRAM_BYTES};
use crate::value::{list, Layer, LayerKind, Value};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::Ordering;

/// Table of `(name, min_args, max_args)` used by both the static checker and runtime.
const SPECS: &[(&str, usize, usize)] = &[
    // Data, books & tokens
    ("load_text", 1, 1),
    ("load_bin", 1, 1),
    ("save_bin", 2, 2),
    ("stream_data", 2, 2),
    ("tokenizer", 0, 1),
    ("build_vocab", 2, 2),
    ("vocab_size", 0, 1),
    ("encode", 1, 2),
    ("decode", 1, 2),
    ("batch", 2, 3),
    // Tensor creation, shape & reductions
    ("zeros", 1, 16),
    ("ones", 1, 16),
    ("randn", 1, 16),
    ("reshape", 2, 16),
    ("concat", 2, 2),
    ("split", 2, 3),
    ("sum", 1, 2),
    ("mean", 1, 2),
    ("argmax", 1, 2),
    ("sample", 1, 2),
    ("free", 1, 1),
    ("save", 2, 2),
    ("load_model", 1, 1),
    // Layers, activations & attention
    ("linear", 2, 2),
    ("conv2d", 3, 5),
    ("embed", 2, 2),
    ("pos_embed", 2, 2),
    ("attention", 3, 4),
    ("causal_mask", 1, 1),
    ("relu", 1, 1),
    ("gelu", 1, 1),
    ("sigmoid", 1, 1),
    ("tanh", 1, 1),
    ("exp", 1, 1),
    ("log", 1, 1),
    ("sqrt", 1, 1),
    ("softmax", 1, 1),
    ("norm", 1, 1),
    ("dropout", 2, 2),
    // Losses, autodiff & optimizers
    ("detach", 1, 1),
    ("mse", 2, 2),
    ("cross_entropy", 2, 2),
    ("grad", 2, 2),
    ("clip_grad", 2, 2),
    ("zero_grad", 1, 1),
    ("sgd", 3, 3),
    ("adam", 3, 6),
    // Diagnostics & utilities
    ("item", 1, 1),
    ("print", 0, 32),
    ("vram_used", 0, 0),
    ("len", 1, 1),
    ("shape", 1, 1),
    ("elapsed", 0, 0),
    ("seed", 1, 1),
];

pub fn spec(name: &str) -> Option<(&'static str, usize, usize)> {
    SPECS.iter().copied().find(|&(n, _, _)| n == name)
}

pub fn builtin_name(name: &str) -> Option<&'static str> {
    spec(name).map(|(n, _, _)| n)
}

fn parse_shape(args: &[Value]) -> Res<Vec<usize>> {
    let items = if args.len() == 1 {
        match &args[0] {
            Value::List(l) => l.borrow().clone(),
            other => vec![other.clone()],
        }
    } else {
        args.to_vec()
    };
    if items.is_empty() {
        return Err("shape cannot be empty".to_string());
    }
    let mut shape = Vec::with_capacity(items.len());
    for it in items {
        shape.push(to_usize(&it)?);
    }
    Ok(shape)
}

fn free_value(v: &Value) -> Res<()> {
    match v {
        Value::Tensor(var) => {
            if !var.freed.get() {
                *var.value.borrow_mut() = Tensor::empty();
                var.freed.set(true);
            }
            Ok(())
        }
        Value::Layer(l) => {
            for p in &l.params {
                if !p.freed.get() {
                    *p.value.borrow_mut() = Tensor::empty();
                    p.freed.set(true);
                }
            }
            Ok(())
        }
        Value::List(l) => {
            let items = l.borrow().clone();
            for it in &items {
                free_value(it)?;
            }
            Ok(())
        }
        other => Err(format!("free() expects a Tensor, Layer or list, got {}", other.type_name())),
    }
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Nil => "nil".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{:.6}", f),
        Value::Str(s) => s.to_string(),
        Value::Mem(m) => format!("@{}", m.name()),
        Value::Dev(d) => format!("@{:?}", d).to_lowercase(),
        Value::List(l) => {
            let items: Vec<String> = l.borrow().iter().map(format_value).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Tensor(var) => match ag::get(var) {
            Ok(t) => {
                if t.numel() == 1 {
                    if t.dtype == DType::I32 {
                        format!("{}", t.data[0] as i64)
                    } else {
                        format!("{:.6}", t.data[0])
                    }
                } else {
                    let preview: Vec<String> = t
                        .data
                        .iter()
                        .take(6)
                        .map(|&x| {
                            if t.dtype == DType::I32 {
                                format!("{}", x as i64)
                            } else {
                                format!("{:.4}", x)
                            }
                        })
                        .collect();
                    let dots = if t.data.len() > 6 { ", ..." } else { "" };
                    format!(
                        "Tensor(shape={:?}, dtype={}, mem={}, [{}{}])",
                        t.shape,
                        t.dtype.name(),
                        t.mem().name(),
                        preview.join(", "),
                        dots
                    )
                }
            }
            Err(_) => "Tensor(<freed>)".to_string(),
        },
        Value::Layer(l) => format!("Layer({:?})", l.kind),
        Value::Fn(f) => format!("<fn {}>", f.name),
        Value::Builtin(b) => format!("<builtin {}>", b),
        Value::Tok(t) => format!("<Tokenizer vocab_size={}>", t.size()),
        Value::Stream(_) => "<Stream>".to_string(),
    }
}

impl Interp {
    fn resolve_tok(&self, arg: Option<&Value>) -> Res<Rc<Tokenizer>> {
        if let Some(Value::Tok(t)) = arg {
            return Ok(t.clone());
        }
        if let Some(t) = &self.tok {
            return Ok(t.clone());
        }
        Err("no tokenizer loaded; call tokenizer(\"vocab.txt\") first".to_string())
    }

    fn argmax_or_sample(&mut self, x: &Var, axis: Option<usize>, temp: Option<f32>) -> Res<Value> {
        let t = ag::get(x)?;
        if t.rank() == 0 {
            return Err("argmax/sample requires a tensor of rank >= 1".to_string());
        }
        let ax = axis.unwrap_or(t.rank() - 1);
        if ax >= t.rank() {
            return Err(format!("axis {} out of range for shape {:?}", ax, t.shape));
        }
        let outer: usize = t.shape[..ax].iter().product();
        let dim = t.shape[ax];
        let inner: usize = t.shape[ax + 1..].iter().product();
        if dim == 0 {
            return Err("cannot reduce along a zero-length axis".to_string());
        }

        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let picked = match temp {
                    None => {
                        let mut best_j = 0usize;
                        let mut best_v = f32::NEG_INFINITY;
                        for j in 0..dim {
                            let v = t.data[(o * dim + j) * inner + i];
                            if v > best_v {
                                best_v = v;
                                best_j = j;
                            }
                        }
                        best_j
                    }
                    Some(tp) if tp <= 1e-6 => {
                        let mut best_j = 0usize;
                        let mut best_v = f32::NEG_INFINITY;
                        for j in 0..dim {
                            let v = t.data[(o * dim + j) * inner + i];
                            if v > best_v {
                                best_v = v;
                                best_j = j;
                            }
                        }
                        best_j
                    }
                    Some(tp) => {
                        let mut vals = Vec::with_capacity(dim);
                        let mut all_nonneg = true;
                        let mut sum_raw = 0.0f32;
                        for j in 0..dim {
                            let v = t.data[(o * dim + j) * inner + i];
                            if v < 0.0 {
                                all_nonneg = false;
                            }
                            sum_raw += v;
                            vals.push(v);
                        }
                        let mut probs = vec![0.0f32; dim];
                        if all_nonneg && (sum_raw - 1.0).abs() < 1e-2 {
                            let inv_t = 1.0 / tp;
                            let mut z = 0.0f32;
                            for j in 0..dim {
                                let p = vals[j].max(1e-12).powf(inv_t);
                                probs[j] = p;
                                z += p;
                            }
                            for p in &mut probs {
                                *p /= z.max(1e-12);
                            }
                        } else {
                            let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut z = 0.0f32;
                            for j in 0..dim {
                                let p = ((vals[j] - mx) / tp).exp();
                                probs[j] = p;
                                z += p;
                            }
                            for p in &mut probs {
                                *p /= z.max(1e-12);
                            }
                        }
                        let u = self.rng.uniform();
                        let mut cdf = 0.0f32;
                        let mut chosen = dim - 1;
                        for (j, &p) in probs.iter().enumerate() {
                            cdf += p;
                            if u <= cdf {
                                chosen = j;
                                break;
                            }
                        }
                        chosen
                    }
                };
                out.push(picked as f32);
            }
        }

        let mut new_shape = t.shape.clone();
        new_shape.remove(ax);
        Ok(const_val(new_shape, out, DType::I32, t.mem()))
    }

    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Res<Value> {
        let (_, min, max) = spec(name).ok_or_else(|| format!("unknown builtin `{}`", name))?;
        if args.len() < min || args.len() > max {
            return Err(format!("{}() expects {}..={} argument(s), got {}", name, min, max, args.len()));
        }

        match name {
            "detach" => {
                let x = to_tensor(&args[0])?;
                Ok(Value::Tensor(ag::detach(&x)?))
            }

        "item" => match &args[0] {
                Value::Int(i) => Ok(Value::Int(*i)),
                Value::Float(f) => Ok(Value::Float(*f)),
                Value::Tensor(x) => {
                    let t = ag::get(x)?;
                    if t.numel() != 1 {
                        return Err(format!("item() expects a scalar tensor, got shape {:?}", t.shape));
                    }
                    if t.dtype == DType::I32 {
                        Ok(Value::Int(t.data[0] as i64))
                    } else {
                        Ok(Value::Float(t.data[0] as f64))
                    }
                }
                other => Err(format!("item() expects a scalar Tensor or number, got {}", other.type_name())),
            },    

            // ------------------------------------------------ Data, books & tokens
            "load_text" => {
                let path = to_str(&args[0])?;
                let text = std::fs::read_to_string(&*path)
                    .map_err(|e| format!("cannot read '{}': {}", path, e))?;
                Ok(Value::Str(Rc::new(text)))
            }
            "load_bin" => {
                let path = to_str(&args[0])?;
                let (data, dtype) = load_bin(&path)?;
                let n = data.len();
                Ok(const_val(vec![n], data, dtype, self.alloc_mem()))
            }
            "save_bin" => {
                let (path, var) = match (&args[0], &args[1]) {
                    (Value::Str(p), Value::Tensor(t)) => (p.clone(), t.clone()),
                    (Value::Tensor(t), Value::Str(p)) => (p.clone(), t.clone()),
                    _ => return Err("save_bin() expects (Tensor, str) or (str, Tensor)".to_string()),
                };
                let t = ag::get(&var)?;
                save_bin(&path, &t.data, t.dtype)?;
                Ok(Value::Nil)
            }
            "stream_data" => {
                let path = to_str(&args[0])?;
                let bs = to_usize(&args[1])?;
                let st = Stream::open(&path, bs)?;
                Ok(Value::Stream(Rc::new(RefCell::new(st))))
            }
            "tokenizer" => {
                let tok = if args.is_empty() {
                    Rc::new(Tokenizer::bytes())
                } else {
                    let path = to_str(&args[0])?;
                    if path.is_empty() || &*path == "bytes" {
                        Rc::new(Tokenizer::bytes())
                    } else {
                        Rc::new(Tokenizer::from_file(&path)?)
                    }
                };
                self.tok = Some(tok.clone());
                Ok(Value::Tok(tok))
            }
            "build_vocab" => {
                let text = to_str(&args[0])?;
                let path = to_str(&args[1])?;
                let sz = build_vocab(&text, &path)?;
                Ok(Value::Int(sz as i64))
            }
            "vocab_size" => {
                let tok = self.resolve_tok(args.first())?;
                Ok(Value::Int(tok.size() as i64))
            }
            "encode" => {
                let (tok, text) = if args.len() == 2 {
                    (self.resolve_tok(Some(&args[0]))?, to_str(&args[1])?)
                } else {
                    (self.resolve_tok(None)?, to_str(&args[0])?)
                };
                let ids = tok.encode(&text)?;
                let data: Vec<f32> = ids.into_iter().map(|i| i as f32).collect();
                let n = data.len();
                Ok(const_val(vec![n], data, DType::I32, self.alloc_mem()))
            }
            "decode" => {
                let (tok, tok_val) = if args.len() == 2 {
                    (self.resolve_tok(Some(&args[0]))?, &args[1])
                } else {
                    (self.resolve_tok(None)?, &args[0])
                };
                let ids: Vec<i32> = match tok_val {
                    Value::Tensor(x) => ag::get(&x)?.data.iter().map(|&v| v as i32).collect(),
                    Value::List(l) => {
                        let items = l.borrow();
                        let mut out = Vec::with_capacity(items.len());
                        for it in items.iter() {
                            out.push(to_int(it)? as i32);
                        }
                        out
                    }
                    Value::Int(i) => vec![*i as i32],
                    other => return Err(format!("decode() expects a Tensor or list of token ids, got {}", other.type_name())),
                };
                let text = tok.decode(&ids)?;
                Ok(Value::Str(Rc::new(text)))
            }
            "batch" => {
                let (raw, mem) = match &args[0] {
                    Value::Tensor(x) => {
                        let t = ag::get(x)?;
                        (t.data.clone(), t.mem())
                    }
                    Value::List(l) => {
                        let items = l.borrow();
                        let mut out = Vec::with_capacity(items.len());
                        for it in items.iter() {
                            out.push(to_f64(it)? as f32);
                        }
                        (out, self.alloc_mem())
                    }
                    other => return Err(format!("batch() expects a Tensor or list, got {}", other.type_name())),
                };
                if args.len() == 2 {
                    let seq_len = to_usize(&args[1])?;
                    if seq_len == 0 {
                        return Err("batch(): seq_len must be > 0".to_string());
                    }
                    let n_seq = raw.len() / seq_len;
                    if n_seq == 0 {
                        return Err(format!("batch(): {} tokens is shorter than seq_len {}", raw.len(), seq_len));
                    }
                    let data = raw[..n_seq * seq_len].to_vec();
                    Ok(const_val(vec![n_seq, seq_len], data, DType::I32, mem))
                } else {
                    let bsize = to_usize(&args[1])?;
                    let seq_len = to_usize(&args[2])?;
                    let block = bsize * seq_len;
                    if block == 0 {
                        return Err("batch(): batch_size and seq_len must be > 0".to_string());
                    }
                    let n_batches = raw.len() / block;
                    if n_batches == 0 {
                        return Err(format!("batch(): {} tokens is shorter than {}x{}", raw.len(), bsize, seq_len));
                    }
                    let data = raw[..n_batches * block].to_vec();
                    Ok(const_val(vec![n_batches, bsize, seq_len], data, DType::I32, mem))
                }
            }

            // ------------------------------------------------ Tensors & I/O
            "zeros" => {
                let shape = parse_shape(&args)?;
                let n = numel(&shape);
                let dt = if self.cx.f16 { DType::F16 } else { DType::F32 };
                let t = Tensor::new(shape, vec![0.0; n], dt, self.alloc_mem());
                Ok(Value::Tensor(ag::leaf(t, self.grad_on)))
            }
            "ones" => {
                let shape = parse_shape(&args)?;
                let n = numel(&shape);
                let dt = if self.cx.f16 { DType::F16 } else { DType::F32 };
                let t = Tensor::new(shape, vec![1.0; n], dt, self.alloc_mem());
                Ok(Value::Tensor(ag::leaf(t, self.grad_on)))
            }
            "randn" => {
                let shape = parse_shape(&args)?;
                let n = numel(&shape);
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    data.push(self.rng.normal());
                }
                let dt = if self.cx.f16 { DType::F16 } else { DType::F32 };
                let t = Tensor::new(shape, data, dt, self.alloc_mem());
                Ok(Value::Tensor(ag::leaf(t, self.grad_on)))
            }
            "reshape" => {
                let x = to_tensor(&args[0])?;
                let shape = parse_shape(&args[1..])?;
                Ok(Value::Tensor(ag::reshape_v(&x, shape, self.opcx())?))
            }
            "concat" => {
                let items = to_list(&args[0])?;
                let mut vars = Vec::with_capacity(items.len());
                for it in &items {
                    vars.push(to_tensor(it)?);
                }
                if vars.is_empty() {
                    return Err("concat(): empty tensor list".to_string());
                }
                let rank = ag::get(&vars[0])?.rank();
                let ax = norm_axis(to_int(&args[1])?, rank)?;
                Ok(Value::Tensor(ag::concat_v(&vars, ax, self.opcx())?))
            }
            "split" => {
                let x = to_tensor(&args[0])?;
                let parts = to_usize(&args[1])?;
                if parts == 0 {
                    return Err("split(): parts must be >= 1".to_string());
                }
                let shape = ag::shape_of(&x)?;
                let ax = if args.len() == 3 {
                    norm_axis(to_int(&args[2])?, shape.len())?
                } else {
                    0
                };
                if shape.is_empty() || shape[ax] % parts != 0 {
                    return Err(format!("split(): axis {} (len {:?}) is not divisible by {}", ax, shape.get(ax), parts));
                }
                let step = shape[ax] / parts;
                let cx = self.opcx();
                let mut out = Vec::with_capacity(parts);
                for p in 0..parts {
                    let mut ranges: Vec<(usize, usize)> = shape.iter().map(|&d| (0, d)).collect();
                    ranges[ax] = (p * step, (p + 1) * step);
                    out.push(Value::Tensor(ag::slice_v(&x, ranges, cx)?));
                }
                Ok(list(out))
            }
            "sum" => {
                let x = to_tensor(&args[0])?;
                let ax = if args.len() == 2 {
                    let rank = ag::get(&x)?.rank();
                    Some(norm_axis(to_int(&args[1])?, rank)?)
                } else {
                    None
                };
                Ok(Value::Tensor(ag::sum_v(&x, ax, self.opcx())?))
            }
            "mean" => {
                let x = to_tensor(&args[0])?;
                let ax = if args.len() == 2 {
                    let rank = ag::get(&x)?.rank();
                    Some(norm_axis(to_int(&args[1])?, rank)?)
                } else {
                    None
                };
                Ok(Value::Tensor(ag::mean_v(&x, ax, self.opcx())?))
            }
            "argmax" => {
                let x = to_tensor(&args[0])?;
                let ax = if args.len() == 2 {
                    let rank = ag::get(&x)?.rank();
                    Some(norm_axis(to_int(&args[1])?, rank)?)
                } else {
                    None
                };
                self.argmax_or_sample(&x, ax, None)
            }
            "sample" => {
                let x = to_tensor(&args[0])?;
                let temp = if args.len() == 2 { to_f64(&args[1])? as f32 } else { 1.0 };
                self.argmax_or_sample(&x, None, Some(temp))
            }
            "free" => {
                free_value(&args[0])?;
                Ok(Value::Nil)
            }
            "save" => {
                let path = to_str(&args[1])?;
                save_value(&args[0], &path)?;
                Ok(Value::Nil)
            }
            "load_model" => {
                let path = to_str(&args[0])?;
                load_value(&path, self.alloc_mem())
            }

            // ------------------------------------------------ Layers, activations & attention
            "linear" => {
                let inp = to_usize(&args[0])?;
                let out = to_usize(&args[1])?;
                if inp == 0 || out == 0 {
                    return Err("linear(): dimensions must be > 0".to_string());
                }
                let scale = (1.0_f32 / inp as f32).sqrt();
                let mut wdata = Vec::with_capacity(inp * out);
                for _ in 0..(inp * out) {
                    wdata.push(self.rng.normal() * scale);
                }
                let mem = self.alloc_mem();
                let w = ag::leaf(Tensor::new(vec![inp, out], wdata, DType::F32, mem), true);
                let b = ag::leaf(Tensor::new(vec![out], vec![0.0; out], DType::F32, mem), true);
                Ok(Value::Layer(Rc::new(Layer {
                    kind: LayerKind::Linear { inp, out },
                    params: vec![w, b],
                })))
            }
            "conv2d" => {
                let inp = to_usize(&args[0])?;
                let out = to_usize(&args[1])?;
                let (kh, kw) = match &args[2] {
                    Value::List(l) => {
                        let items = l.borrow();
                        if items.len() != 2 {
                            return Err("conv2d(): kernel list must have 2 elements [kh, kw]".to_string());
                        }
                        (to_usize(&items[0])?, to_usize(&items[1])?)
                    }
                    other => {
                        let k = to_usize(other)?;
                        (k, k)
                    }
                };
                let stride = if args.len() >= 4 { to_usize(&args[3])? } else { 1 };
                let pad = if args.len() >= 5 { to_usize(&args[4])? } else { 0 };
                let fan_in = (inp * kh * kw).max(1) as f32;
                let scale = (1.0_f32 / fan_in).sqrt();
                let n_w = out * inp * kh * kw;
                let mut wdata = Vec::with_capacity(n_w);
                for _ in 0..n_w {
                    wdata.push(self.rng.normal() * scale);
                }
                let mem = self.alloc_mem();
                let w = ag::leaf(Tensor::new(vec![out, inp, kh, kw], wdata, DType::F32, mem), true);
                let b = ag::leaf(Tensor::new(vec![out], vec![0.0; out], DType::F32, mem), true);
                Ok(Value::Layer(Rc::new(Layer {
                    kind: LayerKind::Conv2d { inp, out, kh, kw, stride, pad },
                    params: vec![w, b],
                })))
            }
            "embed" => {
                // Supports both `embed(vocab_size, dim)` -> Layer and `embed(ids, &w_embed)` -> Tensor.
                if let Value::Tensor(w) = &args[1] {
                    let idx = match &args[0] {
                        Value::Tensor(t) => ag::get(t)?.clone(),
                        Value::Int(i) => Tensor::scalar(*i as f32, DType::I32, Mem::Ram),
                        other => return Err(format!("embed(): expected token ids, got {}", other.type_name())),
                    };
                    return Ok(Value::Tensor(ag::gather_v(w, idx, self.opcx())?));
                }
                let vocab = to_usize(&args[0])?;
                let dim = to_usize(&args[1])?;
                if vocab == 0 || dim == 0 {
                    return Err("embed(): vocab_size and dim must be > 0".to_string());
                }
                let scale = (1.0_f32 / dim as f32).sqrt();
                let mut wdata = Vec::with_capacity(vocab * dim);
                for _ in 0..(vocab * dim) {
                    wdata.push(self.rng.normal() * scale);
                }
                let w = ag::leaf(Tensor::new(vec![vocab, dim], wdata, DType::F32, self.alloc_mem()), true);
                Ok(Value::Layer(Rc::new(Layer {
                    kind: LayerKind::Embed { vocab, dim },
                    params: vec![w],
                })))
            }
            "pos_embed" => {
                let len = to_usize(&args[0])?;
                let dim = to_usize(&args[1])?;
                let mut wdata = Vec::with_capacity(len * dim);
                for _ in 0..(len * dim) {
                    wdata.push(self.rng.normal() * 0.02);
                }
                let w = ag::leaf(Tensor::new(vec![len, dim], wdata, DType::F32, self.alloc_mem()), true);
                Ok(Value::Layer(Rc::new(Layer {
                    kind: LayerKind::PosEmbed { len, dim },
                    params: vec![w],
                })))
            }
            "attention" => {
                let q = to_tensor(&args[0])?;
                let k = to_tensor(&args[1])?;
                let v = to_tensor(&args[2])?;
                let q_shape = ag::shape_of(&q)?;
                let k_shape = ag::shape_of(&k)?;
                if q_shape.len() < 2 || k_shape.len() < 2 {
                    return Err("attention(): q, k, v must have rank >= 2".to_string());
                }
                let d_k = *k_shape.last().unwrap() as f32;
                let cx = self.opcx();
                let kt = ag::swap_v(&k, k_shape.len() - 2, k_shape.len() - 1, cx)?;
                let mut scores = ag::matmul_v(&q, &kt, cx)?;
                scores = ag::mul_s(&scores, 1.0 / d_k.max(1.0).sqrt(), cx)?;
                if args.len() == 4 {
                    let mask = to_tensor(&args[3])?;
                    scores = ag::bin(ag::Op::Add, &scores, &mask, cx)?;
                }
                let weights = ag::softmax_v(&scores, cx)?;
                Ok(Value::Tensor(ag::matmul_v(&weights, &v, cx)?))
            }
            "causal_mask" => {
                let seq_len = to_usize(&args[0])?;
                let mut data = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        data[i * seq_len + j] = -1e9;
                    }
                }
                Ok(const_val(vec![seq_len, seq_len], data, DType::F32, self.alloc_mem()))
            }
            "relu" => Ok(Value::Tensor(ag::un(Un::Relu, &to_tensor(&args[0])?, self.opcx())?)),
            "gelu" => Ok(Value::Tensor(ag::un(Un::Gelu, &to_tensor(&args[0])?, self.opcx())?)),
            "sigmoid" => Ok(Value::Tensor(ag::un(Un::Sigmoid, &to_tensor(&args[0])?, self.opcx())?)),
            "tanh" => Ok(Value::Tensor(ag::un(Un::Tanh, &to_tensor(&args[0])?, self.opcx())?)),
            "exp" => Ok(Value::Tensor(ag::un(Un::Exp, &to_tensor(&args[0])?, self.opcx())?)),
            "log" => Ok(Value::Tensor(ag::un(Un::Log, &to_tensor(&args[0])?, self.opcx())?)),
            "sqrt" => Ok(Value::Tensor(ag::un(Un::Sqrt, &to_tensor(&args[0])?, self.opcx())?)),
            "softmax" => Ok(Value::Tensor(ag::softmax_v(&to_tensor(&args[0])?, self.opcx())?)),
            "norm" => Ok(Value::Tensor(ag::layernorm_v(&to_tensor(&args[0])?, self.opcx())?)),
            "dropout" => {
                let x = to_tensor(&args[0])?;
                let rate = to_f64(&args[1])? as f32;
                if !(0.0..1.0).contains(&rate) {
                    return Err("dropout(): rate must be in [0.0, 1.0)".to_string());
                }
                if !self.grad_on || rate == 0.0 {
                    return Ok(Value::Tensor(x));
                }
                let (shape, mem) = {
                    let t = ag::get(&x)?;
                    (t.shape.clone(), t.mem())
                };
                let n = numel(&shape);
                let scale = 1.0 / (1.0 - rate);
                let mut mask = Vec::with_capacity(n);
                for _ in 0..n {
                    mask.push(if self.rng.uniform() >= rate { scale } else { 0.0 });
                }
                let mvar = ag::leaf(Tensor::new(shape, mask, DType::F32, mem), false);
                Ok(Value::Tensor(ag::bin(ag::Op::Mul, &x, &mvar, self.opcx())?))
            }

            // ------------------------------------------------ Losses, autodiff & optimizers
            "mse" => {
                let pred = to_tensor(&args[0])?;
                let target = to_tensor(&args[1])?;
                let cx = self.opcx();
                let diff = ag::bin(ag::Op::Sub, &pred, &target, cx)?;
                let sq = ag::bin(ag::Op::Mul, &diff, &diff, cx)?;
                Ok(Value::Tensor(ag::mean_v(&sq, None, cx)?))
            }
            "cross_entropy" => {
                let logits = to_tensor(&args[0])?;
                if matches!(logits.op, ag::Op::Softmax) {
                    eprintln!("[neuro] warning: cross_entropy() already applies softmax internally; avoid passing softmax() output directly");
                }
                let target = match &args[1] {
                    Value::Tensor(t) => ag::get(t)?.clone(),
                    Value::List(l) => {
                        let items = l.borrow();
                        let mut data = Vec::with_capacity(items.len());
                        for it in items.iter() {
                            data.push(to_f64(it)? as f32);
                        }
                        let n = data.len();
                        Tensor::new(vec![n], data, DType::I32, ag::get(&logits)?.mem())
                    }
                    other => return Err(format!("cross_entropy(): target must be a Tensor or list, got {}", other.type_name())),
                };
                Ok(Value::Tensor(ag::ce_v(&logits, target, self.opcx())?))
            }
            "grad" => {
                let loss = to_tensor(&args[0])?;
                let params = flatten_params(&args[1])?;
                let grads = ag::backward(&loss, &params, self.opcx())?;
                let out: Vec<Value> = grads.into_iter().map(|g| Value::Tensor(ag::leaf(g, false))).collect();
                Ok(list(out))
            }
            "clip_grad" => {
                let grads = flatten_params(&args[0])?;
                let max_norm = to_f64(&args[1])? as f32;
                if max_norm <= 0.0 {
                    return Err("clip_grad(): max_norm must be > 0".to_string());
                }
                let mut sq_sum = 0.0f64;
                for g in &grads {
                    let t = ag::get(g)?;
                    for &v in &t.data {
                        sq_sum += (v as f64) * (v as f64);
                    }
                }
                let total_norm = sq_sum.sqrt() as f32;
                if total_norm > max_norm {
                    let scale = max_norm / total_norm;
                    for g in &grads {
                        let mut t = g.value.borrow_mut();
                        for v in &mut t.data {
                            *v *= scale;
                        }
                    }
                }
                Ok(Value::Float(total_norm as f64))
            }
            "zero_grad" => {
                let grads = flatten_params(&args[0])?;
                for g in &grads {
                    if !g.freed.get() {
                        let mut t = g.value.borrow_mut();
                        for v in &mut t.data {
                            *v = 0.0;
                        }
                    }
                }
                Ok(Value::Nil)
            }
            "sgd" => {
                let params = flatten_params(&args[0])?;
                let grads = flatten_params(&args[1])?;
                let lr = to_f64(&args[2])? as f32;
                if params.len() != grads.len() {
                    return Err(format!("sgd(): {} parameter(s) but {} gradient(s)", params.len(), grads.len()));
                }
                for (p, g) in params.iter().zip(grads.iter()) {
                    let gt = ag::get(g)?;
                    let mut pt = p.value.borrow_mut();
                    if pt.shape != gt.shape {
                        return Err(format!("sgd(): shape mismatch {:?} vs {:?}", pt.shape, gt.shape));
                    }
                    for (w, &dw) in pt.data.iter_mut().zip(gt.data.iter()) {
                        *w -= lr * dw;
                    }
                }
                Ok(Value::Nil)
            }
            "adam" => {
                let params = flatten_params(&args[0])?;
                let grads = flatten_params(&args[1])?;
                let lr = to_f64(&args[2])? as f32;
                let b1 = if args.len() >= 4 { to_f64(&args[3])? as f32 } else { 0.9 };
                let b2 = if args.len() >= 5 { to_f64(&args[4])? as f32 } else { 0.999 };
                let eps = if args.len() >= 6 { to_f64(&args[5])? as f32 } else { 1e-8 };
                if params.len() != grads.len() {
                    return Err(format!("adam(): {} parameter(s) but {} gradient(s)", params.len(), grads.len()));
                }
                for (p, g) in params.iter().zip(grads.iter()) {
                    let gt = ag::get(g)?;
                    let mut pt = p.value.borrow_mut();
                    if pt.shape != gt.shape {
                        return Err(format!("adam(): shape mismatch {:?} vs {:?}", pt.shape, gt.shape));
                    }
                    let n = pt.data.len();
                    let st = self.adam.entry(p.id).or_insert_with(|| AdamState {
                        m: vec![0.0; n],
                        v: vec![0.0; n],
                        t: 0,
                    });
                    if st.m.len() != n {
                        st.m = vec![0.0; n];
                        st.v = vec![0.0; n];
                        st.t = 0;
                    }
                    st.t += 1;
                    let bc1 = 1.0 - b1.powi(st.t as i32);
                    let bc2 = 1.0 - b2.powi(st.t as i32);
                    for i in 0..n {
                        let grad_i = gt.data[i];
                        st.m[i] = b1 * st.m[i] + (1.0 - b1) * grad_i;
                        st.v[i] = b2 * st.v[i] + (1.0 - b2) * grad_i * grad_i;
                        let m_hat = st.m[i] / bc1;
                        let v_hat = st.v[i] / bc2;
                        pt.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                    }
                }
                Ok(Value::Nil)
            }

            // ------------------------------------------------ Diagnostics & utilities
            "print" => {
                let parts: Vec<String> = args.iter().map(format_value).collect();
                println!("{}", parts.join(" "));
                Ok(Value::Nil)
            }
            "vram_used" => {
                let bytes = VRAM_BYTES.load(Ordering::Relaxed);
                Ok(Value::Float(bytes as f64 / (1024.0 * 1024.0)))
            }
            "len" => match &args[0] {
                Value::List(l) => Ok(Value::Int(l.borrow().len() as i64)),
                Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                Value::Tensor(x) => {
                    let sh = ag::shape_of(x)?;
                    Ok(Value::Int(sh.first().copied().unwrap_or(0) as i64))
                }
                other => Err(format!("len() not supported for {}", other.type_name())),
            },
            "shape" => {
                let x = to_tensor(&args[0])?;
                let sh = ag::shape_of(&x)?;
                Ok(list(sh.into_iter().map(|d| Value::Int(d as i64)).collect()))
            }
            "elapsed" => Ok(Value::Float(self.start.elapsed().as_secs_f64())),
            "seed" => {
                let s = to_int(&args[0])? as u64;
                self.rng = crate::tensor::Rng::new(s);
                Ok(Value::Nil)
            }
            _ => Err(format!("builtin `{}` not implemented", name)),
        }
    }
}