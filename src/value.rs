//! Runtime values of the interpreter.

use crate::ast::FnDef;
use crate::autograd::Var;
use crate::data::{Stream, Tokenizer};
use crate::tensor::{DType, Device, Mem};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub enum Value {
    Nil,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(Rc<String>),
    List(Rc<RefCell<Vec<Value>>>),
    Tensor(Var),
    Fn(Rc<FnDef>),
    Layer(Rc<Layer>),
    Tok(Rc<Tokenizer>),
    Stream(Rc<RefCell<Stream>>),
    Builtin(&'static str),
    Mem(Mem),
    Dev(Device),
}

#[derive(Debug)]
pub enum LayerKind {
    Linear { inp: usize, out: usize },
    Conv2d { inp: usize, out: usize, kh: usize, kw: usize, stride: usize, pad: usize },
    Embed { vocab: usize, dim: usize },
    PosEmbed { len: usize, dim: usize },
}

/// A parametric layer. Parameters: Linear/Conv2d = [weight, bias], Embed/PosEmbed = [weight].
pub struct Layer {
    pub kind: LayerKind,
    pub params: Vec<Var>,
}

pub fn list(items: Vec<Value>) -> Value {
    Value::List(Rc::new(RefCell::new(items)))
}

pub fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{:.1}", v);
    }
    if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e9) {
        return format!("{:e}", v);
    }
    let s = format!("{:.6}", v);
    let t = s.trim_end_matches('0');
    if t.ends_with('.') {
        format!("{}0", t)
    } else {
        t.to_string()
    }
}

impl LayerKind {
    pub fn describe(&self) -> String {
        match self {
            LayerKind::Linear { inp, out } => format!("linear({} -> {})", inp, out),
            LayerKind::Conv2d { inp, out, kh, kw, stride, pad } => {
                format!("conv2d({} -> {}, kernel {}x{}, stride {}, pad {})", inp, out, kh, kw, stride, pad)
            }
            LayerKind::Embed { vocab, dim } => format!("embed({}, {})", vocab, dim),
            LayerKind::PosEmbed { len, dim } => format!("pos_embed({}, {})", len, dim),
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Int(_) => "integer",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Tensor(_) => "Tensor",
            Value::Fn(_) => "function",
            Value::Layer(_) => "layer",
            Value::Tok(_) => "tokenizer",
            Value::Stream(_) => "stream",
            Value::Builtin(_) => "builtin",
            Value::Mem(_) => "memory space",
            Value::Dev(_) => "device",
        }
    }

    /// Human-readable form used by `print` and `str`.
    pub fn show(&self) -> String {
        match self {
            Value::Nil => "nil".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => fmt_f64(*f),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => s.as_ref().clone(),
            Value::List(l) => {
                let parts: Vec<String> = l.borrow().iter().map(|v| v.show_nested()).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Tensor(v) => {
                if v.freed.get() {
                    return "<freed tensor>".to_string();
                }
                let t = v.value.borrow();
                if t.numel() == 1 && t.rank() == 0 {
                    return if t.dtype == DType::I32 { format!("{}", t.data[0] as i64) } else { fmt_f64(t.data[0] as f64) };
                }
                let shown: Vec<String> = t
                    .data
                    .iter()
                    .take(8)
                    .map(|&x| if t.dtype == DType::I32 { format!("{}", x as i64) } else { fmt_f64(x as f64) })
                    .collect();
                let more = if t.numel() > 8 { ", ..." } else { "" };
                format!("Tensor{:?} ({}, {}) [{}{}]", t.shape, t.dtype.name(), t.mem().name(), shown.join(", "), more)
            }
            Value::Fn(f) => format!("<fn {}>", f.name),
            Value::Layer(l) => format!("<layer {}>", l.kind.describe()),
            Value::Tok(t) => format!("<tokenizer, vocab {}>", t.size()),
            Value::Stream(_) => "<stream>".to_string(),
            Value::Builtin(n) => format!("<builtin {}>", n),
            Value::Mem(m) => m.name().to_string(),
            Value::Dev(d) => match d {
                Device::Cpu => "cpu".to_string(),
                Device::Gpu => "gpu".to_string(),
                Device::Npu => "npu".to_string(),
            },
        }
    }

    fn show_nested(&self) -> String {
        match self {
            Value::Str(s) => format!("\"{}\"", s),
            other => other.show(),
        }
    }
}
