//! Tree-walking interpreter: executes `.nr` programs directly (no transpiling).

use crate::ast::*;
use crate::autograd as ag;
use crate::autograd::{Cx, Var};
use crate::builtins::builtin_name;
use crate::data::{StreamItem, Tokenizer};
use crate::tensor::*;
use crate::value::*;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

pub struct Slot {
    pub value: Value,
    pub mutable: bool,
    /// Parameter declared `&mut T`: assignments write through to the caller's object.
    pub by_mut_ref: bool,
}

pub struct Frame {
    pub vars: HashMap<String, Slot>,
    /// Symbolic tensor dimensions bound by annotations such as `Tensor[B, D]`.
    pub dims: HashMap<String, usize>,
}

pub enum Flow {
    Next,
    Return(Value),
}

/// Hardware / allocation context set by directives.
#[derive(Clone, Copy)]
pub struct RunCtx {
    pub device: Device,
    pub mem: Option<Mem>,
    pub simd: bool,
    pub f16: bool,
    pub fuse: bool,
}

pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: u32,
}

pub enum Ix {
    At(i64),
    Rg(Option<i64>, Option<i64>),
}

pub struct Interp {
    pub globals: HashMap<String, Slot>,
    pub frames: Vec<Frame>,
    pub cx: RunCtx,
    pub grad_on: bool,
    pub rng: Rng,
    pub tok: Option<Rc<Tokenizer>>,
    pub adam: HashMap<u64, AdamState>,
    pub warned: HashSet<&'static str>,
    pub start: Instant,
}

// ---------------------------------------------------------------- conversions

pub fn to_int(v: &Value) -> Res<i64> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Float(f) => {
            if f.is_finite() && *f == f.trunc() {
                Ok(*f as i64)
            } else {
                Err(format!("expected an integer, got {}", fmt_f64(*f)))
            }
        }
        Value::Tensor(x) => {
            let t = ag::get(x)?;
            if t.numel() == 1 && t.data[0] == t.data[0].trunc() {
                Ok(t.data[0] as i64)
            } else {
                Err(format!("expected an integer, got a tensor of shape {:?}", t.shape))
            }
        }
        other => Err(format!("expected an integer, got {}", other.type_name())),
    }
}

pub fn to_usize(v: &Value) -> Res<usize> {
    let i = to_int(v)?;
    if i < 0 {
        return Err(format!("expected a non-negative integer, got {}", i));
    }
    Ok(i as usize)
}

pub fn to_f64(v: &Value) -> Res<f64> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        Value::Tensor(x) => {
            let t = ag::get(x)?;
            if t.numel() == 1 {
                Ok(t.data[0] as f64)
            } else {
                Err(format!("expected a number, got a tensor of shape {:?}", t.shape))
            }
        }
        other => Err(format!("expected a number, got {}", other.type_name())),
    }
}

pub fn to_str(v: &Value) -> Res<Rc<String>> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(format!("expected a str, got {}", other.type_name())),
    }
}

pub fn to_tensor(v: &Value) -> Res<Var> {
    match v {
        Value::Tensor(x) => Ok(x.clone()),
        other => Err(format!("expected a Tensor, got {}", other.type_name())),
    }
}

pub fn to_list(v: &Value) -> Res<Vec<Value>> {
    match v {
        Value::List(l) => {
            let items = l.borrow().clone();
            Ok(items)
        }
        other => Err(format!("expected a list, got {}", other.type_name())),
    }
}

pub fn truthy(v: &Value) -> Res<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(*i != 0),
        Value::Float(f) => Ok(*f != 0.0),
        Value::Nil => Ok(false),
        Value::Str(s) => Ok(!s.is_empty()),
        Value::List(l) => {
            let n = l.borrow().len();
            Ok(n > 0)
        }
        Value::Tensor(_) => Ok(to_f64(v)? != 0.0),
        other => Err(format!("cannot use {} as a condition", other.type_name())),
    }
}

pub fn norm_axis(ax: i64, rank: usize) -> Res<usize> {
    let r = rank as i64;
    let a = if ax < 0 { ax + r } else { ax };
    if a < 0 || a >= r {
        return Err(format!("axis {} out of range for rank {}", ax, rank));
    }
    Ok(a as usize)
}

pub fn norm_at(i: i64, len: usize) -> Res<usize> {
    let k = if i < 0 { i + len as i64 } else { i };
    if k < 0 || k >= len as i64 {
        return Err(format!("index {} out of range for length {}", i, len));
    }
    Ok(k as usize)
}

pub fn norm_range(lo: Option<i64>, hi: Option<i64>, len: usize) -> Res<(usize, usize)> {
    let fix = |x: i64| if x < 0 { x + len as i64 } else { x };
    let s = fix(lo.unwrap_or(0));
    let e = fix(hi.unwrap_or(len as i64));
    if s < 0 || e < s || e > len as i64 {
        return Err(format!(
            "slice [{}..{}] out of range for length {}",
            lo.map(|v| v.to_string()).unwrap_or_default(),
            hi.map(|v| v.to_string()).unwrap_or_default(),
            len
        ));
    }
    Ok((s as usize, e as usize))
}

/// Non-differentiable tensor value.
pub fn const_val(shape: Vec<usize>, data: Vec<f32>, dtype: DType, mem: Mem) -> Value {
    Value::Tensor(ag::leaf(Tensor::new(shape, data, dtype, mem), false))
}

/// Flattens tensors, layers and (nested) lists into their parameter tensors.
pub fn flatten_params(v: &Value) -> Res<Vec<Var>> {
    match v {
        Value::Tensor(x) => Ok(vec![x.clone()]),
        Value::Layer(l) => Ok(l.params.clone()),
        Value::List(l) => {
            let items = l.borrow().clone();
            let mut out = Vec::new();
            for it in items.iter() {
                out.extend(flatten_params(it)?);
            }
            Ok(out)
        }
        other => Err(format!("expected a tensor, layer or list of them, got {}", other.type_name())),
    }
}

fn expected(what: &str, v: &Value) -> String {
    format!("expected {}, got {}", what, v.type_name())
}

fn check_type(t: &Type, v: &Value, dims: &mut HashMap<String, usize>) -> Res<()> {
    match t {
        Type::Ref { inner, .. } => check_type(inner, v, dims),
        Type::F32 | Type::F16 => match v {
            Value::Int(_) | Value::Float(_) => Ok(()),
            _ => Err(expected("a number", v)),
        },
        Type::I32 => match v {
            Value::Int(i) => {
                if *i > i32::MAX as i64 || *i < i32::MIN as i64 {
                    Err(format!("{} does not fit in i32", i))
                } else {
                    Ok(())
                }
            }
            _ => Err(expected("an i32", v)),
        },
        Type::Usize => match v {
            Value::Int(i) => {
                if *i < 0 {
                    Err(format!("usize cannot be negative ({})", i))
                } else {
                    Ok(())
                }
            }
            _ => Err(expected("a usize", v)),
        },
        Type::Bool => match v {
            Value::Bool(_) => Ok(()),
            _ => Err(expected("a bool", v)),
        },
        Type::Str => match v {
            Value::Str(_) => Ok(()),
            _ => Err(expected("a str", v)),
        },
        Type::Tensor(spec) => match v {
            Value::Tensor(x) => {
                if let Some(spec) = spec {
                    let shape = ag::shape_of(x)?;
                    if spec.len() != shape.len() {
                        return Err(format!("expected a tensor of rank {}, got shape {:?}", spec.len(), shape));
                    }
                    for (s, &d) in spec.iter().zip(shape.iter()) {
                        match s {
                            DimSpec::Lit(n) => {
                                if *n != d {
                                    return Err(format!("expected dimension {}, got {} (shape {:?})", n, d, shape));
                                }
                            }
                            DimSpec::Name(nm) => {
                                if nm == "_" {
                                    continue;
                                }
                                match dims.get(nm) {
                                    Some(&bound) => {
                                        if bound != d {
                                            return Err(format!(
                                                "dimension `{}` is {} here but was bound to {} earlier (shape {:?})",
                                                nm, d, bound, shape
                                            ));
                                        }
                                    }
                                    None => {
                                        dims.insert(nm.clone(), d);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => Err(expected("a Tensor", v)),
        },
        Type::Named(n) => match (n.as_str(), v) {
            ("List", Value::List(_)) | ("Layer", Value::Layer(_)) | ("Tokenizer", Value::Tok(_)) | ("Any", _) => Ok(()),
            _ => Err(expected(n, v)),
        },
    }
}

fn scalar_var(v: &Value, mem: Mem) -> Res<Var> {
    match v {
        Value::Int(i) => Ok(ag::konst(*i as f32, DType::I32, mem)),
        Value::Float(f) => Ok(ag::konst(*f as f32, DType::F32, mem)),
        other => Err(format!("cannot combine a Tensor with {}", other.type_name())),
    }
}

/// `a >< b` that also accepts vectors (rank 1 operands are promoted, then squeezed).
fn matmul_any(a: &Var, b: &Var, cx: Cx) -> Res<Var> {
    let sa = ag::shape_of(a)?;
    let sb = ag::shape_of(b)?;
    let a2 = if sa.len() == 1 { ag::reshape_v(a, vec![1, sa[0]], cx)? } else { a.clone() };
    let b2 = if sb.len() == 1 { ag::reshape_v(b, vec![sb[0], 1], cx)? } else { b.clone() };
    let r = ag::matmul_v(&a2, &b2, cx)?;
    if sa.len() == 1 || sb.len() == 1 {
        let mut shape = ag::shape_of(&r)?;
        if sb.len() == 1 {
            shape.pop();
        }
        if sa.len() == 1 {
            let idx = if sb.len() == 1 { shape.len() - 1 } else { shape.len() - 2 };
            shape.remove(idx);
        }
        return ag::reshape_v(&r, shape, cx);
    }
    Ok(r)
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Nil, Value::Nil) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            if Rc::ptr_eq(x, y) {
                return true;
            }
            let (xb, yb) = (x.borrow(), y.borrow());
            xb.len() == yb.len() && xb.iter().zip(yb.iter()).all(|(u, v)| values_eq(u, v))
        }
        _ => match (to_f64(a), to_f64(b)) {
            (Ok(x), Ok(y)) => x == y,
            _ => false,
        },
    }
}

// ---------------------------------------------------------------- interpreter

impl Interp {
    pub fn new() -> Interp {
        Interp {
            globals: HashMap::new(),
            frames: Vec::new(),
            cx: RunCtx { device: Device::Cpu, mem: None, simd: false, f16: false, fuse: false },
            grad_on: true,
            rng: Rng::new(1234),
            tok: None,
            adam: HashMap::new(),
            warned: HashSet::new(),
            start: Instant::now(),
        }
    }

    pub fn run(&mut self, prog: &[Stmt]) -> Res<()> {
        // Hoist top-level functions so they can be called before their definition.
        for s in prog {
            if let StmtKind::FnDef(f) = &s.kind {
                self.declare(&f.name, Slot { value: Value::Fn(f.clone()), mutable: false, by_mut_ref: false });
            }
        }
        self.exec_block(prog)?;
        Ok(())
    }

    // ------------------------------------------------------------ context helpers

    pub fn opcx(&self) -> Cx {
        Cx { ex: Exec { device: self.cx.device, simd: self.cx.simd, f16: self.cx.f16 }, grad: self.grad_on }
    }

    /// Memory space for freshly allocated tensors.
    pub fn alloc_mem(&self) -> Mem {
        match self.cx.mem {
            Some(m) => m,
            None => {
                if self.cx.device == Device::Gpu {
                    Mem::Vram
                } else {
                    Mem::Ram
                }
            }
        }
    }

    fn note(&mut self, key: &'static str, msg: &str) {
        if self.warned.insert(key) {
            eprintln!("[neuro] note: {}", msg);
        }
    }

    fn apply_dirs(&mut self, names: &[String]) {
        for n in names {
            match n.as_str() {
                "cpu" => self.cx.device = Device::Cpu,
                "gpu" => {
                    self.cx.device = Device::Gpu;
                    if let Some(name) = crate::gpu::gpu_device_name() {
                        if self.warned.insert("gpu") {
                            eprintln!("[neuro] active Vulkan GPU: {}", name);
                        }
                    } else {
                        self.note("gpu", "no Vulkan GPU detected; falling back to CPU");
                    }
                }

                "npu" => {
                    self.cx.device = Device::Npu;
                    self.note("npu", "no NPU available; @npu regions fall back to the CPU backend");
                }
                "ram" => self.cx.mem = Some(Mem::Ram),
                "vram" => self.cx.mem = Some(Mem::Vram),
                "simd" => self.cx.simd = true,
                "f16_auto" => self.cx.f16 = true,
                "fuse" => self.cx.fuse = true,
                _ => {}
            }
        }
    }

    /// `@sync`: every kernel is synchronous on the CPU backend, so this is a full memory fence.
    fn sync(&mut self) {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    // ------------------------------------------------------------ scopes

    pub fn lookup(&self, n: &str) -> Option<&Slot> {
        if let Some(f) = self.frames.last() {
            if let Some(s) = f.vars.get(n) {
                return Some(s);
            }
        }
        self.globals.get(n)
    }

    fn declare(&mut self, name: &str, slot: Slot) {
        if let Some(f) = self.frames.last_mut() {
            f.vars.insert(name.to_string(), slot);
        } else {
            self.globals.insert(name.to_string(), slot);
        }
    }

    fn set_var(&mut self, name: &str, v: Value) {
        if let Some(f) = self.frames.last_mut() {
            if let Some(s) = f.vars.get_mut(name) {
                s.value = v;
                return;
            }
        }
        if let Some(s) = self.globals.get_mut(name) {
            s.value = v;
        }
    }

    // ------------------------------------------------------------ statements

    pub fn exec_block(&mut self, body: &[Stmt]) -> Res<Flow> {
        for s in body {
            if let Flow::Return(v) = self.exec_stmt(s)? {
                return Ok(Flow::Return(v));
            }
        }
        Ok(Flow::Next)
    }

    fn exec_stmt(&mut self, s: &Stmt) -> Res<Flow> {
        match self.exec_inner(s) {
            Ok(f) => Ok(f),
            Err(e) => {
                if e.starts_with("line ") {
                    Err(e)
                } else {
                    Err(format!("line {}: {}", s.line, e))
                }
            }
        }
    }

    fn exec_inner(&mut self, s: &Stmt) -> Res<Flow> {
        match &s.kind {
            StmtKind::Let { name, mutable, ty, value } => {
                let v = self.eval(value)?;
                if let Some(t) = ty {
                    let mut dims = match self.frames.last() {
                        Some(f) => f.dims.clone(),
                        None => HashMap::new(),
                    };
                    check_type(t, &v, &mut dims).map_err(|e| format!("`{}`: {}", name, e))?;
                    if let Some(f) = self.frames.last_mut() {
                        f.dims = dims;
                    }
                }
                self.declare(name, Slot { value: v, mutable: *mutable, by_mut_ref: false });
                Ok(Flow::Next)
            }
            StmtKind::Assign { target, value } => {
                let v = self.eval(value)?;
                self.assign(target, v)?;
                Ok(Flow::Next)
            }
            StmtKind::Expr(e) => {
                self.eval(e)?;
                Ok(Flow::Next)
            }
            StmtKind::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e)?,
                    None => Value::Nil,
                };
                Ok(Flow::Return(v))
            }
            StmtKind::If { arms, els } => {
                for (c, body) in arms {
                    let cv = self.eval(c)?;
                    if truthy(&cv)? {
                        return self.exec_block(body);
                    }
                }
                if let Some(b) = els {
                    return self.exec_block(b);
                }
                Ok(Flow::Next)
            }
            StmtKind::For { var, iter, body } => self.exec_for(var, iter, body),
            StmtKind::NoGrad(body) => {
                let saved = self.grad_on;
                self.grad_on = false;
                let r = self.exec_block(body);
                self.grad_on = saved;
                r
            }
            StmtKind::FnDef(f) => {
                self.declare(&f.name, Slot { value: Value::Fn(f.clone()), mutable: false, by_mut_ref: false });
                Ok(Flow::Next)
            }
            StmtKind::Directive { names, body } => {
                let saved = self.cx;
                self.apply_dirs(names);
                let r = self.exec_block(body);
                self.cx = saved;
                let flow = r?;
                if names.iter().any(|n| n == "sync") {
                    self.sync();
                }
                Ok(flow)
            }
            StmtKind::Sync => {
                self.sync();
                Ok(Flow::Next)
            }
        }
    }

    fn assign(&mut self, target: &Expr, v: Value) -> Res<()> {
        match target {
            Expr::Var(name) => {
                let (mutable, by_ref, old) = match self.lookup(name) {
                    Some(s) => (s.mutable, s.by_mut_ref, s.value.clone()),
                    None => return Err(format!("assignment to undeclared variable `{}` (declare it with let/mut)", name)),
                };
                if !mutable {
                    return Err(format!("cannot assign to immutable variable `{}` (declare it with `mut`)", name));
                }
                if by_ref {
                    return self.write_through(&old, &v);
                }
                self.set_var(name, v);
                Ok(())
            }
            Expr::Index(base, items) => {
                let bv = self.eval(base)?;
                if let Value::List(l) = bv {
                    if items.len() == 1 {
                        if let IdxItem::At(ie) = &items[0] {
                            let iv = self.eval(ie)?;
                            let i = to_int(&iv)?;
                            let len = l.borrow().len();
                            let k = norm_at(i, len)?;
                            l.borrow_mut()[k] = v;
                            return Ok(());
                        }
                    }
                }
                Err("only list elements can be assigned with `l[i] = v`".to_string())
            }
            _ => Err("invalid assignment target".to_string()),
        }
    }

    /// Assignment to a `&mut` parameter: update the caller's tensor/list in place.
    fn write_through(&mut self, old: &Value, new: &Value) -> Res<()> {
        match (old, new) {
            (Value::Tensor(o), Value::Tensor(n)) => {
                if Rc::ptr_eq(o, n) {
                    return Ok(());
                }
                let dmem = ag::get(o)?.mem();
                let nt = {
                    let r = ag::get(n)?;
                    r.with_mem(dmem)
                };
                let mut dst = o.value.borrow_mut();
                if dst.shape != nt.shape {
                    return Err(format!("in-place update needs matching shapes ({:?} vs {:?})", dst.shape, nt.shape));
                }
                *dst = nt;
                Ok(())
            }
            (Value::List(o), Value::List(n)) => {
                if !Rc::ptr_eq(o, n) {
                    let items = n.borrow().clone();
                    *o.borrow_mut() = items;
                }
                Ok(())
            }
            _ => Err(format!(
                "`&mut` parameters can only be reassigned with a Tensor or list (got {} -> {})",
                old.type_name(),
                new.type_name()
            )),
        }
    }

    fn bind_loop(&mut self, var: &str, v: Value) {
        self.declare(var, Slot { value: v, mutable: false, by_mut_ref: false });
    }

    fn exec_for(&mut self, var: &str, iter: &ForIter, body: &[Stmt]) -> Res<Flow> {
        match iter {
            ForIter::Range(a, b) => {
                let av = self.eval(a)?;
                let bv = self.eval(b)?;
                let lo = to_int(&av)?;
                let hi = to_int(&bv)?;
                let mut i = lo;
                while i < hi {
                    self.bind_loop(var, Value::Int(i));
                    if let Flow::Return(v) = self.exec_block(body)? {
                        return Ok(Flow::Return(v));
                    }
                    i += 1;
                }
                Ok(Flow::Next)
            }
            ForIter::Expr(e) => {
                let it = self.eval(e)?;
                match it {
                    Value::List(l) => {
                        let items = l.borrow().clone();
                        for item in items {
                            self.bind_loop(var, item);
                            if let Flow::Return(v) = self.exec_block(body)? {
                                return Ok(Flow::Return(v));
                            }
                        }
                        Ok(Flow::Next)
                    }
                    Value::Stream(s) => loop {
                        let item = s.borrow_mut().next()?;
                        let val = match item {
                            Some(StreamItem::Text(t)) => Value::Str(Rc::new(t)),
                            Some(StreamItem::Tokens(data, dtype)) => {
                                let n = data.len();
                                const_val(vec![n], data, dtype, self.alloc_mem())
                            }
                            None => return Ok(Flow::Next),
                        };
                        self.bind_loop(var, val);
                        if let Flow::Return(v) = self.exec_block(body)? {
                            return Ok(Flow::Return(v));
                        }
                    },
                    Value::Tensor(x) => {
                        let shape = ag::shape_of(&x)?;
                        if shape.is_empty() {
                            return Err("cannot iterate over a scalar tensor".to_string());
                        }
                        for i in 0..shape[0] {
                            let row = self.index_tensor(&x, &[Ix::At(i as i64)])?;
                            self.bind_loop(var, row);
                            if let Flow::Return(v) = self.exec_block(body)? {
                                return Ok(Flow::Return(v));
                            }
                        }
                        Ok(Flow::Next)
                    }
                    other => Err(format!("cannot iterate over {}", other.type_name())),
                }
            }
        }
    }

    // ------------------------------------------------------------ functions

    pub fn call_value(&mut self, cv: Value, args: Vec<Value>) -> Res<Value> {
        match cv {
            Value::Fn(f) => self.call_fn(&f, args),
            Value::Builtin(name) => self.call_builtin(name, args),
            Value::Layer(l) => {
                if args.len() != 1 {
                    return Err(format!("a layer takes exactly 1 argument, got {}", args.len()));
                }
                let x = args.into_iter().next().unwrap();
                self.apply_layer(&l, x)
            }
            other => Err(format!("{} is not callable", other.type_name())),
        }
    }

    pub fn call_fn(&mut self, f: &Rc<FnDef>, args: Vec<Value>) -> Res<Value> {
        if args.len() != f.params.len() {
            return Err(format!("fn {} expects {} argument(s), got {}", f.name, f.params.len(), args.len()));
        }
        if self.frames.len() >= 1000 {
            return Err("maximum call depth exceeded".to_string());
        }
        let mut dims: HashMap<String, usize> = HashMap::new();
        let mut vars: HashMap<String, Slot> = HashMap::new();
        for (p, a) in f.params.iter().zip(args.into_iter()) {
            let mut by_mut = false;
            if let Some(t) = &p.ty {
                check_type(t, &a, &mut dims).map_err(|e| format!("argument `{}` of {}: {}", p.name, f.name, e))?;
                if let Type::Ref { mutable: true, .. } = t {
                    by_mut = true;
                }
            }
            vars.insert(p.name.clone(), Slot { value: a, mutable: by_mut, by_mut_ref: by_mut });
        }
        self.frames.push(Frame { vars, dims });
        let saved = self.cx;
        self.apply_dirs(&f.dirs);
        let r = self.exec_block(&f.body);
        self.cx = saved;
        let frame = self.frames.pop();
        let flow = r?;
        let ret = match flow {
            Flow::Return(v) => v,
            Flow::Next => Value::Nil,
        };
        if let Some(t) = &f.ret {
            let mut dims = match frame {
                Some(fr) => fr.dims,
                None => HashMap::new(),
            };
            check_type(t, &ret, &mut dims).map_err(|e| format!("return value of {}: {}", f.name, e))?;
        }
        if f.dirs.iter().any(|n| n == "sync") {
            self.sync();
        }
        Ok(ret)
    }

    // ------------------------------------------------------------ expressions

    pub fn eval(&mut self, e: &Expr) -> Res<Value> {
        match e {
            Expr::Int(i) => Ok(Value::Int(*i)),
            Expr::Float(f) => Ok(Value::Float(*f)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Str(s) => Ok(Value::Str(Rc::new(s.clone()))),
            Expr::Var(n) => self.eval_var(n),
            Expr::List(items) => {
                let mut v = Vec::with_capacity(items.len());
                for it in items {
                    v.push(self.eval(it)?);
                }
                Ok(list(v))
            }
            Expr::Unary(op, x) => {
                let v = self.eval(x)?;
                self.unary(*op, v)
            }
            Expr::Ref(_, x) => self.eval(x),
            Expr::Bin(op, l, r) => self.eval_bin(*op, l, r),
            Expr::Call(c, args) => self.eval_call(c, args),
            Expr::Index(b, items) => self.eval_index(b, items),
            Expr::Field(b, name) => {
                let v = self.eval(b)?;
                self.field(v, name)
            }
        }
    }

    fn eval_var(&mut self, n: &str) -> Res<Value> {
        if let Some(s) = self.lookup(n) {
            return Ok(s.value.clone());
        }
        match n {
            "ram" => Ok(Value::Mem(Mem::Ram)),
            "vram" => Ok(Value::Mem(Mem::Vram)),
            "cpu" => Ok(Value::Dev(Device::Cpu)),
            "gpu" => Ok(Value::Dev(Device::Gpu)),
            "npu" => Ok(Value::Dev(Device::Npu)),
            _ => match builtin_name(n) {
                Some(b) => Ok(Value::Builtin(b)),
                None => Err(format!("undefined variable `{}`", n)),
            },
        }
    }

    fn unary(&mut self, op: UnOp, v: Value) -> Res<Value> {
        match (op, v) {
            (UnOp::Not, v) => Ok(Value::Bool(!truthy(&v)?)),
            (UnOp::Neg, Value::Int(i)) => Ok(Value::Int(i.wrapping_neg())),
            (UnOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
            (UnOp::Neg, Value::Tensor(x)) => {
                let cx = self.opcx();
                Ok(Value::Tensor(ag::un(Un::Neg, &x, cx)?))
            }
            (UnOp::Neg, other) => Err(format!("cannot negate {}", other.type_name())),
        }
    }

    fn eval_bin(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Res<Value> {
        match op {
            BinOp::And => {
                let a = self.eval(l)?;
                if !truthy(&a)? {
                    return Ok(Value::Bool(false));
                }
                let b = self.eval(r)?;
                Ok(Value::Bool(truthy(&b)?))
            }
            BinOp::Or => {
                let a = self.eval(l)?;
                if truthy(&a)? {
                    return Ok(Value::Bool(true));
                }
                let b = self.eval(r)?;
                Ok(Value::Bool(truthy(&b)?))
            }
            BinOp::Pipe => self.eval_pipe(l, r),
            _ => {
                let a = self.eval(l)?;
                let b = self.eval(r)?;
                self.binop(op, a, b)
            }
        }
    }

    // ------------------------------------------------------------ pipelines and @fuse

    fn act_of(&self, e: &Expr) -> Option<Un> {
        if let Expr::Var(n) = e {
            if self.lookup(n).is_none() {
                return match n.as_str() {
                    "relu" => Some(Un::Relu),
                    "gelu" => Some(Un::Gelu),
                    "sigmoid" => Some(Un::Sigmoid),
                    _ => None,
                };
            }
        }
        None
    }

    fn linear_of(&self, e: &Expr) -> Option<Rc<Layer>> {
        if let Expr::Var(n) = e {
            if let Some(s) = self.lookup(n) {
                if let Value::Layer(l) = &s.value {
                    if matches!(l.kind, LayerKind::Linear { .. }) {
                        return Some(l.clone());
                    }
                }
            }
        }
        None
    }

    fn eval_pipe(&mut self, l: &Expr, r: &Expr) -> Res<Value> {
        // `x |> linear_layer |> gelu` under @fuse becomes a single fused kernel.
        if self.cx.fuse {
            if let (Some(act), Expr::Bin(BinOp::Pipe, ll, lr)) = (self.act_of(r), l) {
                if let Some(layer) = self.linear_of(lr) {
                    let x = self.eval(ll)?;
                    return self.apply_linear(&layer, x, Some(act));
                }
            }
        }
        let lv = self.eval(l)?;
        match r {
            Expr::Call(c, args) => {
                let cv = self.eval(c)?;
                let mut vals = vec![lv];
                for a in args {
                    vals.push(self.eval(a)?);
                }
                self.call_value(cv, vals)
            }
            other => {
                let rv = self.eval(other)?;
                self.call_value(rv, vec![lv])
            }
        }
    }

    fn eval_call(&mut self, callee: &Expr, args: &[Expr]) -> Res<Value> {
        // `act(linear_layer(x))` under @fuse becomes a single fused kernel.
        if self.cx.fuse && args.len() == 1 {
            if let Some(act) = self.act_of(callee) {
                if let Expr::Call(ic, iargs) = &args[0] {
                    if iargs.len() == 1 {
                        if let Some(layer) = self.linear_of(ic) {
                            let x = self.eval(&iargs[0])?;
                            return self.apply_linear(&layer, x, Some(act));
                        }
                    }
                }
            }
        }
        let cv = self.eval(callee)?;
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval(a)?);
        }
        self.call_value(cv, vals)
    }

    // ------------------------------------------------------------ layers

    pub fn apply_linear(&mut self, l: &Rc<Layer>, x: Value, act: Option<Un>) -> Res<Value> {
        let cx = self.opcx();
        let xv = to_tensor(&x)?;
        let b = l.params.get(1);
        Ok(Value::Tensor(ag::linear_v(&xv, &l.params[0], b, act, cx)?))
    }

    pub fn apply_layer(&mut self, l: &Rc<Layer>, x: Value) -> Res<Value> {
        let cx = self.opcx();
        match &l.kind {
            LayerKind::Linear { .. } => self.apply_linear(l, x, None),
            LayerKind::Conv2d { stride, pad, .. } => {
                let xv = to_tensor(&x)?;
                let shape = ag::shape_of(&xv)?;
                let batched = shape.len() == 4;
                if !batched && shape.len() != 3 {
                    return Err(format!("conv2d expects [N,C,H,W] or [C,H,W], got shape {:?}", shape));
                }
                let input = if batched {
                    xv
                } else {
                    let mut s = vec![1usize];
                    s.extend_from_slice(&shape);
                    ag::reshape_v(&xv, s, cx)?
                };
                let out = ag::conv2d_v(&input, &l.params[0], l.params.get(1), *stride, *pad, cx)?;
                if batched {
                    Ok(Value::Tensor(out))
                } else {
                    let os = ag::shape_of(&out)?;
                    Ok(Value::Tensor(ag::reshape_v(&out, os[1..].to_vec(), cx)?))
                }
            }
            LayerKind::Embed { .. } => {
                let idx = match &x {
                    Value::Tensor(t) => {
                        let g = ag::get(t)?;
                        g.clone()
                    }
                    Value::Int(i) => Tensor::scalar(*i as f32, DType::I32, Mem::Ram),
                    other => return Err(format!("embedding expects token ids, got {}", other.type_name())),
                };
                Ok(Value::Tensor(ag::gather_v(&l.params[0], idx, cx)?))
            }
            LayerKind::PosEmbed { len, dim } => {
                let xv = to_tensor(&x)?;
                let shape = ag::shape_of(&xv)?;
                if shape.len() < 2 {
                    return Err(format!("pos_embed expects [.., T, D], got shape {:?}", shape));
                }
                let (t, d) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                if t > *len {
                    return Err(format!("sequence length {} exceeds pos_embed capacity {}", t, len));
                }
                if d != *dim {
                    return Err(format!("pos_embed has dim {} but input has dim {}", dim, d));
                }
                let pe = ag::slice_v(&l.params[0], vec![(0, t), (0, d)], cx)?;
                Ok(Value::Tensor(ag::bin(ag::Op::Add, &xv, &pe, cx)?))
            }
        }
    }

    // ------------------------------------------------------------ operators

    fn binop(&mut self, op: BinOp, a: Value, b: Value) -> Res<Value> {
        match op {
            BinOp::Shift => return self.transfer(a, b),
            BinOp::Eq => return Ok(Value::Bool(values_eq(&a, &b))),
            BinOp::Ne => return Ok(Value::Bool(!values_eq(&a, &b))),
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                let x = to_f64(&a)?;
                let y = to_f64(&b)?;
                let r = match op {
                    BinOp::Lt => x < y,
                    BinOp::Gt => x > y,
                    BinOp::Le => x <= y,
                    _ => x >= y,
                };
                return Ok(Value::Bool(r));
            }
            _ => {}
        }
        if matches!(a, Value::Tensor(_)) || matches!(b, Value::Tensor(_)) {
            return self.tensor_bin(op, a, b);
        }
        match (&a, &b) {
            (Value::Str(x), Value::Str(y)) if op == BinOp::Add => Ok(Value::Str(Rc::new(format!("{}{}", x, y)))),
            (Value::List(x), Value::List(y)) if op == BinOp::Add => {
                let mut items = x.borrow().clone();
                items.extend(y.borrow().iter().cloned());
                Ok(list(items))
            }
            _ => num_bin(op, &a, &b),
        }
    }

    fn tensor_bin(&mut self, op: BinOp, a: Value, b: Value) -> Res<Value> {
        let cx = self.opcx();
        if op == BinOp::Pow {
            let x = match &a {
                Value::Tensor(x) => x.clone(),
                _ => return Err("`**` needs a tensor base when the exponent is a tensor".to_string()),
            };
            let p = match &b {
                Value::Int(_) | Value::Float(_) => to_f64(&b)?,
                _ => return Err("tensor exponent must be a number".to_string()),
            };
            return Ok(Value::Tensor(ag::pow_s(&x, p as f32, cx)?));
        }
        let (va, vb) = match (&a, &b) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            (Value::Tensor(x), other) => {
                let m = ag::get(x)?.mem();
                (x.clone(), scalar_var(other, m)?)
            }
            (other, Value::Tensor(y)) => {
                let m = ag::get(y)?.mem();
                (scalar_var(other, m)?, y.clone())
            }
            _ => return Err("internal: tensor_bin called without tensors".to_string()),
        };
        let r = match op {
            BinOp::Add => ag::bin(ag::Op::Add, &va, &vb, cx)?,
            BinOp::Sub => ag::bin(ag::Op::Sub, &va, &vb, cx)?,
            BinOp::Mul => ag::bin(ag::Op::Mul, &va, &vb, cx)?,
            BinOp::Div => ag::bin(ag::Op::Div, &va, &vb, cx)?,
            BinOp::MatMul => matmul_any(&va, &vb, cx)?,
            other => return Err(format!("operator {:?} is not supported on tensors", other)),
        };
        Ok(Value::Tensor(r))
    }

    /// `x >> vram`, `x >> ram`, `x >> gpu|cpu|npu`.
    fn transfer(&mut self, v: Value, target: Value) -> Res<Value> {
        let mem = match target {
            Value::Mem(m) => m,
            Value::Dev(Device::Gpu) => Mem::Vram,
            Value::Dev(_) => Mem::Ram,
            other => return Err(format!("`>>` target must be ram, vram, cpu, gpu or npu, got {}", other.type_name())),
        };
        if let Value::Tensor(x) = &v {
            let cx = self.opcx();
            return Ok(Value::Tensor(ag::transfer_v(x, mem, cx)?));
        }
        if matches!(v, Value::Layer(_) | Value::List(_)) {
            move_params(&v, mem)?;
            return Ok(v);
        }
        Err(format!("cannot move {} between memories", v.type_name()))
    }

    fn field(&mut self, v: Value, name: &str) -> Res<Value> {
        match (&v, name) {
            (Value::Tensor(x), "T") => {
                let cx = self.opcx();
                let rank = ag::get(x)?.rank();
                if rank < 2 {
                    return Ok(v.clone());
                }
                Ok(Value::Tensor(ag::swap_v(x, rank - 2, rank - 1, cx)?))
            }
            (Value::Tensor(x), "shape") => {
                let sh = ag::shape_of(x)?;
                Ok(list(sh.into_iter().map(|d| Value::Int(d as i64)).collect()))
            }
            (Value::Tensor(x), "ndim") => {
                let rank = ag::get(x)?.rank();
                Ok(Value::Int(rank as i64))
            }
            _ => Err(format!("{} has no field `{}`", v.type_name(), name)),
        }
    }

    // ------------------------------------------------------------ indexing

    fn eval_index(&mut self, base: &Expr, items: &[IdxItem]) -> Res<Value> {
        let bv = self.eval(base)?;
        let mut ixs: Vec<Ix> = Vec::new();
        for it in items {
            match it {
                IdxItem::At(e) => {
                    let v = self.eval(e)?;
                    ixs.push(Ix::At(to_int(&v)?));
                }
                IdxItem::Range(a, b) => {
                    let lo = match a {
                        Some(e) => {
                            let v = self.eval(e)?;
                            Some(to_int(&v)?)
                        }
                        None => None,
                    };
                    let hi = match b {
                        Some(e) => {
                            let v = self.eval(e)?;
                            Some(to_int(&v)?)
                        }
                        None => None,
                    };
                    ixs.push(Ix::Rg(lo, hi));
                }
            }
        }
        match bv {
            Value::List(l) => {
                if ixs.len() != 1 {
                    return Err("lists take exactly one index".to_string());
                }
                let len = l.borrow().len();
                match &ixs[0] {
                    Ix::At(i) => {
                        let k = norm_at(*i, len)?;
                        let item = l.borrow()[k].clone();
                        Ok(item)
                    }
                    Ix::Rg(lo, hi) => {
                        let (s, e) = norm_range(*lo, *hi, len)?;
                        let part = l.borrow()[s..e].to_vec();
                        Ok(list(part))
                    }
                }
            }
            Value::Str(s) => {
                if ixs.len() != 1 {
                    return Err("strings take exactly one index".to_string());
                }
                let chars: Vec<char> = s.chars().collect();
                match &ixs[0] {
                    Ix::At(i) => {
                        let k = norm_at(*i, chars.len())?;
                        Ok(Value::Str(Rc::new(chars[k].to_string())))
                    }
                    Ix::Rg(lo, hi) => {
                        let (a, b) = norm_range(*lo, *hi, chars.len())?;
                        Ok(Value::Str(Rc::new(chars[a..b].iter().collect())))
                    }
                }
            }
            Value::Tensor(x) => self.index_tensor(&x, &ixs),
            other => Err(format!("cannot index into {}", other.type_name())),
        }
    }

    pub fn index_tensor(&mut self, x: &Var, ixs: &[Ix]) -> Res<Value> {
        let shape = ag::shape_of(x)?;
        if ixs.len() > shape.len() {
            return Err(format!("too many indices ({}) for a tensor of shape {:?}", ixs.len(), shape));
        }
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(shape.len());
        let mut squeeze: Vec<usize> = Vec::new();
        for (d, &len) in shape.iter().enumerate() {
            match ixs.get(d) {
                None => ranges.push((0, len)),
                Some(Ix::At(i)) => {
                    let k = norm_at(*i, len)?;
                    ranges.push((k, k + 1));
                    squeeze.push(d);
                }
                Some(Ix::Rg(lo, hi)) => {
                    let r = norm_range(*lo, *hi, len)?;
                    ranges.push(r);
                }
            }
        }
        let cx = self.opcx();
        let sliced = ag::slice_v(x, ranges.clone(), cx)?;
        if squeeze.is_empty() {
            return Ok(Value::Tensor(sliced));
        }
        let new_shape: Vec<usize> = ranges
            .iter()
            .enumerate()
            .filter(|(d, _)| !squeeze.contains(d))
            .map(|(_, &(s, e))| e - s)
            .collect();
        Ok(Value::Tensor(ag::reshape_v(&sliced, new_shape, cx)?))
    }
}

/// In-place memory move for layers and lists of parameters.
fn move_params(v: &Value, mem: Mem) -> Res<()> {
    match v {
        Value::Tensor(p) => {
            let t = {
                let r = ag::get(p)?;
                r.with_mem(mem)
            };
            *p.value.borrow_mut() = t;
            Ok(())
        }
        Value::Layer(l) => {
            for p in l.params.iter() {
                let t = {
                    let r = ag::get(p)?;
                    r.with_mem(mem)
                };
                *p.value.borrow_mut() = t;
            }
            Ok(())
        }
        Value::List(l) => {
            let items = l.borrow().clone();
            for it in items.iter() {
                move_params(it, mem)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn num_bin(op: BinOp, a: &Value, b: &Value) -> Res<Value> {
    if let (Value::Int(x), Value::Int(y)) = (a, b) {
        return match op {
            BinOp::Add => Ok(Value::Int(x.wrapping_add(*y))),
            BinOp::Sub => Ok(Value::Int(x.wrapping_sub(*y))),
            BinOp::Mul => Ok(Value::Int(x.wrapping_mul(*y))),
            BinOp::Div => {
                if *y == 0 {
                    Err("division by zero".to_string())
                } else {
                    Ok(Value::Int(x.wrapping_div(*y)))
                }
            }
            BinOp::Mod => {
                if *y == 0 {
                    Err("modulo by zero".to_string())
                } else {
                    Ok(Value::Int(x.wrapping_rem(*y)))
                }
            }
            BinOp::Pow => {
                if *y >= 0 {
                    let e = u32::try_from(*y).map_err(|_| "exponent too large".to_string())?;
                    Ok(Value::Int(x.wrapping_pow(e)))
                } else {
                    Ok(Value::Float((*x as f64).powf(*y as f64)))
                }
            }
            _ => Err(format!("operator {:?} is not defined for integers", op)),
        };
    }
    let x = to_f64(a).map_err(|_| format!("operator {:?} is not defined for {} and {}", op, a.type_name(), b.type_name()))?;
    let y = to_f64(b).map_err(|_| format!("operator {:?} is not defined for {} and {}", op, a.type_name(), b.type_name()))?;
    match op {
        BinOp::Add => Ok(Value::Float(x + y)),
        BinOp::Sub => Ok(Value::Float(x - y)),
        BinOp::Mul => Ok(Value::Float(x * y)),
        BinOp::Div => Ok(Value::Float(x / y)),
        BinOp::Mod => Ok(Value::Float(x % y)),
        BinOp::Pow => Ok(Value::Float(x.powf(y))),
        _ => Err(format!("operator {:?} is not defined for numbers", op)),
    }
}
