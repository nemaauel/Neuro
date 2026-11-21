//! Parse-time checker: scopes, mutability, arity, annotations and best-effort shape inference.
//! Anything it cannot prove statically is left to the runtime shape checks.

use crate::ast::*;
use crate::builtins::spec;
use crate::tensor::Res;
use std::collections::{HashMap, HashSet};

type Dim = Option<i64>;

#[derive(Clone, Debug)]
enum Ty {
    Unknown,
    Int(Option<i64>),
    Float,
    Bool,
    Str,
    /// A list of ints (usable as a shape).
    Shape(Vec<Dim>),
    List,
    Tensor(Option<Vec<Dim>>),
    Layer,
    Fn,
}

#[derive(Clone)]
struct Info {
    mutable: bool,
    by_ref: bool,
    ty: Ty,
}

struct FnSig {
    params: Vec<Param>,
    ret: Option<Type>,
}

struct Checker {
    errors: Vec<String>,
    scopes: Vec<HashMap<String, Info>>,
    globals_all: HashSet<String>,
    fns: HashMap<String, FnSig>,
    in_fn: bool,
    line: usize,
    ret: Option<Type>,
}

pub fn check(prog: &[Stmt]) -> Res<()> {
    let mut c = Checker {
        errors: Vec::new(),
        scopes: vec![HashMap::new()],
        globals_all: HashSet::new(),
        fns: HashMap::new(),
        in_fn: false,
        line: 0,
        ret: None,
    };
    c.collect_globals(prog);
    c.block(prog);
    if c.errors.is_empty() {
        Ok(())
    } else {
        Err(c.errors.join("\n"))
    }
}

fn ty_name(t: &Ty) -> String {
    match t {
        Ty::Unknown => "unknown".to_string(),
        Ty::Int(_) => "integer".to_string(),
        Ty::Float => "float".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Str => "str".to_string(),
        Ty::Shape(_) | Ty::List => "list".to_string(),
        Ty::Tensor(Some(d)) => format!("Tensor{}", fmt_dims(d)),
        Ty::Tensor(None) => "Tensor".to_string(),
        Ty::Layer => "layer".to_string(),
        Ty::Fn => "function".to_string(),
    }
}

fn fmt_dims(d: &[Dim]) -> String {
    let parts: Vec<String> = d
        .iter()
        .map(|x| match x {
            Some(v) => v.to_string(),
            None => "?".to_string(),
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn strip(t: &Type) -> &Type {
    match t {
        Type::Ref { inner, .. } => strip(inner),
        other => other,
    }
}

fn type_to_ty(t: &Type) -> Ty {
    match strip(t) {
        Type::Tensor(Some(dims)) => Ty::Tensor(Some(
            dims.iter()
                .map(|d| match d {
                    DimSpec::Lit(n) => Some(*n as i64),
                    DimSpec::Name(_) => None,
                })
                .collect(),
        )),
        Type::Tensor(None) => Ty::Tensor(None),
        Type::F32 | Type::F16 => Ty::Float,
        Type::I32 | Type::Usize => Ty::Int(None),
        Type::Bool => Ty::Bool,
        Type::Str => Ty::Str,
        Type::Named(n) if n == "List" => Ty::List,
        Type::Named(n) if n == "Layer" => Ty::Layer,
        _ => Ty::Unknown,
    }
}

/// Forget constants for mutable variables: they may change between statements.
fn widen(t: Ty) -> Ty {
    match t {
        Ty::Int(_) => Ty::Int(None),
        Ty::Tensor(_) => Ty::Tensor(None),
        Ty::Shape(_) => Ty::List,
        other => other,
    }
}

impl Checker {
    fn error(&mut self, msg: String) {
        self.errors.push(format!("line {}: {}", self.line, msg));
    }

    fn declare(&mut self, name: &str, info: Info) {
        if let Some(s) = self.scopes.last_mut() {
            s.insert(name.to_string(), info);
        }
    }

    fn lookup(&self, n: &str) -> Option<Info> {
        if let Some(s) = self.scopes.last() {
            if let Some(i) = s.get(n) {
                return Some(i.clone());
            }
        }
        if self.in_fn && self.globals_all.contains(n) {
            let ty = if self.fns.contains_key(n) { Ty::Fn } else { Ty::Unknown };
            return Some(Info { mutable: true, by_ref: false, ty });
        }
        None
    }

    fn collect_globals(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match &s.kind {
                StmtKind::Let { name, .. } => {
                    self.globals_all.insert(name.clone());
                }
                StmtKind::FnDef(f) => {
                    self.globals_all.insert(f.name.clone());
                    self.fns.insert(f.name.clone(), FnSig { params: f.params.clone(), ret: f.ret.clone() });
                }
                StmtKind::For { var, body, .. } => {
                    self.globals_all.insert(var.clone());
                    self.collect_globals(body);
                }
                StmtKind::If { arms, els } => {
                    for (_, b) in arms {
                        self.collect_globals(b);
                    }
                    if let Some(b) = els {
                        self.collect_globals(b);
                    }
                }
                StmtKind::NoGrad(b) => self.collect_globals(b),
                StmtKind::Directive { body, .. } => self.collect_globals(body),
                _ => {}
            }
        }
    }

    fn block(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        self.line = s.line;
        match &s.kind {
            StmtKind::Let { name, mutable, ty, value } => {
                let vt = self.expr(value);
                if let Some(t) = ty {
                    self.compat(t, &vt, &format!("`{}`", name));
                }
                let vt = if *mutable { widen(vt) } else { vt };
                self.declare(name, Info { mutable: *mutable, by_ref: false, ty: vt });
            }
            StmtKind::Assign { target, value } => {
                let vt = self.expr(value);
                match target {
                    Expr::Var(n) => match self.lookup(n) {
                        None => self.error(format!("assignment to undeclared variable `{}`", n)),
                        Some(info) => {
                            if !info.mutable && !info.by_ref {
                                self.error(format!("cannot assign to immutable variable `{}` (declare it with `mut`)", n));
                            } else if let Some(sc) = self.scopes.last_mut() {
                                if let Some(slot) = sc.get_mut(n) {
                                    slot.ty = widen(vt);
                                }
                            }
                        }
                    },
                    other => {
                        self.expr(other);
                    }
                }
            }
            StmtKind::Expr(e) => {
                self.expr(e);
            }
            StmtKind::Return(e) => {
                if !self.in_fn {
                    self.error("`return` outside of a function".to_string());
                }
                let t = match e {
                    Some(x) => Some(self.expr(x)),
                    None => None,
                };
                if let (Some(ann), Some(t)) = (self.ret.clone(), t) {
                    self.compat(&ann, &t, "return value");
                }
            }
            StmtKind::If { arms, els } => {
                for (c, b) in arms {
                    self.expr(c);
                    self.block(b);
                }
                if let Some(b) = els {
                    self.block(b);
                }
            }
            StmtKind::For { var, iter, body } => {
                let vt = match iter {
                    ForIter::Range(a, b) => {
                        self.expr(a);
                        self.expr(b);
                        Ty::Int(None)
                    }
                    ForIter::Expr(e) => {
                        self.expr(e);
                        Ty::Unknown
                    }
                };
                self.declare(var, Info { mutable: false, by_ref: false, ty: vt });
                self.block(body);
            }
            StmtKind::NoGrad(b) => self.block(b),
            StmtKind::FnDef(f) => self.func(f),
            StmtKind::Directive { body, .. } => self.block(body),
            StmtKind::Sync => {}
        }
    }

    fn func(&mut self, f: &FnDef) {
        self.declare(&f.name, Info { mutable: false, by_ref: false, ty: Ty::Fn });
        let saved_line = self.line;
        let saved_in_fn = self.in_fn;
        let saved_ret = self.ret.clone();
        self.scopes.push(HashMap::new());
        self.in_fn = true;
        self.ret = f.ret.clone();
        let mut seen: HashSet<String> = HashSet::new();
        for p in &f.params {
            self.line = f.line;
            if !seen.insert(p.name.clone()) {
                self.error(format!("duplicate parameter `{}` in fn {}", p.name, f.name));
            }
            let by_ref = matches!(&p.ty, Some(Type::Ref { mutable: true, .. }));
            let ty = match &p.ty {
                Some(t) => type_to_ty(t),
                None => Ty::Unknown,
            };
            self.declare(&p.name, Info { mutable: by_ref, by_ref, ty });
        }
        self.block(&f.body);
        self.scopes.pop();
        self.in_fn = saved_in_fn;
        self.ret = saved_ret;
        self.line = saved_line;
    }

    // ------------------------------------------------------------ annotations

    fn compat(&mut self, t: &Type, v: &Ty, what: &str) {
        let t = strip(t);
        if matches!(v, Ty::Unknown | Ty::Fn) {
            return;
        }
        match t {
            Type::Tensor(spec) => match v {
                Ty::Tensor(vd) => {
                    if let (Some(spec), Some(vd)) = (spec, vd) {
                        if spec.len() != vd.len() {
                            self.error(format!(
                                "{}: expected rank {} tensor, got {}",
                                what,
                                spec.len(),
                                ty_name(v)
                            ));
                            return;
                        }
                        for (s, d) in spec.iter().zip(vd.iter()) {
                            if let (DimSpec::Lit(n), Some(k)) = (s, d) {
                                if *n as i64 != *k {
                                    self.error(format!("{}: dimension {} does not match {}", what, k, n));
                                }
                            }
                        }
                    }
                }
                _ => self.error(format!("{}: expected Tensor, got {}", what, ty_name(v))),
            },
            Type::F32 | Type::F16 => {
                if !matches!(v, Ty::Int(_) | Ty::Float) {
                    self.error(format!("{}: expected a number, got {}", what, ty_name(v)));
                }
            }
            Type::I32 | Type::Usize => match v {
                Ty::Int(Some(k)) => {
                    if matches!(t, Type::Usize) && *k < 0 {
                        self.error(format!("{}: usize cannot be negative ({})", what, k));
                    }
                    if matches!(t, Type::I32) && (*k > i32::MAX as i64 || *k < i32::MIN as i64) {
                        self.error(format!("{}: {} does not fit in i32", what, k));
                    }
                }
                Ty::Int(None) => {}
                _ => self.error(format!("{}: expected an integer, got {}", what, ty_name(v))),
            },
            Type::Bool => {
                if !matches!(v, Ty::Bool) {
                    self.error(format!("{}: expected bool, got {}", what, ty_name(v)));
                }
            }
            Type::Str => {
                if !matches!(v, Ty::Str) {
                    self.error(format!("{}: expected str, got {}", what, ty_name(v)));
                }
            }
            Type::Named(n) => {
                let ok = match n.as_str() {
                    "List" => matches!(v, Ty::List | Ty::Shape(_)),
                    "Layer" => matches!(v, Ty::Layer),
                    _ => true,
                };
                if !ok {
                    self.error(format!("{}: expected {}, got {}", what, n, ty_name(v)));
                }
            }
            Type::Ref { .. } => {}
        }
    }

    fn bind(&mut self, ann: &Type, t: &Ty, env: &mut HashMap<String, i64>) {
        if let (Type::Tensor(Some(specs)), Ty::Tensor(Some(dims))) = (strip(ann), t) {
            if specs.len() != dims.len() {
                return;
            }
            for (s, d) in specs.iter().zip(dims.iter()) {
                if let (DimSpec::Name(n), Some(v)) = (s, d) {
                    if n == "_" {
                        continue;
                    }
                    match env.get(n) {
                        Some(old) if old != v => {
                            let msg = format!("symbolic dimension `{}` is {} here but was bound to {} earlier", n, v, old);
                            self.error(msg);
                        }
                        Some(_) => {}
                        None => {
                            env.insert(n.clone(), *v);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ expressions

    fn var(&mut self, n: &str) -> Ty {
        if let Some(i) = self.lookup(n) {
            return i.ty;
        }
        if self.fns.contains_key(n) || spec(n).is_some() || matches!(n, "ram" | "vram" | "cpu" | "gpu" | "npu") {
            return Ty::Fn;
        }
        self.error(format!("undefined variable `{}`", n));
        Ty::Unknown
    }

    fn expr(&mut self, e: &Expr) -> Ty {
        match e {
            Expr::Int(i) => Ty::Int(Some(*i)),
            Expr::Float(_) => Ty::Float,
            Expr::Bool(_) => Ty::Bool,
            Expr::Str(_) => Ty::Str,
            Expr::Var(n) => self.var(n),
            Expr::List(items) => {
                let tys: Vec<Ty> = items.iter().map(|i| self.expr(i)).collect();
                if tys.iter().all(|t| matches!(t, Ty::Int(_))) {
                    Ty::Shape(
                        tys.iter()
                            .map(|t| match t {
                                Ty::Int(v) => *v,
                                _ => None,
                            })
                            .collect(),
                    )
                } else {
                    Ty::List
                }
            }
            Expr::Unary(op, x) => {
                let t = self.expr(x);
                match (op, t) {
                    (UnOp::Not, _) => Ty::Bool,
                    (UnOp::Neg, Ty::Int(Some(v))) => Ty::Int(v.checked_neg()),
                    (UnOp::Neg, other) => other,
                }
            }
            Expr::Ref(_, x) => self.expr(x),
            Expr::Bin(op, l, r) => {
                if *op == BinOp::Pipe {
                    return self.pipe(l, r);
                }
                let a = self.expr(l);
                let b = self.expr(r);
                self.binop(*op, a, b)
            }
            Expr::Call(c, args) => self.call(c, args, 0),
            Expr::Index(b, items) => self.index(b, items),
            Expr::Field(b, name) => {
                let t = self.expr(b);
                self.field(t, name)
            }
        }
    }

    fn pipe(&mut self, l: &Expr, r: &Expr) -> Ty {
        self.expr(l);
        match r {
            Expr::Call(c, args) => {
                self.call(c, args, 1);
            }
            other => {
                self.expr(other);
            }
        }
        Ty::Unknown
    }

    fn field(&mut self, t: Ty, name: &str) -> Ty {
        match (&t, name) {
            (Ty::Tensor(Some(d)), "T") => {
                let mut d = d.clone();
                if d.len() >= 2 {
                    let n = d.len();
                    d.swap(n - 2, n - 1);
                }
                Ty::Tensor(Some(d))
            }
            (Ty::Tensor(None), "T") => Ty::Tensor(None),
            (Ty::Tensor(Some(d)), "shape") => Ty::Shape(d.clone()),
            (Ty::Tensor(None), "shape") => Ty::List,
            (Ty::Int(_), _) | (Ty::Float, _) | (Ty::Bool, _) | (Ty::Str, _) => {
                self.error(format!("{} has no field `{}`", ty_name(&t), name));
                Ty::Unknown
            }
            _ => Ty::Unknown,
        }
    }

    fn index(&mut self, b: &Expr, items: &[IdxItem]) -> Ty {
        let bt = self.expr(b);
        // None = integer index; Some((start, end)) = range, where end None means open-ended.
        let mut infos: Vec<Option<(Dim, Option<Dim>)>> = Vec::new();
        let mut at_consts: Vec<Dim> = Vec::new();
        for it in items {
            match it {
                IdxItem::At(e) => {
                    let t = self.expr(e);
                    at_consts.push(match t {
                        Ty::Int(v) => v,
                        _ => None,
                    });
                    infos.push(None);
                }
                IdxItem::Range(a, c) => {
                    let sa = match a {
                        Some(e) => match self.expr(e) {
                            Ty::Int(v) => v,
                            _ => None,
                        },
                        None => Some(0),
                    };
                    let sb: Option<Dim> = match c {
                        Some(e) => Some(match self.expr(e) {
                            Ty::Int(v) => v,
                            _ => None,
                        }),
                        None => None,
                    };
                    at_consts.push(None);
                    infos.push(Some((sa, sb)));
                }
            }
        }
        match bt {
            Ty::Shape(d) => {
                if items.len() == 1 {
                    if let Some(None) = infos.first() {
                        if let Some(Some(i)) = at_consts.first() {
                            let i = *i;
                            if i >= 0 && (i as usize) < d.len() {
                                return Ty::Int(d[i as usize]);
                            }
                            if i < 0 || (i as usize) >= d.len() {
                                if i >= 0 {
                                    self.error(format!("index {} out of range for list of length {}", i, d.len()));
                                }
                            }
                        }
                    }
                }
                Ty::Unknown
            }
            Ty::Tensor(Some(dims)) => {
                if infos.len() > dims.len() {
                    self.error(format!("too many indices ({}) for a tensor of rank {}", infos.len(), dims.len()));
                    return Ty::Tensor(None);
                }
                let mut out: Vec<Dim> = Vec::new();
                for (k, d) in dims.iter().enumerate() {
                    match infos.get(k) {
                        None => out.push(*d),
                        Some(None) => {
                            if let (Some(i), Some(n)) = (at_consts[k], *d) {
                                if i >= n || i < -n {
                                    self.error(format!("index {} out of range for axis {} of length {}", i, k, n));
                                }
                            }
                        }
                        Some(Some((s, e))) => {
                            let end: Dim = match e {
                                Some(v) => *v,
                                None => *d,
                            };
                            match (*s, end, *d) {
                                (Some(s), Some(e), Some(n)) => {
                                    if s >= 0 && e >= 0 && (s > e || e > n) {
                                        self.error(format!("slice [{}..{}] out of range for axis {} of length {}", s, e, k, n));
                                        out.push(None);
                                    } else if s >= 0 && e >= 0 {
                                        out.push(Some(e - s));
                                    } else {
                                        out.push(None);
                                    }
                                }
                                (Some(s), Some(e), None) => {
                                    if s >= 0 && e >= 0 && e >= s {
                                        out.push(Some(e - s));
                                    } else {
                                        out.push(None);
                                    }
                                }
                                _ => out.push(None),
                            }
                        }
                    }
                }
                Ty::Tensor(Some(out))
            }
            Ty::Tensor(None) => Ty::Tensor(None),
            _ => Ty::Unknown,
        }
    }

    fn binop(&mut self, op: BinOp, a: Ty, b: Ty) -> Ty {
        match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::And | BinOp::Or => {
                return Ty::Bool;
            }
            BinOp::Shift => return a,
            BinOp::MatMul => return self.matmul_ty(a, b),
            _ => {}
        }
        let sym = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            _ => "**",
        };
        match (&a, &b) {
            (Ty::Int(x), Ty::Int(y)) => {
                if matches!(op, BinOp::Div | BinOp::Mod) && *y == Some(0) {
                    self.error("division by zero".to_string());
                    return Ty::Int(None);
                }
                let v = match (x, y) {
                    (Some(x), Some(y)) => match op {
                        BinOp::Add => x.checked_add(*y),
                        BinOp::Sub => x.checked_sub(*y),
                        BinOp::Mul => x.checked_mul(*y),
                        BinOp::Div => x.checked_div(*y),
                        BinOp::Mod => x.checked_rem(*y),
                        BinOp::Pow => u32::try_from(*y).ok().and_then(|e| x.checked_pow(e)),
                        _ => None,
                    },
                    _ => None,
                };
                Ty::Int(v)
            }
            (Ty::Tensor(sa), Ty::Tensor(sb)) => {
                if op == BinOp::Pow {
                    return Ty::Tensor(sa.clone());
                }
                Ty::Tensor(self.broadcast(sym, sa, sb))
            }
            (Ty::Tensor(s), _) => Ty::Tensor(s.clone()),
            (_, Ty::Tensor(s)) => Ty::Tensor(s.clone()),
            (Ty::Float, Ty::Int(_)) | (Ty::Int(_), Ty::Float) | (Ty::Float, Ty::Float) => Ty::Float,
            (Ty::Str, Ty::Str) => Ty::Str,
            (Ty::List, Ty::List) | (Ty::Shape(_), Ty::List) | (Ty::List, Ty::Shape(_)) | (Ty::Shape(_), Ty::Shape(_)) => Ty::List,
            _ => Ty::Unknown,
        }
    }

    fn broadcast(&mut self, sym: &str, a: &Option<Vec<Dim>>, b: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
        let (sa, sb) = match (a, b) {
            (Some(x), Some(y)) => (x, y),
            _ => return None,
        };
        let n = sa.len().max(sb.len());
        let mut out: Vec<Dim> = vec![None; n];
        for i in 0..n {
            let da: Dim = if i + sa.len() >= n { sa[i + sa.len() - n] } else { Some(1) };
            let db: Dim = if i + sb.len() >= n { sb[i + sb.len() - n] } else { Some(1) };
            out[i] = match (da, db) {
                (Some(x), Some(y)) => {
                    if x == y {
                        Some(x)
                    } else if x == 1 {
                        Some(y)
                    } else if y == 1 {
                        Some(x)
                    } else {
                        self.error(format!("shape mismatch in `{}`: {} vs {}", sym, fmt_dims(sa), fmt_dims(sb)));
                        None
                    }
                }
                _ => None,
            };
        }
        Some(out)
    }

    fn matmul_ty(&mut self, a: Ty, b: Ty) -> Ty {
        if let (Ty::Tensor(Some(sa)), Ty::Tensor(Some(sb))) = (&a, &b) {
            if sa.len() >= 2 && sb.len() >= 2 {
                let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let (k2, n) = (sb[sb.len() - 2], sb[sb.len() - 1]);
                if let (Some(x), Some(y)) = (k, k2) {
                    if x != y {
                        self.error(format!(
                            "matmul shape mismatch: {} >< {} (inner dims {} vs {})",
                            fmt_dims(sa),
                            fmt_dims(sb),
                            x,
                            y
                        ));
                    }
                }
                let mut out: Vec<Dim> = if sa.len() > 2 { sa[..sa.len() - 2].to_vec() } else { sb[..sb.len() - 2].to_vec() };
                out.push(m);
                out.push(n);
                return Ty::Tensor(Some(out));
            }
            return Ty::Tensor(None);
        }
        if matches!(a, Ty::Tensor(_)) || matches!(b, Ty::Tensor(_)) {
            Ty::Tensor(None)
        } else {
            Ty::Unknown
        }
    }

    fn shape_from_args(&mut self, tys: &[Ty]) -> Ty {
        if tys.len() == 1 {
            if let Ty::Shape(d) = &tys[0] {
                for x in d.iter() {
                    if let Some(v) = x {
                        if *v < 0 {
                            self.error(format!("negative dimension {} in shape", v));
                        }
                    }
                }
                return Ty::Tensor(Some(d.clone()));
            }
        }
        if tys.iter().all(|t| matches!(t, Ty::Int(_))) {
            let dims: Vec<Dim> = tys
                .iter()
                .map(|t| match t {
                    Ty::Int(v) => *v,
                    _ => None,
                })
                .collect();
            for x in dims.iter() {
                if let Some(v) = x {
                    if *v < 0 {
                        self.error(format!("negative dimension {} in shape", v));
                    }
                }
            }
            return Ty::Tensor(Some(dims));
        }
        Ty::Tensor(None)
    }

    /// `extra` counts an implicit first argument supplied by `|>`.
    fn call(&mut self, c: &Expr, args: &[Expr], extra: usize) -> Ty {
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        let n = args.len() + extra;
        let name = match c {
            Expr::Var(name) => name.clone(),
            other => {
                self.expr(other);
                return Ty::Unknown;
            }
        };
        let local = self.lookup(&name);
        let shadowed = match &local {
            Some(info) => !matches!(info.ty, Ty::Fn),
            None => false,
        };
        if !shadowed {
            let sig = self.fns.get(&name).map(|s| (s.params.clone(), s.ret.clone()));
            if let Some((params, ret)) = sig {
                if n != params.len() {
                    self.error(format!("fn `{}` expects {} argument(s), got {}", name, params.len(), n));
                    return Ty::Unknown;
                }
                let mut env: HashMap<String, i64> = HashMap::new();
                if extra == 0 {
                    for (p, t) in params.iter().zip(arg_tys.iter()) {
                        if let Some(ann) = &p.ty {
                            self.compat(ann, t, &format!("argument `{}` of {}", p.name, name));
                            self.bind(ann, t, &mut env);
                        }
                    }
                }
                return match ret {
                    Some(Type::Tensor(Some(dims))) => Ty::Tensor(Some(
                        dims.iter()
                            .map(|d| match d {
                                DimSpec::Lit(k) => Some(*k as i64),
                                DimSpec::Name(s) => env.get(s).copied(),
                            })
                            .collect(),
                    )),
                    Some(other) => type_to_ty(&other),
                    None => Ty::Unknown,
                };
            }
        }
        if local.is_some() {
            if let Some(info) = &local {
                if matches!(info.ty, Ty::Layer) && n != 1 {
                    self.error(format!("layer `{}` takes exactly 1 argument, got {}", name, n));
                }
            }
            return Ty::Unknown;
        }
        match spec(&name) {
            Some((_, min, max)) => {
                if n < min || n > max {
                    let want = if min == max { format!("{}", min) } else { format!("{}..{}", min, max) };
                    self.error(format!("{}() takes {} argument(s), got {}", name, want, n));
                    return Ty::Unknown;
                }
                if extra == 0 {
                    match name.as_str() {
                        "zeros" | "ones" | "randn" => return self.shape_from_args(&arg_tys),
                        "linear" | "conv2d" | "embed" | "pos_embed" => return Ty::Layer,
                        _ => {}
                    }
                }
                Ty::Unknown
            }
            None => {
                self.error(format!("undefined function `{}`", name));
                Ty::Unknown
            }
        }
    }
}
