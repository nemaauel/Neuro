//! Reverse-mode autograd. Nodes form an Rc graph; creation ids give the tape order,
//! so `backward` replays the tape in descending id order.

use crate::tensor::*;
use std::cell::{Cell, Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub type Var = Rc<Node>;

#[derive(Clone, Copy)]
pub struct Cx {
    pub ex: Exec,
    pub grad: bool,
}

pub enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    PowS(f32),
    Un(Un),
    MatMul,
    Swap(usize, usize),
    /// Holds the parent's original shape (also used as identity for memory transfers).
    Reshape(Vec<usize>),
    Slice(Vec<(usize, usize)>),
    Concat(usize),
    Sum(Option<usize>),
    Softmax,
    LayerNorm,
    Gather(Tensor),
    CrossEntropy(Tensor),
    Linear(Option<Un>),
    Conv2d(usize, usize),
}

pub struct Node {
    pub id: u64,
    pub value: RefCell<Tensor>,
    pub parents: Vec<Var>,
    pub op: Op,
    pub requires_grad: bool,
    pub freed: Cell<bool>,
}

impl Drop for Node {
    /// Iterative drop so very long graphs cannot overflow the stack.
    fn drop(&mut self) {
        let mut stack: Vec<Var> = std::mem::take(&mut self.parents);
        while let Some(n) = stack.pop() {
            if let Ok(mut node) = Rc::try_unwrap(n) {
                stack.append(&mut node.parents);
            }
        }
    }
}

thread_local! {
    static NEXT_ID: Cell<u64> = Cell::new(1);
}

fn next_id() -> u64 {
    NEXT_ID.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    })
}

pub fn leaf(value: Tensor, requires_grad: bool) -> Var {
    Rc::new(Node {
        id: next_id(),
        value: RefCell::new(value),
        parents: Vec::new(),
        op: Op::Leaf,
        requires_grad,
        freed: Cell::new(false),
    })
}

pub fn make(value: Tensor, parents: Vec<Var>, op: Op, grad_on: bool) -> Var {
    let rg = grad_on && parents.iter().any(|p| p.requires_grad);
    let (parents, op) = if rg { (parents, op) } else { (Vec::new(), Op::Leaf) };
    Rc::new(Node {
        id: next_id(),
        value: RefCell::new(value),
        parents,
        op,
        requires_grad: rg,
        freed: Cell::new(false),
    })
}

pub fn konst(v: f32, dtype: DType, mem: Mem) -> Var {
    leaf(Tensor::scalar(v, dtype, mem), false)
}

pub fn get(v: &Var) -> Res<Ref<'_, Tensor>> {
    if v.freed.get() {
        return Err("tensor was released with free() and can no longer be used".to_string());
    }
    Ok(v.value.borrow())
}

pub fn shape_of(v: &Var) -> Res<Vec<usize>> {
    let t = get(v)?;
    Ok(t.shape.clone())
}

pub fn same_mem(a: &Tensor, b: &Tensor) -> Res<Mem> {
    if a.mem() != b.mem() {
        return Err(format!(
            "memory mismatch: one operand lives in {} and the other in {}; move data with `>>` (e.g. x >> vram)",
            a.mem().name(),
            b.mem().name()
        ));
    }
    Ok(a.mem())
}

fn promote(a: DType, b: DType, int_ok: bool) -> DType {
    if int_ok && a == DType::I32 && b == DType::I32 {
        DType::I32
    } else if a == DType::F16 && b == DType::F16 {
        DType::F16
    } else {
        DType::F32
    }
}

// ---------------------------------------------------------------- forward builders

pub fn bin(op: Op, a: &Var, b: &Var, cx: Cx) -> Res<Var> {
    let out = {
        let ta = get(a)?;
        let tb = get(b)?;
        same_mem(&ta, &tb)?;
        let dt = promote(ta.dtype, tb.dtype, matches!(op, Op::Add | Op::Sub | Op::Mul));
        match op {
            Op::Add => zip_bc(&ta, &tb, dt, |x, y| x + y)?,
            Op::Sub => zip_bc(&ta, &tb, dt, |x, y| x - y)?,
            Op::Mul => zip_bc(&ta, &tb, dt, |x, y| x * y)?,
            Op::Div => zip_bc(&ta, &tb, dt, |x, y| x / y)?,
            _ => return Err("internal: bin() called with a non-binary op".to_string()),
        }
    };
    Ok(make(out, vec![a.clone(), b.clone()], op, cx.grad))
}

pub fn mul_s(a: &Var, s: f32, cx: Cx) -> Res<Var> {
    let mem = get(a)?.mem();
    bin(Op::Mul, a, &konst(s, DType::F32, mem), cx)
}

pub fn pow_s(a: &Var, p: f32, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        map_t(&t, DType::F32, |x| x.powf(p))
    };
    Ok(make(out, vec![a.clone()], Op::PowS(p), cx.grad))
}

pub fn un(k: Un, a: &Var, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        let dt = if t.dtype == DType::F16 { DType::F16 } else { DType::F32 };
        map_t(&t, dt, |x| un_fwd(k, x))
    };
    Ok(make(out, vec![a.clone()], Op::Un(k), cx.grad))
}

pub fn matmul_v(a: &Var, b: &Var, cx: Cx) -> Res<Var> {
    let out = {
        let ta = get(a)?;
        let tb = get(b)?;
        same_mem(&ta, &tb)?;
        matmul(&ta, &tb, cx.ex)?
    };
    Ok(make(out, vec![a.clone(), b.clone()], Op::MatMul, cx.grad))
}

pub fn swap_v(a: &Var, i: usize, j: usize, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        if i >= t.rank() || j >= t.rank() {
            return Err(format!("transpose: axes {} and {} out of range for shape {:?}", i, j, t.shape));
        }
        swap_axes(&t, i, j)
    };
    Ok(make(out, vec![a.clone()], Op::Swap(i, j), cx.grad))
}

pub fn reshape_v(a: &Var, shape: Vec<usize>, cx: Cx) -> Res<Var> {
    let (out, old) = {
        let t = get(a)?;
        if numel(&shape) != t.numel() {
            return Err(format!("cannot reshape {:?} ({} elements) into {:?} ({} elements)", t.shape, t.numel(), shape, numel(&shape)));
        }
        (Tensor::new(shape, t.data.clone(), t.dtype, t.mem()), t.shape.clone())
    };
    Ok(make(out, vec![a.clone()], Op::Reshape(old), cx.grad))
}

/// Copies `a` into another memory space (`>>`); gradients flow back unchanged.
pub fn transfer_v(a: &Var, mem: Mem, cx: Cx) -> Res<Var> {
    let (out, old) = {
        let t = get(a)?;
        (t.with_mem(mem), t.shape.clone())
    };
    Ok(make(out, vec![a.clone()], Op::Reshape(old), cx.grad))
}

pub fn slice_v(a: &Var, ranges: Vec<(usize, usize)>, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        if ranges.len() != t.rank() {
            return Err(format!("slice: {} ranges for a tensor of rank {}", ranges.len(), t.rank()));
        }
        for (d, &(s, e)) in ranges.iter().enumerate() {
            if s > e || e > t.shape[d] {
                return Err(format!("slice [{}..{}] out of range for axis {} of length {}", s, e, d, t.shape[d]));
            }
        }
        slice_nd(&t, &ranges)
    };
    Ok(make(out, vec![a.clone()], Op::Slice(ranges), cx.grad))
}

pub fn concat_v(vars: &[Var], axis: usize, cx: Cx) -> Res<Var> {
    if vars.is_empty() {
        return Err("concat: empty list".to_string());
    }
    let out = {
        let guards: Vec<Ref<Tensor>> = vars.iter().map(|v| get(v)).collect::<Res<Vec<_>>>()?;
        let refs: Vec<&Tensor> = guards.iter().map(|g| &**g).collect();
        for t in refs.iter() {
            same_mem(refs[0], t)?;
        }
        concat_nd(&refs, axis)?
    };
    Ok(make(out, vars.to_vec(), Op::Concat(axis), cx.grad))
}

pub fn sum_v(a: &Var, axis: Option<usize>, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        match axis {
            None => sum_all(&t),
            Some(ax) => {
                if ax >= t.rank() {
                    return Err(format!("sum: axis {} out of range for shape {:?}", ax, t.shape));
                }
                sum_axis(&t, ax, false)
            }
        }
    };
    Ok(make(out, vec![a.clone()], Op::Sum(axis), cx.grad))
}

pub fn mean_v(a: &Var, axis: Option<usize>, cx: Cx) -> Res<Var> {
    let n = {
        let t = get(a)?;
        match axis {
            None => t.numel(),
            Some(ax) => {
                if ax >= t.rank() {
                    return Err(format!("mean: axis {} out of range for shape {:?}", ax, t.shape));
                }
                t.shape[ax]
            }
        }
    };
    let s = sum_v(a, axis, cx)?;
    mul_s(&s, 1.0 / n.max(1) as f32, cx)
}

pub fn softmax_v(a: &Var, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        softmax_last(&t)
    };
    Ok(make(out, vec![a.clone()], Op::Softmax, cx.grad))
}

pub fn layernorm_v(a: &Var, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(a)?;
        layernorm_fwd(&t)
    };
    Ok(make(out, vec![a.clone()], Op::LayerNorm, cx.grad))
}

pub fn gather_v(w: &Var, idx: Tensor, cx: Cx) -> Res<Var> {
    let out = {
        let tw = get(w)?;
        if tw.rank() != 2 {
            return Err(format!("embedding weight must be rank 2, got {:?}", tw.shape));
        }
        let (v, d) = (tw.shape[0], tw.shape[1]);
        let mut data = Vec::with_capacity(idx.data.len() * d);
        for &t in idx.data.iter() {
            if t < 0.0 || (t as usize) >= v {
                return Err(format!("embedding index {} out of range for a vocabulary of {}", t, v));
            }
            let r = t as usize;
            data.extend_from_slice(&tw.data[r * d..(r + 1) * d]);
        }
        let mut shape = idx.shape.clone();
        shape.push(d);
        Tensor::new(shape, data, DType::F32, tw.mem())
    };
    Ok(make(out, vec![w.clone()], Op::Gather(idx), cx.grad))
}

pub fn ce_v(logits: &Var, target: Tensor, cx: Cx) -> Res<Var> {
    let out = {
        let t = get(logits)?;
        ce_fwd(&t, &target)?
    };
    Ok(make(out, vec![logits.clone()], Op::CrossEntropy(target), cx.grad))
}

pub fn linear_v(x: &Var, w: &Var, b: Option<&Var>, act: Option<Un>, cx: Cx) -> Res<Var> {
    let out = {
        let tx = get(x)?;
        let tw = get(w)?;
        same_mem(&tx, &tw)?;
        let tbg = match b {
            Some(bv) => Some(get(bv)?),
            None => None,
        };
        if let Some(tb) = &tbg {
            same_mem(&tx, tb)?;
        }
        linear_fwd(&tx, &tw, tbg.as_deref(), act, cx.ex)?
    };
    let mut parents = vec![x.clone(), w.clone()];
    if let Some(bv) = b {
        parents.push(bv.clone());
    }
    Ok(make(out, parents, Op::Linear(act), cx.grad))
}

pub fn conv2d_v(x: &Var, w: &Var, b: Option<&Var>, stride: usize, pad: usize, cx: Cx) -> Res<Var> {
    let out = {
        let tx = get(x)?;
        let tw = get(w)?;
        same_mem(&tx, &tw)?;
        let tbg = match b {
            Some(bv) => Some(get(bv)?),
            None => None,
        };
        if let Some(tb) = &tbg {
            same_mem(&tx, tb)?;
        }
        conv2d_fwd(&tx, &tw, tbg.as_deref(), stride, pad)?
    };
    let mut parents = vec![x.clone(), w.clone()];
    if let Some(bv) = b {
        parents.push(bv.clone());
    }
    Ok(make(out, parents, Op::Conv2d(stride, pad), cx.grad))
}

pub fn detach(v: &Var) -> Res<Var> {
    let t = get(v)?.clone();
    Ok(leaf(t, false))
}

// ---------------------------------------------------------------- backward

fn backward_op(node: &Node, dy: &Tensor, cx: Cx) -> Res<Vec<Option<Tensor>>> {
    let p = &node.parents;
    let ex = cx.ex;
    match &node.op {
        Op::Leaf => Ok(Vec::new()),
        Op::Add => {
            let sa = shape_of(&p[0])?;
            let sb = shape_of(&p[1])?;
            Ok(vec![Some(unbroadcast(dy, &sa)), Some(unbroadcast(dy, &sb))])
        }
        Op::Sub => {
            let sa = shape_of(&p[0])?;
            let sb = shape_of(&p[1])?;
            let neg = map_t(dy, DType::F32, |g| -g);
            Ok(vec![Some(unbroadcast(dy, &sa)), Some(unbroadcast(&neg, &sb))])
        }
        Op::Mul => {
            let ta = get(&p[0])?;
            let tb = get(&p[1])?;
            let ga = if p[0].requires_grad {
                Some(unbroadcast(&zip_bc(dy, &tb, DType::F32, |g, y| g * y)?, &ta.shape))
            } else {
                None
            };
            let gb = if p[1].requires_grad {
                Some(unbroadcast(&zip_bc(dy, &ta, DType::F32, |g, x| g * x)?, &tb.shape))
            } else {
                None
            };
            Ok(vec![ga, gb])
        }
        Op::Div => {
            let ta = get(&p[0])?;
            let tb = get(&p[1])?;
            let ga = if p[0].requires_grad {
                Some(unbroadcast(&zip_bc(dy, &tb, DType::F32, |g, y| g / y)?, &ta.shape))
            } else {
                None
            };
            let gb = if p[1].requires_grad {
                let t1 = zip_bc(dy, &ta, DType::F32, |g, x| g * x)?;
                let t2 = zip_bc(&t1, &tb, DType::F32, |t, y| -t / (y * y))?;
                Some(unbroadcast(&t2, &tb.shape))
            } else {
                None
            };
            Ok(vec![ga, gb])
        }
        Op::PowS(pw) => {
            let tx = get(&p[0])?;
            let pw = *pw;
            let data: Vec<f32> = tx
                .data
                .iter()
                .zip(dy.data.iter())
                .map(|(&x, &g)| g * pw * x.powf(pw - 1.0))
                .collect();
            Ok(vec![Some(Tensor::new(tx.shape.clone(), data, DType::F32, tx.mem()))])
        }
        Op::Un(k) => {
            let tx = get(&p[0])?;
            let y = node.value.borrow();
            let data: Vec<f32> = tx
                .data
                .iter()
                .zip(y.data.iter())
                .zip(dy.data.iter())
                .map(|((&x, &yy), &g)| g * un_grad(*k, x, yy))
                .collect();
            Ok(vec![Some(Tensor::new(tx.shape.clone(), data, DType::F32, tx.mem()))])
        }
        Op::MatMul => {
            let ta = get(&p[0])?;
            let tb = get(&p[1])?;
            let (ra, rb) = (ta.rank(), tb.rank());
            let ga = if p[0].requires_grad {
                let bt = swap_axes(&tb, rb - 2, rb - 1);
                let full = matmul(dy, &bt, ex)?;
                Some(unbroadcast(&full, &ta.shape))
            } else {
                None
            };
            let gb = if p[1].requires_grad {
                let at = swap_axes(&ta, ra - 2, ra - 1);
                let full = matmul(&at, dy, ex)?;
                Some(unbroadcast(&full, &tb.shape))
            } else {
                None
            };
            Ok(vec![ga, gb])
        }
        Op::Swap(i, j) => Ok(vec![Some(swap_axes(dy, *i, *j))]),
        Op::Reshape(orig) => {
            let pm = get(&p[0])?.mem();
            Ok(vec![Some(Tensor::new(orig.clone(), dy.data.clone(), dy.dtype, pm))])
        }
        Op::Slice(r) => {
            let ps = shape_of(&p[0])?;
            Ok(vec![Some(slice_scatter(dy, &ps, r))])
        }
        Op::Concat(axis) => {
            let mut out = Vec::with_capacity(p.len());
            let mut off = 0usize;
            for par in p.iter() {
                let s = shape_of(par)?;
                let mut r: Vec<(usize, usize)> = s.iter().map(|&d| (0, d)).collect();
                r[*axis] = (off, off + s[*axis]);
                off += s[*axis];
                out.push(Some(slice_nd(dy, &r)));
            }
            Ok(out)
        }
        Op::Sum(ax) => {
            let ps = shape_of(&p[0])?;
            let g = match ax {
                None => Tensor::new(ps.clone(), vec![dy.data[0]; numel(&ps)], DType::F32, dy.mem()),
                Some(a) => expand_axis(dy, &ps, *a),
            };
            Ok(vec![Some(g)])
        }
        Op::Softmax => {
            let y = node.value.borrow();
            let d = y.shape.last().copied().unwrap_or(1).max(1);
            let rows = y.data.len() / d;
            let mut out = vec![0.0f32; y.data.len()];
            for r in 0..rows {
                let yr = &y.data[r * d..(r + 1) * d];
                let gr = &dy.data[r * d..(r + 1) * d];
                let dot: f32 = yr.iter().zip(gr.iter()).map(|(&a, &b)| a * b).sum();
                for j in 0..d {
                    out[r * d + j] = yr[j] * (gr[j] - dot);
                }
            }
            Ok(vec![Some(Tensor::new(y.shape.clone(), out, DType::F32, y.mem()))])
        }
        Op::LayerNorm => {
            let tx = get(&p[0])?;
            let d = tx.shape.last().copied().unwrap_or(1).max(1);
            let rows = tx.data.len() / d;
            let mut out = vec![0.0f32; tx.data.len()];
            for r in 0..rows {
                let row = &tx.data[r * d..(r + 1) * d];
                let g = &dy.data[r * d..(r + 1) * d];
                let mean = row.iter().sum::<f32>() / d as f32;
                let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                let rstd = 1.0 / (var + LN_EPS).sqrt();
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                for j in 0..d {
                    let xh = (row[j] - mean) * rstd;
                    s1 += g[j];
                    s2 += g[j] * xh;
                }
                for j in 0..d {
                    let xh = (row[j] - mean) * rstd;
                    out[r * d + j] = rstd * (g[j] - s1 / d as f32 - xh * s2 / d as f32);
                }
            }
            Ok(vec![Some(Tensor::new(tx.shape.clone(), out, DType::F32, tx.mem()))])
        }
        Op::Gather(idx) => {
            let tw = get(&p[0])?;
            let (v, d) = (tw.shape[0], tw.shape[1]);
            let mut out = vec![0.0f32; v * d];
            for (i, &t) in idx.data.iter().enumerate() {
                let r = t as usize;
                for j in 0..d {
                    out[r * d + j] += dy.data[i * d + j];
                }
            }
            Ok(vec![Some(Tensor::new(vec![v, d], out, DType::F32, tw.mem()))])
        }
        Op::CrossEntropy(tg) => {
            let tx = get(&p[0])?;
            let d = tx.shape[tx.rank() - 1];
            let rows = tx.data.len() / d;
            let g = dy.data[0] / rows as f32;
            let mut out = vec![0.0f32; tx.data.len()];
            for r in 0..rows {
                let row = &tx.data[r * d..(r + 1) * d];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
                for j in 0..d {
                    out[r * d + j] = ((row[j] - mx).exp() / sum) * g;
                }
                out[r * d + tg.data[r] as usize] -= g;
            }
            Ok(vec![Some(Tensor::new(tx.shape.clone(), out, DType::F32, tx.mem()))])
        }
        Op::Linear(act) => {
            let tx = get(&p[0])?;
            let tw = get(&p[1])?;
            let tbg = if p.len() == 3 { Some(get(&p[2])?) } else { None };
            let tb: Option<&Tensor> = tbg.as_deref();
            let (inp, outd) = (tw.shape[0], tw.shape[1]);
            let rows: usize = tx.shape[..tx.rank() - 1].iter().product();
            let y = node.value.borrow();
            // GELU needs the pre-activation: recompute it instead of storing it (@fuse trade-off).
            let dz: Vec<f32> = match act {
                None => dy.data.clone(),
                Some(a) => {
                    let zt = if *a == Un::Gelu { Some(linear_fwd(&tx, &tw, tb, None, ex)?) } else { None };
                    (0..dy.data.len())
                        .map(|i| {
                            let yv = y.data[i];
                            let zv = match &zt {
                                Some(z) => z.data[i],
                                None => yv,
                            };
                            dy.data[i] * un_grad(*a, zv, yv)
                        })
                        .collect()
                }
            };
            let mut grads: Vec<Option<Tensor>> = vec![None, None];
            if p[0].requires_grad {
                let wt = transpose2(&tw.data, inp, outd);
                let mut dx = vec![0.0f32; rows * inp];
                gemm(&dz, &wt, &mut dx, rows, outd, inp, ex);
                grads[0] = Some(Tensor::new(tx.shape.clone(), dx, DType::F32, tx.mem()));
            }
            if p[1].requires_grad {
                let xt = transpose2(&tx.data, rows, inp);
                let mut dw = vec![0.0f32; inp * outd];
                gemm(&xt, &dz, &mut dw, inp, rows, outd, ex);
                grads[1] = Some(Tensor::new(vec![inp, outd], dw, DType::F32, tw.mem()));
            }
            if tb.is_some() {
                let g = if p[2].requires_grad {
                    let mut db = vec![0.0f32; outd];
                    for r in 0..rows {
                        for j in 0..outd {
                            db[j] += dz[r * outd + j];
                        }
                    }
                    Some(Tensor::new(vec![outd], db, DType::F32, tw.mem()))
                } else {
                    None
                };
                grads.push(g);
            }
            Ok(grads)
        }
        Op::Conv2d(stride, pad) => {
            let tx = get(&p[0])?;
            let tw = get(&p[1])?;
            let (dx, dw, db) = conv2d_bwd(&tx, &tw, dy, *stride, *pad)?;
            let gx = if p[0].requires_grad { Some(Tensor::new(tx.shape.clone(), dx, DType::F32, tx.mem())) } else { None };
            let gw = if p[1].requires_grad { Some(Tensor::new(tw.shape.clone(), dw, DType::F32, tw.mem())) } else { None };
            let mut g = vec![gx, gw];
            if p.len() == 3 {
                let n = db.len();
                g.push(if p[2].requires_grad { Some(Tensor::new(vec![n], db, DType::F32, tw.mem())) } else { None });
            }
            Ok(g)
        }
    }
}

/// Backpropagates from a scalar `loss` and returns one gradient per entry of `params`
/// (zeros for parameters that do not influence the loss).
pub fn backward(loss: &Var, params: &[Var], cx: Cx) -> Res<Vec<Tensor>> {
    let (lshape, lmem) = {
        let t = get(loss)?;
        if t.numel() != 1 {
            return Err(format!("grad: loss must be a scalar, got shape {:?}", t.shape));
        }
        (t.shape.clone(), t.mem())
    };
    if !loss.requires_grad {
        return Err("grad: loss does not depend on any trainable tensor (was it computed inside no_grad?)".to_string());
    }
    let mut seen: HashSet<u64> = HashSet::new();
    let mut nodes: Vec<Var> = Vec::new();
    let mut stack: Vec<Var> = vec![loss.clone()];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.id) {
            continue;
        }
        if !n.requires_grad {
            continue;
        }
        for par in n.parents.iter() {
            stack.push(par.clone());
        }
        nodes.push(n);
    }
    nodes.sort_by(|a, b| b.id.cmp(&a.id));

    let mut grads: HashMap<u64, Tensor> = HashMap::new();
    grads.insert(loss.id, Tensor::new(lshape, vec![1.0], DType::F32, lmem));
    for n in nodes.iter() {
        if matches!(n.op, Op::Leaf) {
            continue;
        }
        let dy = match grads.remove(&n.id) {
            Some(g) => g,
            None => continue,
        };
        let pg = backward_op(n, &dy, cx)?;
        for (par, g) in n.parents.iter().zip(pg.into_iter()) {
            if !par.requires_grad {
                continue;
            }
            if let Some(g) = g {
                match grads.get_mut(&par.id) {
                    Some(acc) => {
                        for (a, b) in acc.data.iter_mut().zip(g.data.iter()) {
                            *a += *b;
                        }
                    }
                    None => {
                        grads.insert(par.id, g);
                    }
                }
            }
        }
    }

    let mut res = Vec::with_capacity(params.len());
    for p in params {
        let t = get(p)?;
        match grads.get(&p.id) {
            Some(g) => {
                if g.shape != t.shape {
                    return Err(format!("internal: gradient shape {:?} != parameter shape {:?}", g.shape, t.shape));
                }
                res.push(g.clone());
            }
            None => res.push(Tensor::new(t.shape.clone(), vec![0.0; t.data.len()], DType::F32, t.mem())),
        }
    }
    Ok(res)
}
