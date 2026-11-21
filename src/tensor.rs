//! Dense f32 tensors, shape helpers and CPU kernels.
//! `Backend` is the seam where a wgpu/Vulkan compute backend plugs in.

use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

pub type Res<T> = Result<T, String>;

/// Live VRAM bytes (logical accounting of every tensor tagged `Mem::Vram`).
pub static VRAM_BYTES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F16,
    I32,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::I32 => "i32",
        }
    }

    pub fn size(self) -> usize {
        match self {
            DType::F16 => 2,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mem {
    Ram,
    Vram,
}

impl Mem {
    pub fn name(self) -> &'static str {
        match self {
            Mem::Ram => "ram",
            Mem::Vram => "vram",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Gpu,
    Npu,
}

/// Row-major tensor. Integer tensors (`i32`) store exact values up to 2^24 in `data`.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub dtype: DType,
    mem: Mem,
    acct: usize,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>, dtype: DType, mem: Mem) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        let acct = if mem == Mem::Vram { data.len() * dtype.size() } else { 0 };
        VRAM_BYTES.fetch_add(acct, Ordering::Relaxed);
        Tensor { shape, data, dtype, mem, acct }
    }

    pub fn scalar(v: f32, dtype: DType, mem: Mem) -> Tensor {
        Tensor::new(Vec::new(), vec![v], dtype, mem)
    }

    pub fn empty() -> Tensor {
        Tensor::new(vec![0], Vec::new(), DType::F32, Mem::Ram)
    }

    pub fn mem(&self) -> Mem {
        self.mem
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Copy of this tensor living in another memory space.
    pub fn with_mem(&self, mem: Mem) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.clone(), self.dtype, mem)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.clone(), self.dtype, self.mem)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.acct > 0 {
            VRAM_BYTES.fetch_sub(self.acct, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------- execution

pub fn cpu_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// Compute backend seam. Only the CPU implementation is compiled in.
pub trait Backend: Sync {
    fn name(&self) -> &'static str;
    /// c[m,n] += a[m,k] * b[k,n]; `c` must be zero-initialised by the caller.
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, simd: bool);
}

pub struct CpuBackend;
pub static CPU_BACKEND: CpuBackend = CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, simd: bool) {
        par_rows(c, n, cpu_threads(), m * k * n, |i: usize, row: &mut [f32]| {
            let arow = &a[i * k..(i + 1) * k];
            for (kk, &av) in arow.iter().enumerate() {
                axpy(row, &b[kk * n..(kk + 1) * n], av, simd);
            }
        });
    }
}

#[derive(Clone, Copy)]
pub struct Exec {
    pub device: Device,
    pub simd: bool,
    pub f16: bool,
}

impl Exec {
    pub fn backend(&self) -> &'static dyn Backend {
        if self.device == Device::Gpu {
            &crate::gpu::GPU_BACKEND
        } else {
            &CPU_BACKEND
        }
    }
}

pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, ex: Exec) {
    ex.backend().gemm(a, b, c, m, k, n, ex.simd);
}

/// Splits `out` into rows of `row_len` and runs `f(row_index, row)`, in parallel when worthwhile.
pub fn par_rows<F>(out: &mut [f32], row_len: usize, threads: usize, work: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    if threads <= 1 || rows < 2 || work < (1 << 16) {
        for (i, r) in out.chunks_mut(row_len).enumerate() {
            f(i, r);
        }
        return;
    }
    let per = (rows + threads - 1) / threads;
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let fr = &f;
            s.spawn(move || {
                for (j, r) in chunk.chunks_mut(row_len).enumerate() {
                    fr(ci * per + j, r);
                }
            });
        }
    });
}

fn axpy(y: &mut [f32], x: &[f32], a: f32, simd: bool) {
    if simd {
        let mut yc = y.chunks_exact_mut(8);
        let mut xc = x.chunks_exact(8);
        for (yy, xx) in (&mut yc).zip(&mut xc) {
            for l in 0..8 {
                yy[l] += a * xx[l];
            }
        }
        for (yy, xx) in yc.into_remainder().iter_mut().zip(xc.remainder()) {
            *yy += a * *xx;
        }
    } else {
        for (yy, xx) in y.iter_mut().zip(x) {
            *yy += a * *xx;
        }
    }
}

// ---------------------------------------------------------------- shapes

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Res<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(format!("cannot broadcast shapes {:?} and {:?}", a, b));
        };
    }
    Ok(out)
}

fn bstrides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let n = out.len();
    let off = n - shape.len();
    let st = strides(shape);
    (0..n)
        .map(|i| if i < off || shape[i - off] == 1 { 0 } else { st[i - off] })
        .collect()
}

/// Reads `src` in row-major order of `out_shape` using arbitrary source strides.
pub fn gather_strided(src: &[f32], out_shape: &[usize], st: &[usize]) -> Vec<f32> {
    let total = numel(out_shape);
    let n = out_shape.len();
    let mut res = Vec::with_capacity(total);
    if total == 0 {
        return res;
    }
    let mut idx = vec![0usize; n];
    let mut off = 0usize;
    for _ in 0..total {
        res.push(src[off]);
        for d in (0..n).rev() {
            idx[d] += 1;
            off += st[d];
            if idx[d] < out_shape[d] {
                break;
            }
            off -= st[d] * out_shape[d];
            idx[d] = 0;
        }
    }
    res
}

fn expanded<'a>(t: &'a Tensor, out: &[usize]) -> Cow<'a, [f32]> {
    if t.shape == out {
        Cow::Borrowed(&t.data[..])
    } else {
        Cow::Owned(gather_strided(&t.data, out, &bstrides(&t.shape, out)))
    }
}

/// Element-wise binary op with broadcasting. Memory checks are the caller's job.
pub fn zip_bc<F: Fn(f32, f32) -> f32>(a: &Tensor, b: &Tensor, dtype: DType, f: F) -> Res<Tensor> {
    let out = broadcast_shape(&a.shape, &b.shape)?;
    let data: Vec<f32> = if a.shape == b.shape {
        a.data.iter().zip(b.data.iter()).map(|(&x, &y)| f(x, y)).collect()
    } else if b.data.len() == 1 && out == a.shape {
        let bv = b.data[0];
        a.data.iter().map(|&x| f(x, bv)).collect()
    } else if a.data.len() == 1 && out == b.shape {
        let av = a.data[0];
        b.data.iter().map(|&y| f(av, y)).collect()
    } else {
        let ea = expanded(a, &out);
        let eb = expanded(b, &out);
        ea.iter().zip(eb.iter()).map(|(&x, &y)| f(x, y)).collect()
    };
    Ok(Tensor::new(out, data, dtype, a.mem))
}

pub fn map_t<F: Fn(f32) -> f32>(a: &Tensor, dtype: DType, f: F) -> Tensor {
    Tensor::new(a.shape.clone(), a.data.iter().map(|&x| f(x)).collect(), dtype, a.mem)
}

pub fn sum_axis(x: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    let outer: usize = x.shape[..axis].iter().product();
    let n = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let mut out = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for j in 0..n {
            let base = (o * n + j) * inner;
            let ob = o * inner;
            for i in 0..inner {
                out[ob + i] += x.data[base + i];
            }
        }
    }
    let mut shape = x.shape.clone();
    if keepdim {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    Tensor::new(shape, out, x.dtype, x.mem)
}

pub fn sum_all(x: &Tensor) -> Tensor {
    let s: f64 = x.data.iter().map(|&v| v as f64).sum();
    Tensor::new(Vec::new(), vec![s as f32], x.dtype, x.mem)
}

/// Sums a broadcast result back down to `shape` (gradient of broadcasting).
pub fn unbroadcast(g: &Tensor, shape: &[usize]) -> Tensor {
    if g.shape == shape {
        return g.clone();
    }
    let mut cur = g.clone();
    while cur.rank() > shape.len() {
        cur = sum_axis(&cur, 0, false);
    }
    for i in 0..shape.len() {
        if shape[i] == 1 && cur.shape[i] != 1 {
            cur = sum_axis(&cur, i, true);
        }
    }
    cur
}

/// Repeats `g` (shape without `axis`) along `axis` up to `shape`.
pub fn expand_axis(g: &Tensor, shape: &[usize], axis: usize) -> Tensor {
    let outer: usize = shape[..axis].iter().product();
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * n * inner);
    for o in 0..outer {
        for _ in 0..n {
            out.extend_from_slice(&g.data[o * inner..(o + 1) * inner]);
        }
    }
    Tensor::new(shape.to_vec(), out, g.dtype, g.mem)
}

pub fn slice_nd(x: &Tensor, r: &[(usize, usize)]) -> Tensor {
    let rank = x.rank();
    if rank == 0 {
        return x.clone();
    }
    let st = strides(&x.shape);
    let oshape: Vec<usize> = r.iter().map(|&(a, b)| b - a).collect();
    let total = numel(&oshape);
    let mut data = Vec::with_capacity(total);
    if total > 0 {
        let last = rank - 1;
        let rowlen = oshape[last];
        let lead = &oshape[..last];
        let nrows = numel(lead);
        let mut idx = vec![0usize; last];
        for _ in 0..nrows {
            let mut off = r[last].0 * st[last];
            for d in 0..last {
                off += (r[d].0 + idx[d]) * st[d];
            }
            data.extend_from_slice(&x.data[off..off + rowlen]);
            for d in (0..last).rev() {
                idx[d] += 1;
                if idx[d] < lead[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Tensor::new(oshape, data, x.dtype, x.mem)
}

/// Inverse of `slice_nd` for gradients: places `g` inside a zero tensor of shape `full`.
pub fn slice_scatter(g: &Tensor, full: &[usize], r: &[(usize, usize)]) -> Tensor {
    let mut out = vec![0.0f32; numel(full)];
    let rank = full.len();
    if rank == 0 {
        if !g.data.is_empty() {
            out[0] = g.data[0];
        }
        return Tensor::new(Vec::new(), out, g.dtype, g.mem);
    }
    let st = strides(full);
    let oshape: Vec<usize> = r.iter().map(|&(a, b)| b - a).collect();
    if numel(&oshape) > 0 {
        let last = rank - 1;
        let rowlen = oshape[last];
        let lead = &oshape[..last];
        let nrows = numel(lead);
        let mut idx = vec![0usize; last];
        let mut src = 0usize;
        for _ in 0..nrows {
            let mut off = r[last].0 * st[last];
            for d in 0..last {
                off += (r[d].0 + idx[d]) * st[d];
            }
            for k in 0..rowlen {
                out[off + k] += g.data[src + k];
            }
            src += rowlen;
            for d in (0..last).rev() {
                idx[d] += 1;
                if idx[d] < lead[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Tensor::new(full.to_vec(), out, g.dtype, g.mem)
}

pub fn concat_nd(ts: &[&Tensor], axis: usize) -> Res<Tensor> {
    if ts.is_empty() {
        return Err("concat: empty list".to_string());
    }
    let first = ts[0];
    let rank = first.rank();
    if axis >= rank {
        return Err(format!("concat: axis {} out of range for rank {}", axis, rank));
    }
    let mut total_ax = 0usize;
    for t in ts {
        if t.rank() != rank {
            return Err(format!("concat: rank mismatch ({:?} vs {:?})", first.shape, t.shape));
        }
        for d in 0..rank {
            if d != axis && t.shape[d] != first.shape[d] {
                return Err(format!(
                    "concat: shapes {:?} and {:?} differ outside axis {}",
                    first.shape, t.shape, axis
                ));
            }
        }
        total_ax += t.shape[axis];
    }
    let outer: usize = first.shape[..axis].iter().product();
    let inner: usize = first.shape[axis + 1..].iter().product();
    let mut data = Vec::with_capacity(outer * total_ax * inner);
    for o in 0..outer {
        for t in ts {
            let c = t.shape[axis] * inner;
            data.extend_from_slice(&t.data[o * c..(o + 1) * c]);
        }
    }
    let mut shape = first.shape.clone();
    shape[axis] = total_ax;
    let dtype = if ts.iter().all(|t| t.dtype == DType::I32) { DType::I32 } else { DType::F32 };
    Ok(Tensor::new(shape, data, dtype, first.mem))
}

pub fn swap_axes(x: &Tensor, a: usize, b: usize) -> Tensor {
    let mut shape = x.shape.clone();
    let mut st = strides(&x.shape);
    shape.swap(a, b);
    st.swap(a, b);
    let data = gather_strided(&x.data, &shape, &st);
    Tensor::new(shape, data, x.dtype, x.mem)
}

pub fn transpose2(d: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; d.len()];
    for i in 0..r {
        for j in 0..c {
            o[j * r + i] = d[i * c + j];
        }
    }
    o
}

// ---------------------------------------------------------------- matmul / linear / conv

/// Software f32 -> f16 -> f32 round trip (round-to-nearest-even), used by `@f16_auto`.
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let abs = f32::from_bits(bits & 0x7fff_ffff);
    if abs >= 65520.0 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    if abs < 6.103_515_6e-5 {
        let q = (abs * 16_777_216.0).round() / 16_777_216.0;
        return f32::from_bits(sign | q.to_bits());
    }
    let m = bits & 0x7fff_ffff;
    let lsb = (m >> 13) & 1;
    let r = (m + 0x0fff + lsb) & !0x1fffu32;
    f32::from_bits(sign | r)
}

fn quantized<'a>(d: &'a [f32], f16: bool) -> Cow<'a, [f32]> {
    if f16 {
        Cow::Owned(d.iter().map(|&v| round_f16(v)).collect())
    } else {
        Cow::Borrowed(d)
    }
}

/// Batched matrix product: `[.., M, K] >< [.., K, N]`. A rank-2 operand is shared across the batch.
pub fn matmul(a: &Tensor, b: &Tensor, ex: Exec) -> Res<Tensor> {
    let (ra, rb) = (a.rank(), b.rank());
    if ra < 2 || rb < 2 {
        return Err(format!("matmul needs rank >= 2, got {:?} >< {:?}", a.shape, b.shape));
    }
    let (m, k) = (a.shape[ra - 2], a.shape[ra - 1]);
    let (k2, n) = (b.shape[rb - 2], b.shape[rb - 1]);
    if k != k2 {
        return Err(format!(
            "matmul shape mismatch: {:?} >< {:?} (inner dims {} vs {})",
            a.shape, b.shape, k, k2
        ));
    }
    let ab = &a.shape[..ra - 2];
    let bb = &b.shape[..rb - 2];
    if !ab.is_empty() && !bb.is_empty() && ab != bb {
        return Err(format!("matmul batch dims differ: {:?} vs {:?}", ab, bb));
    }
    let batch: Vec<usize> = if ab.is_empty() { bb.to_vec() } else { ab.to_vec() };
    let nb: usize = batch.iter().product();
    let a_cow = quantized(&a.data, ex.f16);
    let b_cow = quantized(&b.data, ex.f16);
    let ad: &[f32] = &a_cow;
    let bd: &[f32] = &b_cow;
    let mut out = vec![0.0f32; nb * m * n];
    if bb.is_empty() {
        gemm(ad, bd, &mut out, nb * m, k, n, ex);
    } else {
        for i in 0..nb {
            let aoff = if ab.is_empty() { 0 } else { i * m * k };
            gemm(
                &ad[aoff..aoff + m * k],
                &bd[i * k * n..(i + 1) * k * n],
                &mut out[i * m * n..(i + 1) * m * n],
                m,
                k,
                n,
                ex,
            );
        }
    }
    let mut shape = batch;
    shape.push(m);
    shape.push(n);
    let mut dtype = DType::F32;
    if ex.f16 {
        for v in out.iter_mut() {
            *v = round_f16(*v);
        }
        dtype = DType::F16;
    }
    Ok(Tensor::new(shape, out, dtype, a.mem))
}

/// Fused `act(x >< w + b)` over `[.., in] x [in, out]`, bias/activation applied as an epilogue.
pub fn linear_fwd(x: &Tensor, w: &Tensor, b: Option<&Tensor>, act: Option<Un>, ex: Exec) -> Res<Tensor> {
    if w.rank() != 2 {
        return Err(format!("linear weight must be rank 2, got {:?}", w.shape));
    }
    let (inp, outd) = (w.shape[0], w.shape[1]);
    if x.rank() == 0 || x.shape[x.rank() - 1] != inp {
        return Err(format!("linear expects last dim {}, got tensor of shape {:?}", inp, x.shape));
    }
    if let Some(bt) = b {
        if bt.shape != vec![outd] {
            return Err(format!("linear bias must have shape [{}], got {:?}", outd, bt.shape));
        }
    }
    let rows: usize = x.shape[..x.rank() - 1].iter().product();
    let x_cow = quantized(&x.data, ex.f16);
    let w_cow = quantized(&w.data, ex.f16);
    let mut out = vec![0.0f32; rows * outd];
    gemm(&x_cow, &w_cow, &mut out, rows, inp, outd, ex);
    if b.is_some() || act.is_some() {
        let bias: Option<&[f32]> = b.map(|t| &t.data[..]);
        for r in 0..rows {
            for j in 0..outd {
                let mut v = out[r * outd + j];
                if let Some(bs) = bias {
                    v += bs[j];
                }
                if let Some(a) = act {
                    v = un_fwd(a, v);
                }
                out[r * outd + j] = v;
            }
        }
    }
    let mut dtype = DType::F32;
    if ex.f16 {
        for v in out.iter_mut() {
            *v = round_f16(*v);
        }
        dtype = DType::F16;
    }
    let mut shape = x.shape[..x.rank() - 1].to_vec();
    shape.push(outd);
    Ok(Tensor::new(shape, out, dtype, x.mem))
}

fn conv_dims(x: &Tensor, w: &Tensor, stride: usize, pad: usize) -> Res<(usize, usize, usize, usize, usize, usize, usize, usize, usize)> {
    if x.rank() != 4 || w.rank() != 4 {
        return Err(format!(
            "conv2d expects x [N,C,H,W] and weight [O,C,KH,KW], got {:?} and {:?}",
            x.shape, w.shape
        ));
    }
    let (n, c, h, wd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (o, c2, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
    if c != c2 {
        return Err(format!("conv2d: input has {} channels but weight expects {}", c, c2));
    }
    if stride == 0 {
        return Err("conv2d: stride must be >= 1".to_string());
    }
    if h + 2 * pad < kh || wd + 2 * pad < kw {
        return Err(format!("conv2d: kernel {}x{} larger than padded input {}x{}", kh, kw, h + 2 * pad, wd + 2 * pad));
    }
    let ho = (h + 2 * pad - kh) / stride + 1;
    let wo = (wd + 2 * pad - kw) / stride + 1;
    Ok((n, c, h, wd, o, kh, kw, ho, wo))
}

pub fn conv2d_fwd(x: &Tensor, w: &Tensor, b: Option<&Tensor>, stride: usize, pad: usize) -> Res<Tensor> {
    let (n, c, h, iw, o, kh, kw, ho, wo) = conv_dims(x, w, stride, pad)?;
    if let Some(bt) = b {
        if bt.shape != vec![o] {
            return Err(format!("conv2d bias must have shape [{}], got {:?}", o, bt.shape));
        }
    }
    let mut out = vec![0.0f32; n * o * ho * wo];
    let xd = &x.data;
    let wdt = &w.data;
    par_rows(&mut out, ho * wo, cpu_threads(), n * o * ho * wo * c * kh * kw, |row: usize, dst: &mut [f32]| {
        let ni = row / o;
        let oi = row % o;
        let bias = b.map(|t| t.data[oi]).unwrap_or(0.0);
        for y in 0..ho {
            for xo in 0..wo {
                let mut acc = bias;
                for ci in 0..c {
                    for ky in 0..kh {
                        let iy = y * stride + ky;
                        if iy < pad || iy - pad >= h {
                            continue;
                        }
                        let iy = iy - pad;
                        for kx in 0..kw {
                            let ix = xo * stride + kx;
                            if ix < pad || ix - pad >= iw {
                                continue;
                            }
                            let ix = ix - pad;
                            acc += xd[((ni * c + ci) * h + iy) * iw + ix] * wdt[((oi * c + ci) * kh + ky) * kw + kx];
                        }
                    }
                }
                dst[y * wo + xo] = acc;
            }
        }
    });
    Ok(Tensor::new(vec![n, o, ho, wo], out, DType::F32, x.mem))
}

/// Returns (dx, dw, db) for `conv2d_fwd`.
pub fn conv2d_bwd(x: &Tensor, w: &Tensor, dy: &Tensor, stride: usize, pad: usize) -> Res<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (n, c, h, iw, o, kh, kw, ho, wo) = conv_dims(x, w, stride, pad)?;
    let mut dx = vec![0.0f32; x.data.len()];
    let mut dw = vec![0.0f32; w.data.len()];
    let mut db = vec![0.0f32; o];
    for ni in 0..n {
        for oi in 0..o {
            for y in 0..ho {
                for xo in 0..wo {
                    let g = dy.data[((ni * o + oi) * ho + y) * wo + xo];
                    db[oi] += g;
                    for ci in 0..c {
                        for ky in 0..kh {
                            let iy = y * stride + ky;
                            if iy < pad || iy - pad >= h {
                                continue;
                            }
                            let iy = iy - pad;
                            for kx in 0..kw {
                                let ix = xo * stride + kx;
                                if ix < pad || ix - pad >= iw {
                                    continue;
                                }
                                let ix = ix - pad;
                                let xi = ((ni * c + ci) * h + iy) * iw + ix;
                                let wi = ((oi * c + ci) * kh + ky) * kw + kx;
                                dw[wi] += g * x.data[xi];
                                dx[xi] += g * w.data[wi];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((dx, dw, db))
}

// ---------------------------------------------------------------- row kernels

fn last_dim(x: &Tensor) -> usize {
    x.shape.last().copied().unwrap_or(1).max(1)
}

pub fn softmax_last(x: &Tensor) -> Tensor {
    let d = last_dim(x);
    let rows = x.data.len() / d;
    let mut out = vec![0.0f32; x.data.len()];
    for r in 0..rows {
        let row = &x.data[r * d..(r + 1) * d];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..d {
            let e = (row[j] - mx).exp();
            out[r * d + j] = e;
            sum += e;
        }
        for j in 0..d {
            out[r * d + j] /= sum;
        }
    }
    Tensor::new(x.shape.clone(), out, DType::F32, x.mem)
}

pub const LN_EPS: f32 = 1e-5;

/// LayerNorm over the last axis without affine parameters.
pub fn layernorm_fwd(x: &Tensor) -> Tensor {
    let d = last_dim(x);
    let rows = x.data.len() / d;
    let mut out = vec![0.0f32; x.data.len()];
    for r in 0..rows {
        let row = &x.data[r * d..(r + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rstd = 1.0 / (var + LN_EPS).sqrt();
        for j in 0..d {
            out[r * d + j] = (row[j] - mean) * rstd;
        }
    }
    Tensor::new(x.shape.clone(), out, DType::F32, x.mem)
}

/// Mean cross-entropy over rows of logits; `tg` holds one class index per row.
pub fn ce_fwd(x: &Tensor, tg: &Tensor) -> Res<Tensor> {
    if x.rank() == 0 {
        return Err("cross_entropy: logits must have rank >= 1".to_string());
    }
    let d = x.shape[x.rank() - 1];
    if d == 0 || x.data.is_empty() {
        return Err("cross_entropy: empty logits".to_string());
    }
    let rows = x.data.len() / d;
    if tg.data.len() != rows {
        return Err(format!(
            "cross_entropy: {} target(s) for {} row(s) of logits (shapes {:?} vs {:?})",
            tg.data.len(),
            rows,
            tg.shape,
            x.shape
        ));
    }
    let mut total = 0.0f64;
    for r in 0..rows {
        let row = &x.data[r * d..(r + 1) * d];
        let t = tg.data[r];
        if t < 0.0 || t as usize >= d {
            return Err(format!("cross_entropy: target {} out of range for {} classes", t, d));
        }
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + row.iter().map(|&v| (v - mx).exp()).sum::<f32>().ln();
        total += (lse - row[t as usize]) as f64;
    }
    Ok(Tensor::new(Vec::new(), vec![(total / rows as f64) as f32], DType::F32, x.mem))
}

// ---------------------------------------------------------------- element-wise functions

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Un {
    Neg,
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
}

const GELU_C: f32 = 0.797_884_6;

pub fn un_fwd(k: Un, x: f32) -> f32 {
    match k {
        Un::Neg => -x,
        Un::Relu => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        Un::Gelu => {
            let u = GELU_C * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + u.tanh())
        }
        Un::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Un::Tanh => x.tanh(),
        Un::Exp => x.exp(),
        Un::Log => x.ln(),
        Un::Sqrt => x.sqrt(),
    }
}

/// Derivative given the input `x` and the output `y` (only GELU and log need `x`).
pub fn un_grad(k: Un, x: f32, y: f32) -> f32 {
    match k {
        Un::Neg => -1.0,
        Un::Relu => {
            if y > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Un::Gelu => {
            let u = GELU_C * (x + 0.044715 * x * x * x);
            let t = u.tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
        }
        Un::Sigmoid => y * (1.0 - y),
        Un::Tanh => 1.0 - y * y,
        Un::Exp => y,
        Un::Log => 1.0 / x,
        Un::Sqrt => 0.5 / y,
    }
}

// ---------------------------------------------------------------- RNG

/// xorshift64* generator, good enough for initialisation, dropout and sampling.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (6.283_185_5 * u2).cos()
    }
}
