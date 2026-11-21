//! Binary model format (`NRMD`): tensors, layers (with their architecture) and nested lists.

use crate::autograd::{get, leaf};
use crate::tensor::{DType, Mem, Res, Tensor};
use crate::value::{list, Layer, LayerKind, Value};
use std::rc::Rc;

const MAGIC: &[u8; 4] = b"NRMD";
const VERSION: u32 = 1;

const TAG_TENSOR: u8 = 0;
const TAG_LIST: u8 = 1;
const TAG_LAYER: u8 = 2;

fn w_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn w_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn dtype_code(d: DType) -> u8 {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::I32 => 2,
    }
}

fn w_tensor(o: &mut Vec<u8>, t: &Tensor) {
    w_u32(o, t.rank() as u32);
    for &d in t.shape.iter() {
        w_u64(o, d as u64);
    }
    o.push(dtype_code(t.dtype));
    for &x in t.data.iter() {
        o.extend_from_slice(&x.to_le_bytes());
    }
}

fn w_value(o: &mut Vec<u8>, v: &Value) -> Res<()> {
    match v {
        Value::Tensor(var) => {
            let t = get(var)?;
            o.push(TAG_TENSOR);
            w_tensor(o, &t);
            Ok(())
        }
        Value::List(l) => {
            let items = l.borrow();
            o.push(TAG_LIST);
            w_u32(o, items.len() as u32);
            for it in items.iter() {
                w_value(o, it)?;
            }
            Ok(())
        }
        Value::Layer(l) => {
            o.push(TAG_LAYER);
            let (code, hyper): (u8, Vec<usize>) = match &l.kind {
                LayerKind::Linear { inp, out } => (0, vec![*inp, *out]),
                LayerKind::Conv2d { inp, out, kh, kw, stride, pad } => (1, vec![*inp, *out, *kh, *kw, *stride, *pad]),
                LayerKind::Embed { vocab, dim } => (2, vec![*vocab, *dim]),
                LayerKind::PosEmbed { len, dim } => (3, vec![*len, *dim]),
            };
            o.push(code);
            w_u32(o, hyper.len() as u32);
            for h in hyper {
                w_u64(o, h as u64);
            }
            w_u32(o, l.params.len() as u32);
            for p in l.params.iter() {
                let t = get(p)?;
                w_tensor(o, &t);
            }
            Ok(())
        }
        other => Err(format!("save: cannot serialize a value of type {}", other.type_name())),
    }
}

pub fn save_value(v: &Value, path: &str) -> Res<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create directory '{}': {}", parent.display(), e))?;
        }
    }
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(MAGIC);
    w_u32(&mut out, VERSION);
    w_value(&mut out, v)?;
    std::fs::write(path, out).map_err(|e| format!("cannot write '{}': {}", path, e))
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let b: &'a [u8] = self.b;
        let end = self.p.checked_add(n).ok_or_else(|| "model file is corrupt".to_string())?;
        if end > b.len() {
            return Err("model file is truncated or corrupt".to_string());
        }
        let s = &b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Res<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Res<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }

    fn tensor(&mut self, mem: Mem) -> Res<Tensor> {
        let rank = self.u32()? as usize;
        if rank > 16 {
            return Err("model file is corrupt (tensor rank too large)".to_string());
        }
        let mut shape = Vec::with_capacity(rank);
        let mut n: usize = 1;
        for _ in 0..rank {
            let d = self.u64()? as usize;
            n = n.checked_mul(d).ok_or_else(|| "model file is corrupt (tensor too large)".to_string())?;
            shape.push(d);
        }
        let dtype = match self.u8()? {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::I32,
            _ => return Err("model file is corrupt (unknown dtype)".to_string()),
        };
        let bytes = n.checked_mul(4).ok_or_else(|| "model file is corrupt".to_string())?;
        let raw = self.take(bytes)?;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]));
        }
        Ok(Tensor::new(shape, data, dtype, mem))
    }

    fn value(&mut self, mem: Mem) -> Res<Value> {
        match self.u8()? {
            TAG_TENSOR => {
                let t = self.tensor(mem)?;
                Ok(Value::Tensor(leaf(t, true)))
            }
            TAG_LIST => {
                let n = self.u32()? as usize;
                let mut items = Vec::new();
                for _ in 0..n {
                    items.push(self.value(mem)?);
                }
                Ok(list(items))
            }
            TAG_LAYER => {
                let code = self.u8()?;
                let nh = self.u32()? as usize;
                let mut h = Vec::with_capacity(nh);
                for _ in 0..nh {
                    h.push(self.u64()? as usize);
                }
                let np = self.u32()? as usize;
                let mut params = Vec::with_capacity(np);
                for _ in 0..np {
                    params.push(self.tensor(mem)?);
                }
                let bad = || "model file is corrupt (layer description does not match its weights)".to_string();
                let (kind, expect): (LayerKind, Vec<Vec<usize>>) = match (code, h.len()) {
                    (0, 2) => (LayerKind::Linear { inp: h[0], out: h[1] }, vec![vec![h[0], h[1]], vec![h[1]]]),
                    (1, 6) => (
                        LayerKind::Conv2d { inp: h[0], out: h[1], kh: h[2], kw: h[3], stride: h[4], pad: h[5] },
                        vec![vec![h[1], h[0], h[2], h[3]], vec![h[1]]],
                    ),
                    (2, 2) => (LayerKind::Embed { vocab: h[0], dim: h[1] }, vec![vec![h[0], h[1]]]),
                    (3, 2) => (LayerKind::PosEmbed { len: h[0], dim: h[1] }, vec![vec![h[0], h[1]]]),
                    _ => return Err(bad()),
                };
                if params.len() != expect.len() || params.iter().zip(expect.iter()).any(|(p, e)| &p.shape != e) {
                    return Err(bad());
                }
                let vars = params.into_iter().map(|t| leaf(t, true)).collect();
                Ok(Value::Layer(Rc::new(Layer { kind, params: vars })))
            }
            _ => Err("model file is corrupt (unknown record tag)".to_string()),
        }
    }
}

/// Loads a model saved with `save_value`; tensors are allocated in `mem`.
pub fn load_value(path: &str, mem: Mem) -> Res<Value> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read '{}': {}", path, e))?;
    let mut rd = Rd { b: &bytes, p: 0 };
    if rd.take(4)? != MAGIC {
        return Err(format!("'{}' is not a Neuro model file", path));
    }
    let version = rd.u32()?;
    if version != VERSION {
        return Err(format!("'{}': unsupported model version {}", path, version));
    }
    rd.value(mem)
}
