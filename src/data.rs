//! Tokenizer, binary token datasets (`NRBN`) and streaming readers.
//!
//! `.bin` layout: "NRBN", kind byte (0 = i32, 1 = u16, 2 = f32), u64 count (LE), raw LE data.
//! Files without the magic are read as raw little-endian i32.

use crate::tensor::{DType, Res};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

const MAGIC: &[u8; 4] = b"NRBN";
const HEADER_LEN: usize = 13;

// ---------------------------------------------------------------- tokenizer

pub struct Tokenizer {
    pub tokens: Vec<String>,
    map: HashMap<String, u32>,
    max_len: usize,
    byte_level: bool,
    unk: Option<u32>,
}

fn escape(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\t' => "\\t".to_string(),
        '\r' => "\\r".to_string(),
        '\\' => "\\\\".to_string(),
        other => other.to_string(),
    }
}

fn unescape(line: &str) -> String {
    let mut out = String::new();
    let mut it = line.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

impl Tokenizer {
    /// Byte-level tokenizer: ids are raw UTF-8 bytes, vocabulary size 256.
    pub fn bytes() -> Tokenizer {
        Tokenizer { tokens: Vec::new(), map: HashMap::new(), max_len: 1, byte_level: true, unk: None }
    }

    pub fn from_tokens(tokens: Vec<String>) -> Res<Tokenizer> {
        if tokens.is_empty() {
            return Err("vocabulary is empty".to_string());
        }
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut max_len = 1usize;
        for (i, t) in tokens.iter().enumerate() {
            if map.insert(t.clone(), i as u32).is_some() {
                return Err(format!("duplicate token {:?} in vocabulary", t));
            }
            max_len = max_len.max(t.chars().count());
        }
        let unk = map.get("<unk>").copied();
        Ok(Tokenizer { tokens, map, max_len, byte_level: false, unk })
    }

    /// One token per line (`\n`, `\t`, `\\` escapes); the line number is the id.
    pub fn from_file(path: &str) -> Res<Tokenizer> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read vocabulary '{}': {}", path, e))?;
        let text = text.replace("\r\n", "\n");
        let tokens: Vec<String> = text.split('\n').filter(|l| !l.is_empty()).map(unescape).collect();
        Tokenizer::from_tokens(tokens)
    }

    pub fn size(&self) -> usize {
        if self.byte_level {
            256
        } else {
            self.tokens.len()
        }
    }

    fn lookup_char(&self, ch: char) -> Option<u32> {
        let mut buf = [0u8; 4];
        let s: &str = ch.encode_utf8(&mut buf);
        self.map.get(s).copied()
    }

    /// Greedy longest-match encoding.
    pub fn encode(&self, text: &str) -> Res<Vec<i32>> {
        if self.byte_level {
            return Ok(text.bytes().map(|b| b as i32).collect());
        }
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::with_capacity(chars.len());
        let mut i = 0usize;
        while i < chars.len() {
            let maxl = self.max_len.min(chars.len() - i);
            let mut found: Option<(u32, usize)> = None;
            for l in (1..=maxl).rev() {
                if l == 1 {
                    found = self.lookup_char(chars[i]).map(|id| (id, 1));
                } else {
                    let cand: String = chars[i..i + l].iter().collect();
                    found = self.map.get(&cand).map(|&id| (id, l));
                }
                if found.is_some() {
                    break;
                }
            }
            match found {
                Some((id, l)) => {
                    out.push(id as i32);
                    i += l;
                }
                None => match self.unk {
                    Some(u) => {
                        out.push(u as i32);
                        i += 1;
                    }
                    None => return Err(format!("character {:?} is not in the vocabulary (add <unk> or rebuild it)", chars[i])),
                },
            }
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[i32]) -> Res<String> {
        if self.byte_level {
            let mut bytes = Vec::with_capacity(ids.len());
            for &id in ids {
                if !(0..=255).contains(&id) {
                    return Err(format!("token id {} is not a byte (0..255)", id));
                }
                bytes.push(id as u8);
            }
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        let mut out = String::new();
        for &id in ids {
            if id < 0 || id as usize >= self.tokens.len() {
                return Err(format!("token id {} out of range for a vocabulary of {}", id, self.tokens.len()));
            }
            out.push_str(&self.tokens[id as usize]);
        }
        Ok(out)
    }
}

fn ensure_parent(path: &str) -> Res<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create directory '{}': {}", parent.display(), e))?;
        }
    }
    Ok(())
}

/// Writes a character-level vocabulary file for `text`; returns its size.
pub fn build_vocab(text: &str, path: &str) -> Res<usize> {
    ensure_parent(path)?;
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();
    let body: Vec<String> = chars.iter().map(|&c| escape(c)).collect();
    std::fs::write(path, body.join("\n") + "\n").map_err(|e| format!("cannot write '{}': {}", path, e))?;
    Ok(chars.len())
}

// ---------------------------------------------------------------- binary datasets

fn elem_size(kind: u8) -> usize {
    if kind == 1 {
        2
    } else {
        4
    }
}

fn decode_elems(buf: &[u8], kind: u8) -> Vec<f32> {
    let es = elem_size(kind);
    let count = buf.len() / es;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &buf[i * es..(i + 1) * es];
        let v = match kind {
            0 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            1 => u16::from_le_bytes([b[0], b[1]]) as f32,
            _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        };
        out.push(v);
    }
    out
}

pub fn load_bin(path: &str) -> Res<(Vec<f32>, DType)> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read '{}': {}", path, e))?;
    if bytes.len() >= HEADER_LEN && &bytes[..4] == MAGIC {
        let kind = bytes[4];
        if kind > 2 {
            return Err(format!("'{}': unknown NRBN data kind {}", path, kind));
        }
        let dtype = if kind == 2 { DType::F32 } else { DType::I32 };
        return Ok((decode_elems(&bytes[HEADER_LEN..], kind), dtype));
    }
    Ok((decode_elems(&bytes, 0), DType::I32))
}

pub fn save_bin(path: &str, data: &[f32], dtype: DType) -> Res<()> {
    ensure_parent(path)?;
    let kind: u8 = if dtype == DType::I32 { 0 } else { 2 };
    let mut out: Vec<u8> = Vec::with_capacity(HEADER_LEN + data.len() * 4);
    out.extend_from_slice(MAGIC);
    out.push(kind);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for &v in data {
        if kind == 0 {
            out.extend_from_slice(&(v as i32).to_le_bytes());
        } else {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write '{}': {}", path, e))
}

// ---------------------------------------------------------------- streaming

pub enum StreamItem {
    Text(String),
    Tokens(Vec<f32>, DType),
}

pub enum Stream {
    Text { reader: BufReader<File>, chunk: usize, carry: Vec<u8>, done: bool },
    Tokens { reader: BufReader<File>, chunk: usize, kind: u8 },
}

impl Stream {
    /// `.bin` files stream token tensors of `chunk` elements, anything else streams text of ~`chunk` bytes.
    pub fn open(path: &str, chunk: usize) -> Res<Stream> {
        if chunk == 0 {
            return Err("stream_data: batch_size must be greater than 0".to_string());
        }
        let f = File::open(path).map_err(|e| format!("cannot open '{}': {}", path, e))?;
        let mut reader = BufReader::new(f);
        if path.ends_with(".bin") {
            let mut kind = 0u8;
            let skip = {
                let head = reader.fill_buf().map_err(|e| format!("cannot read '{}': {}", path, e))?;
                if head.len() >= HEADER_LEN && &head[..4] == MAGIC {
                    kind = head[4];
                    true
                } else {
                    false
                }
            };
            if skip {
                if kind > 2 {
                    return Err(format!("'{}': unknown NRBN data kind {}", path, kind));
                }
                reader.consume(HEADER_LEN);
            }
            Ok(Stream::Tokens { reader, chunk, kind })
        } else {
            Ok(Stream::Text { reader, chunk, carry: Vec::new(), done: false })
        }
    }

    pub fn next(&mut self) -> Res<Option<StreamItem>> {
        match self {
            Stream::Tokens { reader, chunk, kind } => {
                let es = elem_size(*kind);
                let mut buf = vec![0u8; *chunk * es];
                let mut filled = 0usize;
                while filled < buf.len() {
                    let n = reader.read(&mut buf[filled..]).map_err(|e| format!("stream read error: {}", e))?;
                    if n == 0 {
                        break;
                    }
                    filled += n;
                }
                let usable = filled - filled % es;
                if usable == 0 {
                    return Ok(None);
                }
                let dtype = if *kind == 2 { DType::F32 } else { DType::I32 };
                Ok(Some(StreamItem::Tokens(decode_elems(&buf[..usable], *kind), dtype)))
            }
            Stream::Text { reader, chunk, carry, done } => {
                if *done && carry.is_empty() {
                    return Ok(None);
                }
                let want = (*chunk).max(4);
                let mut buf = std::mem::take(carry);
                if !*done {
                    let mut tmp = vec![0u8; want];
                    let mut filled = 0usize;
                    while filled < want {
                        let n = reader.read(&mut tmp[filled..]).map_err(|e| format!("stream read error: {}", e))?;
                        if n == 0 {
                            break;
                        }
                        filled += n;
                    }
                    buf.extend_from_slice(&tmp[..filled]);
                    if filled < want {
                        *done = true;
                    }
                }
                if buf.is_empty() {
                    return Ok(None);
                }
                let check = std::str::from_utf8(&buf)
                    .map(|_| ())
                    .map_err(|e| (e.valid_up_to(), e.error_len().is_some()));
                match check {
                    Ok(()) => Ok(Some(StreamItem::Text(String::from_utf8(buf).unwrap_or_default()))),
                    Err((valid, invalid)) => {
                        if *done || invalid {
                            Ok(Some(StreamItem::Text(String::from_utf8_lossy(&buf).into_owned())))
                        } else {
                            let rest = buf.split_off(valid);
                            *carry = rest;
                            Ok(Some(StreamItem::Text(String::from_utf8(buf).unwrap_or_default())))
                        }
                    }
                }
            }
        }
    }
}
