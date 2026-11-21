//! Tokenizer for `.nr` sources: emits Indent/Dedent/Newline for the indentation-based grammar.

use crate::tensor::Res;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    Directive(String),
    Fn,
    Let,
    Mut,
    Return,
    For,
    In,
    If,
    Elif,
    Else,
    NoGrad,
    True,
    False,
    And,
    Or,
    Not,
    Arrow,
    Pipe,
    Shift,
    MatMul,
    Pow,
    DotDot,
    Amp,
    Colon,
    Assign,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

fn push(out: &mut Vec<Token>, tok: Tok, line: usize) {
    out.push(Token { tok, line });
}

pub fn lex(src: &str) -> Res<Vec<Token>> {
    let src = src.replace("\r\n", "\n").replace('\r', "\n");
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut out: Vec<Token> = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut indents: Vec<usize> = vec![0];
    let mut depth = 0i32;
    let mut at_start = true;

    while i < n {
        if at_start && depth == 0 {
            // Measure indentation; blank and comment-only lines are skipped entirely.
            let mut col = 0usize;
            let mut j = i;
            while j < n && (chars[j] == ' ' || chars[j] == '\t') {
                col += if chars[j] == '\t' { 4 } else { 1 };
                j += 1;
            }
            if j >= n {
                break;
            }
            if chars[j] == '\n' {
                i = j + 1;
                line += 1;
                continue;
            }
            if chars[j] == '#' {
                while j < n && chars[j] != '\n' {
                    j += 1;
                }
                i = j;
                continue;
            }
            let cur = *indents.last().unwrap();
            if col > cur {
                indents.push(col);
                push(&mut out, Tok::Indent, line);
            } else if col < cur {
                while col < *indents.last().unwrap() {
                    indents.pop();
                    push(&mut out, Tok::Dedent, line);
                }
                if col != *indents.last().unwrap() {
                    return Err(format!("line {}: inconsistent indentation", line));
                }
            }
            i = j;
            at_start = false;
            continue;
        }

        let c = chars[i];
        let nx = if i + 1 < n { chars[i + 1] } else { '\0' };
        match c {
            ' ' | '\t' => i += 1,
            '\n' => {
                if depth == 0 {
                    push(&mut out, Tok::Newline, line);
                    at_start = true;
                }
                line += 1;
                i += 1;
            }
            '#' => {
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
            }
            '"' | '\'' => {
                let q = c;
                i += 1;
                let mut s = String::new();
                loop {
                    if i >= n || chars[i] == '\n' {
                        return Err(format!("line {}: unterminated string literal", line));
                    }
                    let ch = chars[i];
                    if ch == q {
                        i += 1;
                        break;
                    }
                    if ch == '\\' {
                        i += 1;
                        if i >= n {
                            return Err(format!("line {}: unterminated string literal", line));
                        }
                        s.push(match chars[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '0' => '\0',
                            other => other,
                        });
                        i += 1;
                    } else {
                        s.push(ch);
                        i += 1;
                    }
                }
                push(&mut out, Tok::Str(s), line);
            }
            '0'..='9' => {
                let start = i;
                let mut is_float = false;
                while i < n && (chars[i].is_ascii_digit() || chars[i] == '_') {
                    i += 1;
                }
                if i + 1 < n && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                    is_float = true;
                    i += 1;
                    while i < n && (chars[i].is_ascii_digit() || chars[i] == '_') {
                        i += 1;
                    }
                }
                if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < n && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < n && chars[j].is_ascii_digit() {
                        is_float = true;
                        i = j;
                        while i < n && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text: String = chars[start..i].iter().filter(|&&ch| ch != '_').collect();
                if is_float {
                    match text.parse::<f64>() {
                        Ok(v) => push(&mut out, Tok::Float(v), line),
                        Err(_) => return Err(format!("line {}: invalid number `{}`", line, text)),
                    }
                } else {
                    match text.parse::<i64>() {
                        Ok(v) => push(&mut out, Tok::Int(v), line),
                        Err(_) => return Err(format!("line {}: invalid integer `{}`", line, text)),
                    }
                }
            }
            '@' => {
                i += 1;
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                if start == i {
                    return Err(format!("line {}: `@` must be followed by a directive name", line));
                }
                let name: String = chars[start..i].iter().collect();
                push(&mut out, Tok::Directive(name), line);
            }
            c2 if c2.is_alphabetic() || c2 == '_' => {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let t = match word.as_str() {
                    "fn" => Tok::Fn,
                    "let" => Tok::Let,
                    "mut" => Tok::Mut,
                    "return" => Tok::Return,
                    "for" => Tok::For,
                    "in" => Tok::In,
                    "if" => Tok::If,
                    "elif" => Tok::Elif,
                    "else" => Tok::Else,
                    "no_grad" => Tok::NoGrad,
                    "true" => Tok::True,
                    "false" => Tok::False,
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    _ => Tok::Ident(word.clone()),
                };
                push(&mut out, t, line);
            }
            '(' => {
                depth += 1;
                push(&mut out, Tok::LParen, line);
                i += 1;
            }
            ')' => {
                if depth > 0 {
                    depth -= 1;
                }
                push(&mut out, Tok::RParen, line);
                i += 1;
            }
            '[' => {
                depth += 1;
                push(&mut out, Tok::LBracket, line);
                i += 1;
            }
            ']' => {
                if depth > 0 {
                    depth -= 1;
                }
                push(&mut out, Tok::RBracket, line);
                i += 1;
            }
            ',' => {
                push(&mut out, Tok::Comma, line);
                i += 1;
            }
            ':' => {
                push(&mut out, Tok::Colon, line);
                i += 1;
            }
            '+' => {
                if nx == '=' {
                    push(&mut out, Tok::PlusEq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Plus, line);
                    i += 1;
                }
            }
            '-' => {
                if nx == '>' {
                    push(&mut out, Tok::Arrow, line);
                    i += 2;
                } else if nx == '=' {
                    push(&mut out, Tok::MinusEq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Minus, line);
                    i += 1;
                }
            }
            '*' => {
                if nx == '*' {
                    push(&mut out, Tok::Pow, line);
                    i += 2;
                } else if nx == '=' {
                    push(&mut out, Tok::StarEq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Star, line);
                    i += 1;
                }
            }
            '/' => {
                if nx == '=' {
                    push(&mut out, Tok::SlashEq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Slash, line);
                    i += 1;
                }
            }
            '%' => {
                push(&mut out, Tok::Percent, line);
                i += 1;
            }
            '=' => {
                if nx == '=' {
                    push(&mut out, Tok::Eq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Assign, line);
                    i += 1;
                }
            }
            '!' => {
                if nx == '=' {
                    push(&mut out, Tok::Ne, line);
                    i += 2;
                } else {
                    return Err(format!("line {}: unexpected `!` (use `not` or `!=`)", line));
                }
            }
            '<' => {
                if nx == '=' {
                    push(&mut out, Tok::Le, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Lt, line);
                    i += 1;
                }
            }
            '>' => {
                if nx == '<' {
                    push(&mut out, Tok::MatMul, line);
                    i += 2;
                } else if nx == '>' {
                    push(&mut out, Tok::Shift, line);
                    i += 2;
                } else if nx == '=' {
                    push(&mut out, Tok::Ge, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Gt, line);
                    i += 1;
                }
            }
            '|' => {
                if nx == '>' {
                    push(&mut out, Tok::Pipe, line);
                    i += 2;
                } else {
                    return Err(format!("line {}: unexpected `|` (did you mean `|>`?)", line));
                }
            }
            '&' => {
                push(&mut out, Tok::Amp, line);
                i += 1;
            }
            '.' => {
                if nx == '.' {
                    push(&mut out, Tok::DotDot, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Dot, line);
                    i += 1;
                }
            }
            other => return Err(format!("line {}: unexpected character `{}`", line, other)),
        }
    }

    if depth > 0 {
        return Err(format!("line {}: unclosed bracket", line));
    }
    if !out.is_empty() && !matches!(out.last().map(|t| &t.tok), Some(Tok::Newline)) {
        push(&mut out, Tok::Newline, line);
    }
    while indents.len() > 1 {
        indents.pop();
        push(&mut out, Tok::Dedent, line);
    }
    push(&mut out, Tok::Eof, line);
    Ok(out)
}
