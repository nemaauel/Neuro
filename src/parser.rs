//! Parser for the indentation-based Neuro grammar.

use crate::ast::*;
use crate::lexer::{lex, Tok, Token};
use crate::tensor::Res;
use std::rc::Rc;

pub fn parse_source(src: &str) -> Res<Vec<Stmt>> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.parse_program()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        let i = (self.pos + n).min(self.toks.len() - 1);
        &self.toks[i].tok
    }

    fn line(&self) -> usize {
        self.toks[self.pos].line
    }

    fn next(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.next();
            true
        } else {
            false
        }
    }

    fn err<T>(&self, msg: &str) -> Res<T> {
        Err(format!("line {}: {}", self.line(), msg))
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Res<()> {
        if self.eat(t) {
            Ok(())
        } else {
            self.err(&format!("expected `{}`, found {}", what, describe(self.peek())))
        }
    }

    fn expect_ident(&mut self) -> Res<String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.next();
                Ok(s)
            }
            other => self.err(&format!("expected a name, found {}", describe(&other))),
        }
    }

    fn end_simple(&mut self) -> Res<()> {
        match self.peek() {
            Tok::Newline => {
                self.next();
                Ok(())
            }
            Tok::Eof | Tok::Dedent => Ok(()),
            other => {
                let d = describe(other);
                self.err(&format!("unexpected {} at end of statement", d))
            }
        }
    }

    fn parse_program(&mut self) -> Res<Vec<Stmt>> {
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            if matches!(self.peek(), Tok::Newline | Tok::Dedent) {
                self.next();
                continue;
            }
            if matches!(self.peek(), Tok::Indent) {
                return self.err("unexpected indentation");
            }
            out.push(self.parse_stmt()?);
        }
        Ok(out)
    }

    fn parse_block(&mut self) -> Res<Vec<Stmt>> {
        if self.eat(&Tok::Newline) {
            if !self.eat(&Tok::Indent) {
                return self.err("expected an indented block");
            }
            let mut body = Vec::new();
            while !matches!(self.peek(), Tok::Dedent | Tok::Eof) {
                body.push(self.parse_stmt()?);
            }
            self.eat(&Tok::Dedent);
            Ok(body)
        } else {
            Ok(vec![self.parse_stmt()?])
        }
    }

    fn parse_stmt(&mut self) -> Res<Stmt> {
        let line = self.line();
        let kind = match self.peek().clone() {
            Tok::Fn => StmtKind::FnDef(Rc::new(self.parse_fn(Vec::new())?)),
            Tok::Directive(_) => return self.parse_directives(Vec::new()),
            Tok::Let | Tok::Mut => self.parse_let()?,
            Tok::Return => {
                self.next();
                let kind = if matches!(self.peek(), Tok::Newline | Tok::Eof | Tok::Dedent) {
                    StmtKind::Return(None)
                } else {
                    StmtKind::Return(Some(self.parse_expr()?))
                };
                self.end_simple()?;
                kind
            }
            Tok::If => self.parse_if()?,
            Tok::For => self.parse_for()?,
            Tok::NoGrad => {
                self.next();
                self.expect(&Tok::Colon, ":")?;
                StmtKind::NoGrad(self.parse_block()?)
            }
            _ => self.parse_simple()?,
        };
        Ok(Stmt { line, kind })
    }

    fn parse_directives(&mut self, mut names: Vec<String>) -> Res<Stmt> {
        let line = self.line();
        while let Tok::Directive(n) = self.peek().clone() {
            if !DIRECTIVES.contains(&n.as_str()) {
                return self.err(&format!("unknown directive `@{}`", n));
            }
            self.next();
            names.push(n);
        }
        if self.eat(&Tok::Colon) {
            let body = self.parse_block()?;
            return Ok(Stmt { line, kind: StmtKind::Directive { names, body } });
        }
        self.eat(&Tok::Newline);
        if names.len() == 1 && names[0] == "sync" {
            return Ok(Stmt { line, kind: StmtKind::Sync });
        }
        match self.peek().clone() {
            Tok::Directive(_) => self.parse_directives(names),
            Tok::Fn => {
                let f = self.parse_fn(names)?;
                Ok(Stmt { line, kind: StmtKind::FnDef(Rc::new(f)) })
            }
            Tok::Eof | Tok::Dedent => self.err("directive is not followed by a statement"),
            _ => {
                let st = self.parse_stmt()?;
                Ok(Stmt { line, kind: StmtKind::Directive { names, body: vec![st] } })
            }
        }
    }

    fn parse_fn(&mut self, dirs: Vec<String>) -> Res<FnDef> {
        let line = self.line();
        self.next(); // fn
        let name = self.expect_ident()?;
        self.expect(&Tok::LParen, "(")?;
        let mut params = Vec::new();
        loop {
            if self.eat(&Tok::RParen) {
                break;
            }
            let pname = self.expect_ident()?;
            let ty = if self.eat(&Tok::Colon) { Some(self.parse_type()?) } else { None };
            params.push(Param { name: pname, ty });
            if !self.eat(&Tok::Comma) {
                self.expect(&Tok::RParen, ")")?;
                break;
            }
        }
        let ret = if self.eat(&Tok::Arrow) { Some(self.parse_type()?) } else { None };
        self.expect(&Tok::Colon, ":")?;
        let body = self.parse_block()?;
        Ok(FnDef { name, params, ret, body, dirs, line })
    }

    fn parse_type(&mut self) -> Res<Type> {
        if self.eat(&Tok::Amp) {
            let mutable = self.eat(&Tok::Mut);
            let inner = self.parse_type()?;
            return Ok(Type::Ref { mutable, inner: Box::new(inner) });
        }
        let name = self.expect_ident()?;
        match name.as_str() {
            "Tensor" => {
                if self.eat(&Tok::LBracket) {
                    let mut dims = Vec::new();
                    loop {
                        if self.eat(&Tok::RBracket) {
                            break;
                        }
                        match self.next().tok {
                            Tok::Int(n) if n >= 0 => dims.push(DimSpec::Lit(n as usize)),
                            Tok::Ident(s) => dims.push(DimSpec::Name(s)),
                            _ => return self.err("bad tensor dimension in type (use a number or a name)"),
                        }
                        if !self.eat(&Tok::Comma) {
                            self.expect(&Tok::RBracket, "]")?;
                            break;
                        }
                    }
                    Ok(Type::Tensor(Some(dims)))
                } else {
                    Ok(Type::Tensor(None))
                }
            }
            "f32" => Ok(Type::F32),
            "f16" => Ok(Type::F16),
            "i32" => Ok(Type::I32),
            "usize" => Ok(Type::Usize),
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            "List" | "Layer" | "Tokenizer" | "Any" => Ok(Type::Named(name)),
            _ => self.err(&format!("unknown type `{}`", name)),
        }
    }

    fn parse_let(&mut self) -> Res<StmtKind> {
        let mut mutable = false;
        if self.eat(&Tok::Let) {
            if self.eat(&Tok::Mut) {
                mutable = true;
            }
        } else {
            self.next(); // `mut name = ...`
            mutable = true;
        }
        let name = self.expect_ident()?;
        let ty = if self.eat(&Tok::Colon) { Some(self.parse_type()?) } else { None };
        self.expect(&Tok::Assign, "=")?;
        let value = self.parse_expr()?;
        self.end_simple()?;
        Ok(StmtKind::Let { name, mutable, ty, value })
    }

    fn parse_if(&mut self) -> Res<StmtKind> {
        self.next(); // if
        let mut arms = Vec::new();
        let cond = self.parse_expr()?;
        self.expect(&Tok::Colon, ":")?;
        let body = self.parse_block()?;
        arms.push((cond, body));
        let mut els = None;
        loop {
            if self.eat(&Tok::Elif) {
                let c = self.parse_expr()?;
                self.expect(&Tok::Colon, ":")?;
                let b = self.parse_block()?;
                arms.push((c, b));
            } else if self.eat(&Tok::Else) {
                self.expect(&Tok::Colon, ":")?;
                els = Some(self.parse_block()?);
                break;
            } else {
                break;
            }
        }
        Ok(StmtKind::If { arms, els })
    }

    fn parse_for(&mut self) -> Res<StmtKind> {
        self.next(); // for
        let var = self.expect_ident()?;
        self.expect(&Tok::In, "in")?;
        let first = self.parse_expr()?;
        let iter = if self.eat(&Tok::DotDot) {
            let end = self.parse_expr()?;
            ForIter::Range(first, end)
        } else {
            ForIter::Expr(first)
        };
        self.expect(&Tok::Colon, ":")?;
        let body = self.parse_block()?;
        Ok(StmtKind::For { var, iter, body })
    }

    fn parse_simple(&mut self) -> Res<StmtKind> {
        let e = self.parse_expr()?;
        let compound = match self.peek() {
            Tok::PlusEq => Some(BinOp::Add),
            Tok::MinusEq => Some(BinOp::Sub),
            Tok::StarEq => Some(BinOp::Mul),
            Tok::SlashEq => Some(BinOp::Div),
            _ => None,
        };
        let kind = if matches!(self.peek(), Tok::Assign) || compound.is_some() {
            if !matches!(e, Expr::Var(_) | Expr::Index(_, _)) {
                return self.err("invalid assignment target");
            }
            self.next();
            let rhs = self.parse_expr()?;
            let value = match compound {
                Some(op) => Expr::Bin(op, Box::new(e.clone()), Box::new(rhs)),
                None => rhs,
            };
            StmtKind::Assign { target: e, value }
        } else {
            StmtKind::Expr(e)
        };
        self.end_simple()?;
        Ok(kind)
    }

    // ------------------------------------------------------------ expressions

    fn parse_expr(&mut self) -> Res<Expr> {
        self.parse_pipe()
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Bin(op, Box::new(l), Box::new(r))
    }

    fn parse_pipe(&mut self) -> Res<Expr> {
        let mut l = self.parse_or()?;
        while self.eat(&Tok::Pipe) {
            let r = self.parse_or()?;
            l = Parser::bin(BinOp::Pipe, l, r);
        }
        Ok(l)
    }

    fn parse_or(&mut self) -> Res<Expr> {
        let mut l = self.parse_and()?;
        while self.eat(&Tok::Or) {
            let r = self.parse_and()?;
            l = Parser::bin(BinOp::Or, l, r);
        }
        Ok(l)
    }

    fn parse_and(&mut self) -> Res<Expr> {
        let mut l = self.parse_not()?;
        while self.eat(&Tok::And) {
            let r = self.parse_not()?;
            l = Parser::bin(BinOp::And, l, r);
        }
        Ok(l)
    }

    fn parse_not(&mut self) -> Res<Expr> {
        if self.eat(&Tok::Not) {
            let e = self.parse_not()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Res<Expr> {
        let mut l = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                Tok::Eq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Gt => BinOp::Gt,
                Tok::Le => BinOp::Le,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.next();
            let r = self.parse_shift()?;
            l = Parser::bin(op, l, r);
        }
        Ok(l)
    }

    fn parse_shift(&mut self) -> Res<Expr> {
        let mut l = self.parse_add()?;
        while self.eat(&Tok::Shift) {
            let r = self.parse_add()?;
            l = Parser::bin(BinOp::Shift, l, r);
        }
        Ok(l)
    }

    fn parse_add(&mut self) -> Res<Expr> {
        let mut l = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.next();
            let r = self.parse_mul()?;
            l = Parser::bin(op, l, r);
        }
        Ok(l)
    }

    fn parse_mul(&mut self) -> Res<Expr> {
        let mut l = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                Tok::MatMul => BinOp::MatMul,
                _ => break,
            };
            self.next();
            let r = self.parse_unary()?;
            l = Parser::bin(op, l, r);
        }
        Ok(l)
    }

    fn parse_unary(&mut self) -> Res<Expr> {
        if self.eat(&Tok::Minus) {
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Neg, Box::new(e)));
        }
        if self.eat(&Tok::Amp) {
            let mutable = self.eat(&Tok::Mut);
            let e = self.parse_unary()?;
            return Ok(Expr::Ref(mutable, Box::new(e)));
        }
        self.parse_pow()
    }

    fn parse_pow(&mut self) -> Res<Expr> {
        let base = self.parse_postfix()?;
        if self.eat(&Tok::Pow) {
            let exp = self.parse_unary()?;
            return Ok(Parser::bin(BinOp::Pow, base, exp));
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Res<Expr> {
        let mut e = self.parse_atom()?;
        loop {
            match self.peek() {
                Tok::LParen => {
                    self.next();
                    let mut args = Vec::new();
                    loop {
                        if self.eat(&Tok::RParen) {
                            break;
                        }
                        args.push(self.parse_expr()?);
                        if !self.eat(&Tok::Comma) {
                            self.expect(&Tok::RParen, ")")?;
                            break;
                        }
                    }
                    e = Expr::Call(Box::new(e), args);
                }
                Tok::LBracket => {
                    self.next();
                    let items = self.parse_index_items()?;
                    e = Expr::Index(Box::new(e), items);
                }
                Tok::Dot => {
                    self.next();
                    let name = self.expect_ident()?;
                    e = Expr::Field(Box::new(e), name);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn range_end(&mut self) -> Res<Option<Expr>> {
        if matches!(self.peek(), Tok::Comma | Tok::RBracket) {
            Ok(None)
        } else {
            Ok(Some(self.parse_expr()?))
        }
    }

    fn parse_index_items(&mut self) -> Res<Vec<IdxItem>> {
        let mut items = Vec::new();
        loop {
            if self.eat(&Tok::DotDot) {
                let end = self.range_end()?;
                items.push(IdxItem::Range(None, end));
            } else {
                let a = self.parse_expr()?;
                if self.eat(&Tok::DotDot) {
                    let end = self.range_end()?;
                    items.push(IdxItem::Range(Some(a), end));
                } else {
                    items.push(IdxItem::At(a));
                }
            }
            if self.eat(&Tok::Comma) {
                continue;
            }
            break;
        }
        self.expect(&Tok::RBracket, "]")?;
        Ok(items)
    }

    fn parse_atom(&mut self) -> Res<Expr> {
        let t = self.next();
        match t.tok {
            Tok::Int(i) => Ok(Expr::Int(i)),
            Tok::Float(f) => Ok(Expr::Float(f)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::Ident(s) => Ok(Expr::Var(s)),
            Tok::LParen => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen, ")")?;
                Ok(e)
            }
            Tok::LBracket => {
                let mut items = Vec::new();
                loop {
                    if self.eat(&Tok::RBracket) {
                        break;
                    }
                    items.push(self.parse_expr()?);
                    if !self.eat(&Tok::Comma) {
                        self.expect(&Tok::RBracket, "]")?;
                        break;
                    }
                }
                Ok(Expr::List(items))
            }
            other => Err(format!("line {}: unexpected {} in expression", t.line, describe(&other))),
        }
    }

    #[allow(dead_code)]
    fn lookahead_is(&self, n: usize, t: &Tok) -> bool {
        self.peek_at(n) == t
    }
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Newline => "end of line".to_string(),
        Tok::Indent => "indent".to_string(),
        Tok::Dedent => "dedent".to_string(),
        Tok::Eof => "end of file".to_string(),
        Tok::Ident(s) => format!("`{}`", s),
        Tok::Int(i) => format!("`{}`", i),
        Tok::Str(_) => "a string".to_string(),
        other => format!("`{:?}`", other),
    }
}
