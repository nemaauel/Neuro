//! Abstract syntax tree for Neuro programs.

use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    MatMul,
    Shift,
    Pipe,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Var(String),
    List(Vec<Expr>),
    Unary(UnOp, Box<Expr>),
    /// `&x` (false) or `&mut x` (true)
    Ref(bool, Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Vec<IdxItem>),
    Field(Box<Expr>, String),
}

#[derive(Clone, Debug)]
pub enum IdxItem {
    At(Expr),
    /// `a..b`, `..b`, `a..`, `..`
    Range(Option<Expr>, Option<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DimSpec {
    Lit(usize),
    /// Symbolic dimension: bound on first use, then checked for consistency (`_` is a wildcard).
    Name(String),
}

#[derive(Clone, Debug)]
pub enum Type {
    Tensor(Option<Vec<DimSpec>>),
    F32,
    F16,
    I32,
    Usize,
    Bool,
    Str,
    Ref { mutable: bool, inner: Box<Type> },
    /// List, Layer, Tokenizer, Any
    Named(String),
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Vec<Stmt>,
    pub dirs: Vec<String>,
    pub line: usize,
}

#[derive(Clone, Debug)]
pub enum ForIter {
    Range(Expr, Expr),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub line: usize,
    pub kind: StmtKind,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let { name: String, mutable: bool, ty: Option<Type>, value: Expr },
    Assign { target: Expr, value: Expr },
    Expr(Expr),
    Return(Option<Expr>),
    If { arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>> },
    For { var: String, iter: ForIter, body: Vec<Stmt> },
    NoGrad(Vec<Stmt>),
    FnDef(Rc<FnDef>),
    Directive { names: Vec<String>, body: Vec<Stmt> },
    Sync,
}

pub const DIRECTIVES: &[&str] = &["cpu", "gpu", "npu", "ram", "vram", "simd", "sync", "f16_auto", "fuse"];
