use neuro::{checker, gpu, interp::Interp, parser, tensor};
use std::path::Path;
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_help() {
    println!("Neuro v{}", VERSION);
    println!("Usage:");
    println!("  neuro [run] <script.nr>   Run a Neuro source file");
    println!("  neuro check <script.nr>   Type-check and validate a script without running it");
    println!("  neuro info                Display runtime hardware and backend status");
    println!("  neuro --version           Print version information");
}

fn run_cli() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: neuro [run|check|info] <script.nr>".to_string());
    }

    match args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            return Ok(());
        }
        "-V" | "--version" | "version" => {
            println!("neuro {}", VERSION);
            return Ok(());
        }
        "info" => {
            let gpu_status = gpu::gpu_device_name().unwrap_or("none (CPU fallback)");
            println!("Neuro v{}", VERSION);
            println!("CPU threads : {}", tensor::cpu_threads());
            println!("Vulkan GPU  : {}", gpu_status);
            return Ok(());
        }
        _ => {}
    }

    let (check_only, file_path) = match args[0].as_str() {
        "check" => {
            if args.len() != 2 {
                return Err("usage: neuro check <script.nr>".to_string());
            }
            (true, &args[1])
        }
        "run" => {
            if args.len() != 2 {
                return Err("usage: neuro run <script.nr>".to_string());
            }
            (false, &args[1])
        }
        _ => {
            if args.len() != 1 {
                return Err("usage: neuro [run|check|info] <script.nr>".to_string());
            }
            (false, &args[0])
        }
    };

    let ext_ok = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("nr"))
        .unwrap_or(false);

    if !ext_ok {
        return Err(format!(
            "invalid file '{}': Neuro source files must use the '.nr' extension",
            file_path
        ));
    }

    let src = std::fs::read_to_string(file_path)
        .map_err(|e| format!("cannot read '{}': {}", file_path, e))?;

    let program = parser::parse_source(&src)?;
    checker::check(&program)?;

    if check_only {
        println!("OK: '{}' passed static checks.", file_path);
        return Ok(());
    }

    let mut runtime = Interp::new();
    runtime.run(&program)
}

fn main() {
    if let Err(err) = run_cli() {
        eprintln!("[neuro error] {}", err);
        process::exit(1);
    }
}