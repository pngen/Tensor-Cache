//! Tensor Cache command-line tool and runtime processes.

mod args;
mod commands;
mod server;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (sub, flags) = args::parse(&argv);
    let code = match commands::dispatch(&sub, &flags) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    };
    std::process::exit(code);
}
