use chunker::chunking::ChunkStream;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

#[cfg(feature = "telemetry")]
use opentelemetry::global;
#[cfg(feature = "telemetry")]
use opentelemetry_sdk::propagation::TraceContextPropagator;
#[cfg(feature = "telemetry")]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "telemetry")]
use tracing_subscriber::util::SubscriberInitExt;

#[cfg(not(feature = "telemetry"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[cfg(feature = "telemetry")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up OpenTelemetry
    global::set_text_map_propagator(TraceContextPropagator::new());

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(opentelemetry_otlp::new_exporter().tonic())
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    // Initialize subscriber with both fmt (logs) and telemetry (traces)
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(telemetry)
        .init();

    let result = run();

    // Ensure all spans are exported before exit
    global::shutdown_tracer_provider();
    
    result
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if not already done by telemetry
    #[cfg(not(feature = "telemetry"))]
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    
    let reader: Box<dyn io::Read> = if args.len() > 1 && args[1] != "-" {
        let path = PathBuf::from(&args[1]);
        let file = File::open(&path)?;
        info!(path = ?path, "starting_chunking_file");
        Box::new(BufReader::new(file))
    } else {
        info!("starting_chunking_stdin");
        Box::new(BufReader::new(io::stdin()))
    };

    let start = Instant::now();
    // Use default chunking options
    let stream = ChunkStream::new(reader, None, None, None)?;
    
    let mut count = 0;
    let mut total_bytes = 0;

    let mut stdout = io::stdout().lock();

    for chunk in stream {
        let chunk = chunk?;
        count += 1;
        total_bytes += chunk.length;
        writeln!(stdout, "{}\t{}\t{}", chunk.hash_hex(), chunk.offset, chunk.length)?;
    }

    let duration = start.elapsed();
    let mb_per_sec = (total_bytes as f64 / 1_000_000.0) / duration.as_secs_f64();

    eprintln!("Chunked {total_bytes} bytes into {count} chunks in {duration:.2?} ({mb_per_sec:.2} MB/s)");

    Ok(())
}
