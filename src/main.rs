use std::{
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use image::DynamicImage;
#[cfg(feature = "raw-viewer")]
use minifb::{Key, Window, WindowOptions};
use ultralytics_inference::{
    Source, SourceIterator, SourceMeta, YOLOModel, annotate::annotate_image, batch::BatchProcessor,
};

const RERUN_LOG_INTERVAL: Duration = Duration::from_millis(100);
const DEFAULT_MODEL_PATH: &str = match option_env!("YOLO_DEFAULT_MODEL") {
    Some(path) => path,
    None => "yolo26n.onnx",
};

#[derive(Parser)]
#[command(
    name = "yolo-rerun-demo",
    about = "Run YOLO webcam inference and view annotated frames in Rerun or a local window."
)]
struct Args {
    /// Path to the YOLO ONNX model file.
    #[arg(short, long, value_name = "PATH", default_value = DEFAULT_MODEL_PATH)]
    model: String,

    /// Webcam device index (for example, 0 for /dev/video0).
    #[arg(short, long, value_name = "INDEX", default_value_t = 0)]
    camera: u32,

    /// Show the annotated frames in a local window instead of launching Rerun.
    #[arg(long, alias = "raw-viewer")]
    viewer: bool,
}

struct CapturedFrame {
    image: DynamicImage,
    meta: SourceMeta,
}

#[derive(Default)]
struct LatestFrameState {
    frame: Option<CapturedFrame>,
    error: Option<String>,
    done: bool,
}

type LatestFrames = Arc<(Mutex<LatestFrameState>, Condvar)>;

fn publish_frame(latest: &LatestFrames, frame: CapturedFrame) {
    let (state, ready) = &**latest;
    if let Ok(mut state) = state.lock() {
        // Replacing this slot drops any frame that inference has not reached yet.
        state.frame = Some(frame);
        ready.notify_one();
    }
}

fn publish_error(latest: &LatestFrames, error: impl ToString) {
    let (state, ready) = &**latest;
    if let Ok(mut state) = state.lock() {
        state.error = Some(error.to_string());
        state.done = true;
        ready.notify_one();
    }
}

fn finish_capture(latest: &LatestFrames) {
    let (state, ready) = &**latest;
    if let Ok(mut state) = state.lock() {
        state.done = true;
        ready.notify_one();
    }
}

fn take_latest_frame(
    latest: &LatestFrames,
) -> Result<Option<CapturedFrame>, Box<dyn std::error::Error>> {
    let (state, ready) = &**latest;
    let mut state = state
        .lock()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    while state.frame.is_none() && state.error.is_none() && !state.done {
        state = ready
            .wait(state)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
    }
    if let Some(error) = state.error.take() {
        return Err(std::io::Error::other(error).into());
    }
    Ok(state.frame.take())
}

fn start_capture(webcam_index: u32) -> Result<LatestFrames, Box<dyn std::error::Error>> {
    let latest = Arc::new((Mutex::new(LatestFrameState::default()), Condvar::new()));
    let capture_latest = Arc::clone(&latest);
    thread::spawn(move || {
        let source = match SourceIterator::new(Source::Webcam(webcam_index)) {
            Ok(source) => source,
            Err(error) => {
                publish_error(&capture_latest, error);
                return;
            }
        };
        for frame in source {
            match frame {
                Ok((image, meta)) => publish_frame(&capture_latest, CapturedFrame { image, meta }),
                Err(error) => {
                    publish_error(&capture_latest, error);
                    return;
                }
            }
        }
        finish_capture(&capture_latest);
    });
    Ok(latest)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.viewer {
        #[cfg(feature = "raw-viewer")]
        return run_annotated_viewer(args.model, args.camera);

        #[cfg(not(feature = "raw-viewer"))]
        return Err("Window viewer support is disabled; rebuild with --features raw-viewer".into());
    }

    let mut model = YOLOModel::load(args.model)?;
    let rec = rerun::RecordingStreamBuilder::new("ultralytics_webcam")
        .batcher_config(rerun::log::ChunkBatcherConfig::LOW_LATENCY)
        .spawn()?;
    let latest_frames = start_capture(args.camera)?;
    let started = Instant::now();
    let mut last_rerun_log = Instant::now();

    let mut processor =
        BatchProcessor::new(&mut model, 1, move |results, images, _paths, metas| {
            for ((image_results, image), meta) in results.iter().zip(images).zip(metas) {
                for result in image_results {
                    eprintln!(
                        "frame {} at {:.2}s: {}; inference {:.1} ms",
                        meta.frame_idx,
                        started.elapsed().as_secs_f64(),
                        result.detection_summary(),
                        result.speed.inference.unwrap_or_default(),
                    );

                    if last_rerun_log.elapsed() < RERUN_LOG_INTERVAL {
                        continue;
                    }
                    let annotation_started = Instant::now();
                    let annotated = annotate_image(image, result, None).to_rgb8();
                    let annotation_time = annotation_started.elapsed();
                    let (width, height) = annotated.dimensions();
                    let logged_image =
                        rerun::Image::from_rgb24(annotated.into_raw(), [width, height]);
                    rec.set_time_sequence("frame", meta.frame_idx as i64);
                    let log_started = Instant::now();
                    if let Err(err) = rec.log("webcam/annotated", &logged_image) {
                        eprintln!("Failed to log frame to Rerun: {err}");
                    }
                    eprintln!(
                        "frame {}: annotation {:.1} ms, Rerun log call {:.1} ms",
                        meta.frame_idx,
                        annotation_time.as_secs_f64() * 1_000.0,
                        log_started.elapsed().as_secs_f64() * 1_000.0,
                    );
                    last_rerun_log = Instant::now();
                }
            }
        });

    while let Some(frame) = take_latest_frame(&latest_frames)? {
        let meta = frame.meta;
        processor.add(frame.image, meta.path.clone(), meta);
    }
    Ok(())
}

#[cfg(feature = "raw-viewer")]
fn display_annotated_frame(
    window: &mut Option<Window>,
    image: &DynamicImage,
) -> Result<(), Box<dyn std::error::Error>> {
    let image = image.to_rgb8();
    let (width, height) = image.dimensions();
    let (width, height) = (width as usize, height as usize);
    if window.is_none() {
        *window = Some(Window::new(
            "Annotated webcam",
            width,
            height,
            WindowOptions::default(),
        )?);
    }
    let pixels: Vec<u32> = image
        .pixels()
        .map(|pixel| (u32::from(pixel[0]) << 16) | (u32::from(pixel[1]) << 8) | u32::from(pixel[2]))
        .collect();
    window
        .as_mut()
        .expect("window was initialized")
        .update_with_buffer(&pixels, width, height)?;
    Ok(())
}

#[cfg(feature = "raw-viewer")]
fn infer_and_display(
    model: &mut YOLOModel,
    window: &mut Option<Window>,
    image: &DynamicImage,
    frame_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let results = model.predict_image(image, format!("Webcam frame {frame_idx}"))?;
    if results.is_empty() {
        display_annotated_frame(window, image)?;
    }
    for result in results {
        eprintln!("frame {frame_idx}: {}", result.detection_summary());
        display_annotated_frame(window, &annotate_image(image, &result, None))?;
    }
    Ok(())
}

#[cfg(feature = "raw-viewer")]
#[cfg(feature = "raw-viewer")]
fn run_annotated_viewer(
    model_path: String,
    webcam_index: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = YOLOModel::load(model_path)?;
    let mut window: Option<Window> = None;
    let latest_frames = start_capture(webcam_index)?;
    while let Some(frame) = take_latest_frame(&latest_frames)? {
        infer_and_display(&mut model, &mut window, &frame.image, frame.meta.frame_idx)?;
        if !window
            .as_ref()
            .is_some_and(|window| window.is_open() && !window.is_key_down(Key::Escape))
        {
            break;
        }
    }
    Ok(())
}
