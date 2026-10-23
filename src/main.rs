use std::{
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};

use image::DynamicImage;
#[cfg(feature = "raw-viewer")]
use minifb::{Key, Window, WindowOptions};
use ultralytics_inference::{
    SourceMeta, YOLOModel, annotate::annotate_image, batch::BatchProcessor,
};

const RERUN_LOG_INTERVAL: Duration = Duration::from_millis(100);

#[cfg(not(feature = "nokhwa-source"))]
use ultralytics_inference::{Source, SourceIterator};

#[cfg(feature = "nokhwa-source")]
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution},
};

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

#[cfg(not(feature = "nokhwa-source"))]
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

#[cfg(not(feature = "nokhwa-source"))]
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

#[cfg(feature = "nokhwa-source")]
fn start_capture(webcam_index: u32) -> Result<LatestFrames, Box<dyn std::error::Error>> {
    let latest = Arc::new((Mutex::new(LatestFrameState::default()), Condvar::new()));
    let capture_latest = Arc::clone(&latest);
    thread::spawn(move || {
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
            Resolution::new(640, 480),
        ));
        let mut camera = match Camera::new(CameraIndex::Index(webcam_index), format) {
            Ok(camera) => camera,
            Err(error) => {
                publish_error(&capture_latest, error);
                return;
            }
        };
        if let Err(error) = camera.open_stream() {
            publish_error(&capture_latest, error);
            return;
        }
        let resolution = camera.resolution();
        let fps = camera.frame_rate() as f32;
        eprintln!(
            "Nokhwa webcam {webcam_index}: {}x{} at {} FPS",
            resolution.width(),
            resolution.height(),
            camera.frame_rate()
        );
        let mut frame_idx = 0;
        loop {
            let frame = match camera.frame() {
                Ok(frame) => frame,
                Err(error) => {
                    publish_error(&capture_latest, error);
                    return;
                }
            };
            let rgb = match frame.decode_image::<RgbFormat>() {
                Ok(rgb) => rgb,
                Err(error) => {
                    publish_error(&capture_latest, error);
                    return;
                }
            };
            publish_frame(
                &capture_latest,
                CapturedFrame {
                    image: DynamicImage::ImageRgb8(rgb),
                    meta: SourceMeta {
                        frame_idx,
                        total_frames: None,
                        path: format!("Webcam {webcam_index}"),
                        fps: Some(fps),
                    },
                },
            );
            frame_idx += 1;
        }
    });
    Ok(latest)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first_arg = args.next();
    if matches!(first_arg.as_deref(), Some("--viewer" | "--raw-viewer")) {
        #[cfg(feature = "raw-viewer")]
        return run_annotated_viewer();

        #[cfg(not(feature = "raw-viewer"))]
        return Err("Window viewer support is disabled; rebuild with --features raw-viewer".into());
    }

    let model_path = first_arg.unwrap_or_else(|| "yolo26n.onnx".to_owned());
    let webcam_index = args
        .next()
        .map(|arg| arg.parse::<u32>())
        .transpose()?
        .unwrap_or(0);

    let mut model = YOLOModel::load(model_path)?;
    let rec = rerun::RecordingStreamBuilder::new("ultralytics_webcam")
        .batcher_config(rerun::log::ChunkBatcherConfig::LOW_LATENCY)
        .spawn()?;
    let latest_frames = start_capture(webcam_index)?;
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
fn viewer_model_and_camera() -> Result<(String, u32), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(2);
    let Some(first_arg) = args.next() else {
        return Ok(("yolo26n.onnx".to_owned(), 0));
    };
    if let Ok(webcam_index) = first_arg.parse::<u32>() {
        return Ok(("yolo26n.onnx".to_owned(), webcam_index));
    }
    let webcam_index = args
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(0);
    Ok((first_arg, webcam_index))
}

#[cfg(feature = "raw-viewer")]
fn run_annotated_viewer() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, webcam_index) = viewer_model_and_camera()?;
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
