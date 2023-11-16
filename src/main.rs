#![feature(portable_simd)]
use std::simd::{f64x1, u32x1, SimdFloat};

use rayon::prelude::*;
use std::io::Write;

const TITLE: &str = "Mandelbrot Set";

const QUADRANTS: [&str; 4] = ["▖", "▘", "▝", "▗"];
const TWO_QUADRANTS: [&str; 6] = ["▚", "▞", "▄", "▀", "▌", "▐"];
const THREE_QUADRANTS: [&str; 4] = ["▙", "▟", "▛", "▜"];
const FULL_BLOCK: [&str; 2] = ["█", " "];

const FRACTALS: [fn(f64x1, f64x1, u32x1) -> u32x1; 3] = [
    |scaled_x: f64x1, scaled_y: f64x1, max_iterations: u32x1| {
        // Mandelbrot Set

        let mut x = f64x1::splat(0.0);
        let mut y = f64x1::splat(0.0);
        let mut iteration = u32x1::splat(0);

        while x * x + y * y <= f64x1::splat(4.0) && iteration < max_iterations {
            let x_temp = x * x - y * y + scaled_x;
            y = f64x1::splat(2.0) * x * y + scaled_y;
            x = x_temp;
            iteration += u32x1::splat(1);
        }

        iteration
    },
    |scaled_x: f64x1, scaled_y: f64x1, max_iterations: u32x1| {
        // Sinking Ship

        let mut zx = scaled_x;
        let mut zy = scaled_y;
        let mut iteration = u32x1::splat(0);

        while zx * zx + zy * zy <= f64x1::splat(4.0) && iteration < max_iterations {
            let zx_temp = zx * zx - zy * zy + scaled_x;
            zy = (f64x1::splat(2.0) * zx * zy).abs() + scaled_y;
            zx = zx_temp;
            iteration += u32x1::splat(1);
        }

        iteration
    },
    |scaled_x: f64x1, scaled_y: f64x1, max_iterations: u32x1| {
        // Julia Set

        let escape_radius = f64x1::splat(2.0);

        let mut zx = scaled_x;
        let mut zy = scaled_y;
        let mut iteration = u32x1::splat(0);

        while zx * zx + zy * zy <= escape_radius * escape_radius && iteration < max_iterations {
            let zx_temp = zx * zx - zy * zy;
            zy = f64x1::splat(2.0) * zx * zy + f64x1::splat(0.8);
            zx = zx_temp + f64x1::splat(0.156);
            iteration += u32x1::splat(1);
        }

        iteration
    }
];

#[derive(Copy, Clone, PartialEq)]
struct Position {
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
}

impl Position {
    fn width(&self) -> f64 {
        self.right - self.left
    }

    fn height(&self) -> f64 {
        self.bottom - self.top
    }

    fn center(&self) -> (f64, f64) {
        (
            (self.left + self.right) / 2.0,
            (self.top + self.bottom) / 2.0,
        )
    }
}

fn scale_number(
    number: f64x1,
    in_min: f64x1,
    in_max: f64x1,
    out_min: f64x1,
    out_max: f64x1,
) -> f64x1 {
    (number - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}

fn get_pixel(blocks: [[bool; 2]; 2]) -> char {
    match blocks {
        [[true, true], [true, true]] => FULL_BLOCK[0].chars().next().unwrap(),
        [[false, false], [false, false]] => FULL_BLOCK[1].chars().next().unwrap(),
        [[false, true], [true, true]] => THREE_QUADRANTS[1].chars().next().unwrap(),
        [[true, false], [true, true]] => THREE_QUADRANTS[0].chars().next().unwrap(),
        [[true, true], [false, true]] => THREE_QUADRANTS[3].chars().next().unwrap(),
        [[true, true], [true, false]] => THREE_QUADRANTS[2].chars().next().unwrap(),
        [[false, false], [true, true]] => TWO_QUADRANTS[2].chars().next().unwrap(),
        [[true, false], [false, true]] => TWO_QUADRANTS[0].chars().next().unwrap(),
        [[true, true], [false, false]] => TWO_QUADRANTS[3].chars().next().unwrap(),
        [[false, true], [true, false]] => TWO_QUADRANTS[1].chars().next().unwrap(),
        [[false, true], [false, true]] => TWO_QUADRANTS[4].chars().next().unwrap(),
        [[true, false], [true, false]] => TWO_QUADRANTS[5].chars().next().unwrap(),
        [[false, false], [false, true]] => QUADRANTS[3].chars().next().unwrap(),
        [[false, true], [false, false]] => QUADRANTS[2].chars().next().unwrap(),
        [[true, false], [false, false]] => QUADRANTS[1].chars().next().unwrap(),
        [[false, false], [true, false]] => QUADRANTS[0].chars().next().unwrap(),
    }
}

struct Pixel {
    character: char,
    foreground_color: crossterm::style::Color,
    background_color: Option<crossterm::style::Color>,
}

fn hsl_to_rgb(hsl: [f64x1; 3]) -> [f64x1; 3] {
    let s = hsl[1] / f64x1::splat(100.0);
    let l = hsl[2] / f64x1::splat(100.0);
    let k = |n: f64x1| (n + hsl[0] / f64x1::splat(30.0)) % f64x1::splat(12.0);

    let a = s * l.simd_min(f64x1::splat(1.0) - l);
    let f = |n: f64x1| {
        l - a
            * (-f64x1::splat(1.0)).simd_max(
                (k(n) - f64x1::splat(3.0))
                    .simd_min((f64x1::splat(9.0) - k(n)).simd_min(f64x1::splat(1.0))),
            )
    };
    [
        f64x1::splat(255.0) * f(f64x1::splat(0.0)),
        f64x1::splat(255.0) * f(f64x1::splat(8.0)),
        f64x1::splat(255.0) * f(f64x1::splat(4.0)),
    ]
}

fn get_color(iteration: u32x1, max_iterations: u32x1) -> [f64x1; 3] {
    if iteration == max_iterations {
        return [f64x1::splat(0.0); 3];
    } else if iteration[0] == 0 {
        return [f64x1::splat(255.0); 3];
    }

    let h = f64x1::splat(iteration[0] as f64) * f64x1::splat(360.0)
        / f64x1::splat(max_iterations[0] as f64);
    hsl_to_rgb([h, f64x1::splat(100.0), f64x1::splat(50.0)])
}

fn calculate_pixel(
    pixel_x: u16,
    pixel_y: u16,
    width: u16,
    height: u16,
    position: &Position,
    max_iterations: u32x1,
    fractal_index: usize
) -> Pixel {
    let mut subpixel_values = [[u32x1::splat(0); 2]; 2];

    for subpixel_y in 0..2 {
        for subpixel_x in 0..2 {
            let scaled_x = scale_number(
                f64x1::splat((pixel_x * 2 + subpixel_x) as f64),
                f64x1::splat(0.0),
                f64x1::splat(width as f64 * 2.0),
                f64x1::splat(position.left),
                f64x1::splat(position.right),
            );
            let scaled_y = scale_number(
                f64x1::splat((pixel_y * 2 + subpixel_y) as f64),
                f64x1::splat(0.0),
                f64x1::splat(height as f64 * 2.0),
                f64x1::splat(position.top),
                f64x1::splat(position.bottom),
            );

            let iteration = FRACTALS[fractal_index](scaled_x, scaled_y, max_iterations);

            subpixel_values[subpixel_y as usize][subpixel_x as usize] = iteration;
        }
    }

    let subpixels_average = (subpixel_values[0][0]
        + subpixel_values[0][1]
        + subpixel_values[1][0]
        + subpixel_values[1][1])
        / u32x1::splat(4);

    let mut subpixels = [[false; 2]; 2];
    let mut subpixels_on_values = Vec::new();
    let mut subpixels_off_values = Vec::new();

    for subpixel_y in 0..2 {
        for subpixel_x in 0..2 {
            if subpixel_values[subpixel_y as usize][subpixel_x as usize] >= subpixels_average {
                subpixels_on_values.push(subpixel_values[subpixel_y as usize][subpixel_x as usize]);
                subpixels[subpixel_y as usize][subpixel_x as usize] = true;
            } else {
                subpixels_off_values
                    .push(subpixel_values[subpixel_y as usize][subpixel_x as usize]);
            }
        }
    }

    if subpixels_on_values.len() == 4 {
        let foreground_color_rgb = get_color(subpixels_average, max_iterations);

        return Pixel {
            character: get_pixel(subpixels),
            foreground_color: crossterm::style::Color::Rgb {
                r: foreground_color_rgb[0][0] as u8,
                g: foreground_color_rgb[1][0] as u8,
                b: foreground_color_rgb[2][0] as u8,
            },
            background_color: None,
        };
    } else {
        let mut subpixels_on_average = u32x1::splat(0);
        if subpixels_on_values.len() != 0 {
            for subpixel_on_value in &subpixels_on_values {
                subpixels_on_average += subpixel_on_value;
            }
            subpixels_on_average /= u32x1::splat(subpixels_on_values.len() as u32);
        }

        let mut subpixels_off_average = u32x1::splat(0);
        if subpixels_off_values.len() != 0 {
            for subpixel_off_value in &subpixels_off_values {
                subpixels_off_average += subpixel_off_value;
            }
            subpixels_off_average /= u32x1::splat(subpixels_off_values.len() as u32);
        }

        let foreground_color_rgb = get_color(subpixels_on_average, max_iterations);
        let background_color_rgb = get_color(subpixels_off_average, max_iterations);

        let foreground_color = crossterm::style::Color::Rgb {
            r: foreground_color_rgb[0][0] as u8,
            g: foreground_color_rgb[1][0] as u8,
            b: foreground_color_rgb[2][0] as u8,
        };

        let background_color = crossterm::style::Color::Rgb {
            r: background_color_rgb[0][0] as u8,
            g: background_color_rgb[1][0] as u8,
            b: background_color_rgb[2][0] as u8,
        };

        return Pixel {
            character: get_pixel(subpixels),
            foreground_color,
            background_color: Some(background_color),
        };
    }
}

fn render_row(
    pixel_y: u16,
    width: u16,
    height: u16,
    position: &Position,
    max_iterations: u32x1,
    fractal_index: usize
) -> String {
    let mut last_fg_color = crossterm::style::Color::Reset;
    let mut last_bg_color = crossterm::style::Color::Reset;

    let mut output = String::new();

    for pixel_x in 0..width {
        let pixel = calculate_pixel(pixel_x, pixel_y, width, height, position, max_iterations, fractal_index);

        let fg_color = pixel.foreground_color;
        if fg_color != last_fg_color {
            output.push_str(&format!(
                "{}",
                crossterm::style::SetForegroundColor(fg_color)
            ));
            last_fg_color = fg_color;
        }

        if let Some(bg_color) = pixel.background_color {
            if bg_color != last_bg_color {
                output.push_str(&format!(
                    "{}",
                    crossterm::style::SetBackgroundColor(bg_color)
                ));
                last_bg_color = bg_color;
            }
        }

        output.push_str(&format!("{}", pixel.character));
    }

    output
}

fn render_frame(width: u16, height: u16, position: &Position, max_iterations: u32x1, fractal_index: usize) -> String {
    let output = (0..height)
        .into_par_iter()
        .map(|pixel_y| render_row(pixel_y, width, height, position, max_iterations, fractal_index))
        .collect::<Vec<String>>()
        .join("\n");
    format!("{}{}", output, crossterm::style::ResetColor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = std::io::BufWriter::new(std::io::stdout());

    let default_position = Position {
        top: -1.0,
        bottom: 1.0,
        left: -2.0,
        right: 1.0,
    };
    let mut position = default_position.clone();
    let mut max_iterations = u32x1::splat(100);
    let mut fractal_index = 0;
    let mut last_terminal_size = (0, 0);

    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(
        writer,
        crossterm::terminal::SetTitle(TITLE),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::cursor::DisableBlinking,
        crossterm::cursor::Hide,
        crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
        crossterm::cursor::MoveTo(0, 0)
    )?;

    loop {
        let mut should_redraw = false;

        match crossterm::event::read()? {
            crossterm::event::Event::Key(event) => {
                if event.kind != crossterm::event::KeyEventKind::Press {
                    continue;
                }

                match event.code {
                    crossterm::event::KeyCode::Char('q') => break,
                    crossterm::event::KeyCode::Char('w') => {
                        let center = position.center();
                        let height = position.height();
                        let zoom = height / 2.0;

                        position.top = center.1 - zoom * 1.1;
                        position.bottom = center.1 + zoom * 0.9;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('s') => {
                        let center = position.center();
                        let height = position.height();
                        let zoom = height / 2.0;

                        position.top = center.1 - zoom * 0.9;
                        position.bottom = center.1 + zoom * 1.1;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('a') => {
                        let center = position.center();
                        let width = position.width();
                        let zoom = width / 2.0;

                        position.left = center.0 - zoom * 1.1;
                        position.right = center.0 + zoom * 0.9;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('d') => {
                        let center = position.center();
                        let width = position.width();
                        let zoom = width / 2.0;

                        position.left = center.0 - zoom * 0.9;
                        position.right = center.0 + zoom * 1.1;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Up => {
                        let center = position.center();
                        let width = position.width();
                        let height = position.height();

                        position.top = center.1 - height / 2.0 * 0.9;
                        position.bottom = center.1 + height / 2.0 * 0.9;
                        position.left = center.0 - width / 2.0 * 0.9;
                        position.right = center.0 + width / 2.0 * 0.9;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Down => {
                        let center = position.center();
                        let width = position.width();
                        let height = position.height();

                        position.top = center.1 - height / 2.0 * 1.1;
                        position.bottom = center.1 + height / 2.0 * 1.1;
                        position.left = center.0 - width / 2.0 * 1.1;
                        position.right = center.0 + width / 2.0 * 1.1;
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Enter => {
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('=') => {
                        max_iterations += u32x1::splat(10);
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('-') => {
                        if max_iterations > u32x1::splat(10) {
                            max_iterations -= u32x1::splat(10);
                            should_redraw = true;
                        }
                    }
                    crossterm::event::KeyCode::Char('[') => {
                        if fractal_index == 0 {
                            fractal_index = FRACTALS.len() - 1;
                        } else {
                            fractal_index -= 1;
                        }
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char(']') => {
                        if fractal_index == FRACTALS.len() - 1 {
                            fractal_index = 0;
                        } else {
                            fractal_index += 1;
                        }
                        should_redraw = true;
                    }
                    crossterm::event::KeyCode::Char('r') => {
                        if position != default_position {
                            position = default_position;
                            should_redraw = true;
                        }
                    }
                    _ => (),
                }
            }
            crossterm::event::Event::Resize(width, height) => {
                if width != last_terminal_size.0 || height != last_terminal_size.1 {
                    crossterm::execute!(
                        writer,
                        crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
                    )?;
                    should_redraw = true;
                }
            }
            _ => (),
        }

        if should_redraw {
            let terminal_size = crossterm::terminal::size()?;
            let rendered =
                render_frame(terminal_size.0, terminal_size.1, &position, max_iterations, fractal_index);
            crossterm::execute!(writer, crossterm::cursor::MoveTo(0, 0))?;
            writer.write_all(rendered.as_bytes())?;
            writer.flush()?;
            last_terminal_size = terminal_size;
        }
    }

    crossterm::execute!(
        writer,
        crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
        crossterm::cursor::Show,
        crossterm::cursor::EnableBlinking,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::style::ResetColor,
    )?;
    crossterm::terminal::disable_raw_mode()?;

    drop(writer);
    Ok(())
}
