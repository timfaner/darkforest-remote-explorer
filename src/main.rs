#[macro_use]
extern crate rocket;

use rocket::http::Method;
use rocket::serde::json::Json;
use rocket_cors::{catch_all_options_routes, AllowedHeaders, AllowedOrigins};
use serde::{Deserialize, Serialize};
use std::env;

use rayon::prelude::*;

use ark_bn254::Fr;
use ark_ff::{BigInteger, FpParameters, PrimeField,Field};
use arkworks_mimc::{
    params::mimc_5_220_bn254::MIMC_5_220_BN254_ROUND_KEYS, params::round_keys_contants_to_vec,
};

use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};

// ----------------------- API Types -----------------------

#[derive(Serialize, Deserialize, Clone, Copy)]
struct Coords {
    x: i64,
    y: i64,
}

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize, Clone)]
struct ChunkFootprint {
    bottomLeft: Coords,
    sideLength: i64,
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct Task {
    chunkFootprint: ChunkFootprint,
    planetRarity: u32,
}

#[derive(Serialize)]
struct Planet {
    coords: Coords,
    hash: String,
}

#[allow(non_snake_case)]
#[derive(Serialize)]
struct Response {
    chunkFootprint: ChunkFootprint,
    planetLocations: Vec<Planet>,
}

// ----------------------- CPU 实现（arkworks-mimc） -----------------------

#[inline]
fn i64_to_fr(x: i64) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        -Fr::from((-x) as u64)
    }
}

#[inline]
fn pow5_fr(x: Fr) -> Fr {
    let x2 = x.square();
    let x4 = x2.square();
    x4 * x
}

fn mimc_feistel_xy_cpu(xi: i64, yi: i64, k: Fr, rks: &[Fr]) -> Fr {
    // 与 GPU 一致：注入一个元素 -> mix；再注入下一个 -> mix；输出 l
    let rounds = 220usize;
    let mut l = Fr::zero();
    let mut r = Fr::zero();

    // 注入 x
    l += i64_to_fr(xi);
    for i in 0..(rounds - 1) {
        let t = k + l + rks[i];
        let ln = pow5_fr(t) + r;
        r = l;
        l = ln;
    }
    let t = k + l;
    r = pow5_fr(t) + r;

    // 注入 y
    l += i64_to_fr(yi);
    for i in 0..(rounds - 1) {
        let t = k + l + rks[i];
        let ln = pow5_fr(t) + r;
        r = l;
        l = ln;
    }
    let t = k + l;
    let _ = pow5_fr(t) + r; // r 最终值不作为输出
    l
}

fn cpu_mine(base_x: i64, base_y: i64, side: i64, rarity: u32) -> Vec<Planet> {
    use ark_bn254::FrParameters;

    // p 与阈值
    let p_be = FrParameters::MODULUS.to_bytes_be();
    let p_big = BigUint::from_bytes_be(&p_be);
    let threshold = &p_big / BigUint::from(rarity);

    // 轮常数 & key
    let rks = round_keys_contants_to_vec::<Fr>(&MIMC_5_220_BN254_ROUND_KEYS);
    let k = Fr::from(7u64);

    let total = (side * side) as usize;

    (0..total)
        .into_par_iter()
        .filter_map(|idx| {
            let xi = base_x + (idx as i64 % side);
            let yi = base_y + (idx as i64 / side);

            let h = mimc_feistel_xy_cpu(xi, yi, k, &rks);
            // 转 BigUint 比较
            let be = h.into_repr().to_bytes_be();
            let n = BigUint::from_bytes_be(&be);
            if n < threshold {
                Some(Planet {
                    coords: Coords { x: xi, y: yi },
                    hash: n.to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

// ----------------------- GPU 实现（wgpu + WGSL） -----------------------

use futures_intrusive::channel::shared::oneshot_channel;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    base_x: i32,
    base_y: i32,
    side: i32,
    rounds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstDataGpu {
    modulus: [u32; 8],
    r2: [u32; 8],
    key_mont: [u32; 8],
    threshold: [u32; 8],
    inv32: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
}

impl GpuCtx {
    async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mimc-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mimc-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind-layout"),
            entries: &[
                // 0: params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: const data
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: round keys (mont)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: hits counter
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: hits array
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipe-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mimc-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_layout,
        })
    }
}

fn bytes_le_to_u32x8(mut le: Vec<u8>) -> [u32; 8] {
    le.resize(32, 0);
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = u32::from_le_bytes(le[i * 4..i * 4 + 4].try_into().unwrap());
    }
    out
}
fn biguint_to_u32x8(n: &BigUint) -> [u32; 8] {
    bytes_le_to_u32x8(n.to_bytes_le())
}
fn to_mont_u32x8(val_std: &BigUint, r_mod_p: &BigUint, p: &BigUint) -> [u32; 8] {
    biguint_to_u32x8(&((val_std * r_mod_p) % p))
}

fn gpu_mine(
    ctx: &GpuCtx,
    base_x: i64,
    base_y: i64,
    side: i64,
    planet_rarity: u32,
) -> anyhow::Result<Vec<Planet>> {
    use ark_bn254::FrParameters;

    // 域常量
    let p_be = FrParameters::MODULUS.to_bytes_be();
    let p_big = BigUint::from_bytes_be(&p_be);
    let r_be = FrParameters::R.to_bytes_be();
    let r_big = BigUint::from_bytes_be(&r_be);
    let r2_be = FrParameters::R2.to_bytes_be();
    let r2_big = BigUint::from_bytes_be(&r2_be);

    // Montgomery inv32（2^32 基数）
    let inv32: u32 = 0xEFFFFFFF;

    // 阈值
    let threshold = &p_big / BigUint::from(planet_rarity);

    // 轮常数：标准 -> mont
    let rk_fr = round_keys_contants_to_vec::<Fr>(&MIMC_5_220_BN254_ROUND_KEYS);
    let rk_mont_u32x8: Vec<[u32; 8]> = rk_fr
        .iter()
        .map(|c| {
            let c_std = BigUint::from_bytes_be(&c.into_repr().to_bytes_be());
            to_mont_u32x8(&c_std, &r_big, &p_big)
        })
        .collect();

    // key=7（mont）
    let key_mont = to_mont_u32x8(&BigUint::from_u64(7).unwrap(), &r_big, &p_big);

    let params = ParamsGpu {
        base_x: base_x as i32,
        base_y: base_y as i32,
        side: side as i32,
        rounds: 220,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let cdata = ConstDataGpu {
        modulus: biguint_to_u32x8(&p_big),
        r2: biguint_to_u32x8(&r2_big),
        key_mont,
        threshold: biguint_to_u32x8(&threshold),
        inv32,
        _pad3: 0,
        _pad4: 0,
        _pad5: 0,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cdata_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdata"),
        contents: bytemuck::bytes_of(&cdata),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // round keys 展平
    let mut rk_flat = Vec::<u32>::with_capacity(8 * rk_mont_u32x8.len());
    for w in &rk_mont_u32x8 {
        rk_flat.extend_from_slice(w);
    }
    let rk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rk"),
        contents: bytemuck::cast_slice::<u32, u8>(&rk_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 输出缓冲
    let total = (side * side) as usize;
    let hit_size_bytes = std::mem::size_of::<i32>() * 2 + 32; // x,y + 8*4 limbs
    let hits_buf_size = (total * hit_size_bytes) as u64;

    let hits_counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hits-counter"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let hits_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits"),
        size: hits_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 绑定
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind"),
        layout: &ctx.bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cdata_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hits_counter_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: hits_buf.as_entire_binding(),
            },
        ],
    });

    // 调度
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        let work = ((total as u32) + 255) / 256;
        cpass.dispatch_workgroups(work, 1, 1);
    }

    // 读回
    let counter_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counter-read"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&hits_counter_buf, 0, &counter_read, 0, 4);

    let hits_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits-read"),
        size: hits_buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&hits_buf, 0, &hits_read, 0, hits_buf_size);

    queue.submit(Some(encoder.finish()));

    // 等待映射（wgpu 0.19: 回调 + poll）
    {
        let slice = counter_read.slice(..);
        let (s, r) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        device.poll(wgpu::Maintain::Wait);

        pollster::block_on(r.receive()).unwrap().unwrap();
    }
    let hit_count: u32 = {
        let data = counter_read.slice(..).get_mapped_range();
        let arr = bytemuck::from_bytes::<u32>(&data);
        *arr
    };
    counter_read.unmap();

    {
        let slice = hits_read.slice(..);
        let (s, r) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(r.receive()).unwrap().unwrap();
    }
    let data = hits_read.slice(..).get_mapped_range().to_vec();
    hits_read.unmap();

    // 解析前 hit_count 条
    let mut out = Vec::with_capacity(hit_count as usize);
    for i in 0..(hit_count as usize) {
        let offset = i * hit_size_bytes;
        let x = i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as i64;
        let y = i32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as i64;
        let mut limbs = [0u32; 8];
        for j in 0..8 {
            let s = offset + 8 + j * 4;
            limbs[j] = u32::from_le_bytes(data[s..s + 4].try_into().unwrap());
        }
        let mut bytes = Vec::with_capacity(32);
        for j in 0..8 {
            bytes.extend_from_slice(&limbs[j].to_le_bytes());
        }
        let n = BigUint::from_bytes_le(&bytes);
        out.push(Planet {
            coords: Coords { x, y },
            hash: n.to_string(),
        });
    }

    Ok(out)
}

// ----------------------- Rocket 集成 -----------------------

struct AppState {
    gpu: Option<GpuCtx>,
}

#[post("/mine?<engine>", data = "<task>")]
async fn mine(engine: Option<String>, task: Json<Task>, state: &rocket::State<AppState>) -> Json<Response> {
    let x0 = task.chunkFootprint.bottomLeft.x;
    let y0 = task.chunkFootprint.bottomLeft.y;
    let side = task.chunkFootprint.sideLength;
    let rarity = task.planetRarity;

    let engine = engine.unwrap_or_else(|| "auto".to_string());

    let planets = match engine.as_str() {
        "cpu" => cpu_mine(x0, y0, side, rarity),
        "gpu" => match &state.gpu {
            Some(gpu) => gpu_mine(gpu, x0, y0, side, rarity).unwrap_or_else(|_| cpu_mine(x0, y0, side, rarity)),
            None => cpu_mine(x0, y0, side, rarity),
        },
        _ => {
            // auto
            if let Some(gpu) = &state.gpu {
                gpu_mine(gpu, x0, y0, side, rarity).unwrap_or_else(|_| cpu_mine(x0, y0, side, rarity))
            } else {
                cpu_mine(x0, y0, side, rarity)
            }
        }
    };

    Json(Response {
        chunkFootprint: task.chunkFootprint.clone(),
        planetLocations: planets,
    })
}

#[launch]
async fn rocket() -> _ {
    // 尝试初始化 GPU；失败则记录并走 CPU
    let gpu = match GpuCtx::new().await {
        Ok(ctx) => {
            eprintln!("[init] GPU ready (wgpu)");
            Some(ctx)
        }
        Err(e) => {
            eprintln!("[init] GPU unavailable, fallback to CPU: {e}");
            None
        }
    };

    let allowed_origins = AllowedOrigins::all();
    let cors = rocket_cors::CorsOptions {
        allowed_origins,
        allowed_methods: vec![Method::Post].into_iter().map(From::from).collect(),
        allowed_headers: AllowedHeaders::all(),
        allow_credentials: true,
        ..Default::default()
    }
    .to_cors()
    .unwrap();
    let options_routes = catch_all_options_routes();

    let port: u16 = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8000);

    let config = rocket::config::Config {
        port,
        address: std::net::Ipv4Addr::new(0, 0, 0, 0).into(),
        ..rocket::config::Config::default()
    };

    rocket::build()
        .configure(config)
        .manage(AppState { gpu })
        .mount("/", routes![mine])
        .mount("/", options_routes)
        .manage(cors.clone()) 
        .attach(cors)
}
