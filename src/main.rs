#[macro_use]
extern crate rocket;

use itertools::iproduct;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rocket::http::Method;
use rocket::serde::json::Json;
use rocket_cors::{catch_all_options_routes, AllowedHeaders, AllowedOrigins};
use serde::{Deserialize, Serialize};
use std::env;

use ark_bn254::Fr;
use ark_ff::{biginteger::BigInteger256 as BI256, BigInteger as _, PrimeField, FpParameters};
use arkworks_mimc::{
    params::{
        mimc_5_220_bn254::{MIMC_5_220_BN254_PARAMS, MIMC_5_220_BN254_ROUND_KEYS},
        round_keys_contants_to_vec,
    },
    MiMC,
};

const K_U64: u64 = 7;

// --- MiMC 实例（固定 x^5/220 轮 / BN254） ---
static MIMC_FEISTEL: Lazy<MiMC<Fr, MIMC_5_220_BN254_PARAMS>> = Lazy::new(|| {
    MiMC::<Fr, MIMC_5_220_BN254_PARAMS>::new(
        /*num_outputs*/ 1,
        Fr::from(K_U64),
        round_keys_contants_to_vec::<Fr>(&MIMC_5_220_BN254_ROUND_KEYS),
    )
});

// --- 工具：把 i64 映射到 Fr（负数按模 p 取代表：-a == p-a） ---
#[inline]
fn i64_to_fr(x: i64) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        -Fr::from((-x) as u64)
    }
}

// --- 工具：把 BigUint 转为 BI256（小端装 4 个 u64 limb） ---
fn biguint_to_bi256(n: &num_bigint::BigUint) -> BI256 {
    let mut le = n.to_bytes_le();
    le.resize(32, 0);
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        let start = i * 8;
        limbs[i] = u64::from_le_bytes(le[start..start + 8].try_into().unwrap());
    }
    BI256(limbs)
}

// --- 计算阈值：floor(p / rarity)（用于与哈希的数值比较） ---
fn threshold_bi256(rarity: u32) -> BI256 {
    // BN254 标量域模数 r
    // 注意：ark-bn254 0.3 的模数常量在 FrParameters::MODULUS
    use ark_bn254::FrParameters;
    let p_big = num_bigint::BigUint::from_bytes_be(&FrParameters::MODULUS.to_bytes_be());
    let t = p_big / num_bigint::BigUint::from(rarity);
    biguint_to_bi256(&t)
}

// --- 输出：把 Fr 转十进制字符串，便于前端/调试 ---
fn fr_to_dec_string(x: &Fr) -> String {
    let bytes = x.into_repr().to_bytes_be();
    num_bigint::BigUint::from_bytes_be(&bytes).to_string()
}

// ------------------------- API types -------------------------

#[derive(Serialize, Deserialize, Clone, Copy)]
struct Coords {
    x: i64,
    y: i64,
}

#[derive(Serialize, Deserialize)]
struct Planet {
    coords: Coords,
    hash: String,
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

#[allow(non_snake_case)]
#[derive(Serialize)]
struct Response {
    chunkFootprint: ChunkFootprint,
    planetLocations: Vec<Planet>,
}

// ------------------------- 核心: /mine -------------------------

#[post("/mine", data = "<task>")]
async fn mine(task: Json<Task>) -> Json<Response> {
    let x0 = task.chunkFootprint.bottomLeft.x;
    let y0 = task.chunkFootprint.bottomLeft.y;
    let n = task.chunkFootprint.sideLength;

    let threshold = threshold_bi256(task.planetRarity);

    let planets = iproduct!(x0..(x0 + n), y0..(y0 + n))
        .par_bridge()
        .filter_map(|(xi, yi)| {
            // MiMC sponge(Feistel variant)：把 (x, y) 两个域元素作为一次吸入
            let h = MIMC_FEISTEL.permute_feistel(vec![i64_to_fr(xi), i64_to_fr(yi)])[0];

            // 与 floor(p/rarity) 比较（按数值而非域元素做比较）
            let h_bi = h.into_repr();
            if h_bi < threshold {
                Some(Planet {
                    coords: Coords { x: xi, y: yi },
                    hash: fr_to_dec_string(&h),
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    Json(Response {
        chunkFootprint: task.chunkFootprint.clone(),
        planetLocations: planets,
    })
}

// ------------------------- Rocket 启动 -------------------------

#[launch]
fn rocket() -> _ {
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

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

    let config = rocket::config::Config {
        port,
        address: std::net::Ipv4Addr::new(0, 0, 0, 0).into(),
        ..rocket::config::Config::default()
    };

    rocket::build()
        .configure(config)
        .mount("/", routes![mine])
        .mount("/", options_routes)
        .manage(cors.clone())
        .attach(cors)
}
