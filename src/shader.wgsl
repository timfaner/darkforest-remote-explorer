struct Params {
    base_x: i32,
    base_y: i32,
    side:   i32,
    rounds: u32,
    _pad0:  u32, _pad1:  u32, _pad2:  u32,
  };
  
  struct ConstData {
    modulus:   array<u32, 8>,
    r2:        array<u32, 8>,
    key_mont:  array<u32, 8>,   // 仅用于最后一轮 t = k + l
    threshold: array<u32, 8>,
    inv32:     u32,
    _pad3:     u32, _pad4: u32, _pad5: u32,
  };
  
  struct Hit { x: i32, y: i32, hash: array<u32, 8>, };
  
  // 219 轮，每轮 8 limb => 2×vec4
  struct RoundKeys { data: array<vec4<u32>, 438>, };
  struct Counter { value: atomic<u32>, };
  struct Hits    { data: array<Hit>, };
  
  @group(0) @binding(0) var<storage, read>        params: Params;
  @group(0) @binding(1) var<storage, read>        cdata:  ConstData;
  @group(0) @binding(2) var<uniform>              rk_mont: RoundKeys;
  @group(0) @binding(3) var<storage, read_write>  hits_counter: Counter;
  @group(0) @binding(4) var<storage, read_write>  hits: Hits;
  
  // ================== 本地聚合缓冲（方案 A） ==================
  const WG_CAP: u32 = 64u; // 每个 workgroup 聚合的最大命中数
  const WG_HASH_LEN: u32 = 512u; // 64 * 8
  var<workgroup> wg_cnt: atomic<u32>;
  var<workgroup> wg_x: array<i32, WG_CAP>;
  var<workgroup> wg_y: array<i32, WG_CAP>;
  var<workgroup> wg_hash: array<u32, WG_HASH_LEN>;
  
  // ================== 算术工具 ==================
  fn umul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;
  
    var lo = p0 + (p1 << 16u);
    var carry = select(0u, 1u, lo < p0);
    let lo2 = lo + (p2 << 16u);
    carry = carry + select(0u, 1u, lo2 < lo);
    lo = lo2;
  
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + carry;
    return vec2<u32>(hi, lo);
  }
  
  fn add3(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s0 = a + b;
    let c0 = select(0u, 1u, s0 < a);
    let s1 = s0 + c;
    let c1 = select(0u, 1u, s1 < s0);
    return vec2<u32>(c0 + c1, s1);
  }
  
  fn lt256(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    var aa: array<u32, 8>; var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3]; aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3]; bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var i: i32 = 7;
    loop {
      if (aa[u32(i)] != bb[u32(i)]) { return aa[u32(i)] < bb[u32(i)]; }
      i = i - 1;
      if (i < 0) { break; }
    }
    return false;
  }
  
  fn sub_n(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var aa: array<u32, 8>; var bb: array<u32, 8>; var out: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3]; aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3]; bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let tmp = aa[i] - bb[i];
      let br1 = select(0u, 1u, aa[i] < bb[i]);
      let res = tmp - borrow;
      let br2 = select(0u, 1u, tmp < borrow);
      out[i] = res;
      borrow = select(0u, 1u, (br1 + br2) != 0u);
    }
    return out;
  }
  
  fn add_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var aa: array<u32, 8>; var bb: array<u32, 8>; var out: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3]; aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3]; bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let r = add3(aa[i], bb[i], carry);
      carry = r.x; out[i] = r.y;
    }
    if (!lt256(out, cdata.modulus)) { out = sub_n(out, cdata.modulus); }
    return out;
  }
  
  // CIOS Montgomery 乘法（修正 carry_hi 汇总）
  fn mont_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var aa: array<u32, 8>; var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3]; aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3]; bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
  
    var t: array<u32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
  
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      var carry_lo: u32 = 0u; var carry_hi: u32 = 0u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let prod = umul32(aa[i], bb[j]);
        let r0   = add3(t[i+j], prod.y, carry_lo);
        t[i+j]   = r0.y;
        let sum_hi = prod.x + r0.x;
        carry_lo = sum_hi;
        carry_hi = carry_hi + select(0u, 1u, sum_hi < prod.x);
      }
      var idx = i + 8u;
      var s   = t[idx] + carry_lo;
      var c   = select(0u, 1u, s < t[idx]); t[idx] = s;
  
      idx = idx + 1u;
      var s2  = t[idx] + carry_hi;
      var c2  = select(0u, 1u, s2 < t[idx]); t[idx] = s2;
  
      var cprop = c + c2;
      while (cprop != 0u) {
        idx = idx + 1u;
        let s3 = t[idx] + 1u;
        let c3 = select(0u, 1u, s3 == 0u);
        t[idx] = s3;
        cprop  = cprop - 1u + c3;
      }
  
      let mi = t[i] * cdata.inv32;
      carry_lo = 0u; carry_hi = 0u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let prod2 = umul32(mi, cdata.modulus[j]);
        let r1    = add3(t[i+j], prod2.y, carry_lo);
        t[i+j]    = r1.y;
        let sum_hi2 = prod2.x + r1.x;
        carry_lo  = sum_hi2;
        carry_hi  = carry_hi + select(0u, 1u, sum_hi2 < prod2.x);
      }
      idx = i + 8u;
      s   = t[idx] + carry_lo;
      c   = select(0u, 1u, s < t[idx]); t[idx] = s;
  
      idx = idx + 1u;
      s2  = t[idx] + carry_hi;
      c2  = select(0u, 1u, s2 < t[idx]); t[idx] = s2;
  
      cprop = c + c2;
      while (cprop != 0u) {
        idx = idx + 1u;
        let s3 = t[idx] + 1u;
        let c3 = select(0u, 1u, s3 == 0u);
        t[idx] = s3;
        cprop  = cprop - 1u + c3;
      }
    }
  
    var u: array<u32, 8>;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) { u[j] = t[j+8u]; }
    if (!lt256(u, cdata.modulus)) { u = sub_n(u, cdata.modulus); }
    return u;
  }
  
  fn to_mont(a: array<u32, 8>) -> array<u32, 8> { return mont_mul(a, cdata.r2); }
  
  fn from_mont(x: array<u32, 8>) -> array<u32, 8> {
    var one: array<u32, 8>; for (var i: u32 = 0u; i < 8u; i = i + 1u) { one[i] = 0u; } one[0] = 1u;
    return mont_mul(x, one);
  }
  
  fn i64_to_mont(x: i32) -> array<u32, 8> {
    var v: array<u32, 8>; for (var i: u32 = 0u; i < 8u; i = i + 1u) { v[i] = 0u; }
    if (x >= 0) { v[0] = u32(x); }
    else {
      var tmp: array<u32, 8>; for (var i: u32 = 0u; i < 8u; i = i + 1u) { tmp[i] = 0u; }
      tmp[0] = u32(-x); v = sub_n(cdata.modulus, tmp);
    }
    return to_mont(v);
  }
  
  fn pow5(x: array<u32, 8>) -> array<u32, 8> {
    let x2 = mont_mul(x, x); let x4 = mont_mul(x2, x2); return mont_mul(x4, x);
  }
  
  fn load_rk(i: u32) -> array<u32, 8> {
    var out: array<u32, 8>;
    let base = i * 2u;
    let v0 = rk_mont.data[base + 0u];
    let v1 = rk_mont.data[base + 1u];
    out[0]=v0.x; out[1]=v0.y; out[2]=v0.z; out[3]=v0.w;
    out[4]=v1.x; out[5]=v1.y; out[6]=v1.z; out[7]=v1.w;
    return out;
  }
  
  // ================== Kernel ==================
  @compute @workgroup_size(128)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32
  ) {
    if (li == 0u) { atomicStore(&wg_cnt, 0u); }
    workgroupBarrier();
  
    let idx = i32(gid.x);
    let n   = params.side * params.side;
    if (idx >= n) { return; }
  
    let xi = params.base_x + (idx % params.side);
    let yi = params.base_y + (idx / params.side);
  
    var l: array<u32, 8>; var r: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { l[i] = 0u; r[i] = 0u; }
  
    // 注入 x
    l = add_mod(l, i64_to_mont(xi));
    for (var i: u32 = 0u; i < params.rounds - 1u; i = i + 1u) {
      let t = add_mod(l, load_rk(i));       // (k + C[i]) 已在 host 侧合并
      let ln = add_mod(pow5(t), r);
      r = l; l = ln;
    }
    var t = add_mod(cdata.key_mont, l);     // 最后一轮用 k + l
    r = add_mod(pow5(t), r);
  
    // 注入 y
    l = add_mod(l, i64_to_mont(yi));
    for (var i: u32 = 0u; i < params.rounds - 1u; i = i + 1u) {
      let t2 = add_mod(l, load_rk(i));
      let ln2 = add_mod(pow5(t2), r);
      r = l; l = ln2;
    }
    var t3 = add_mod(cdata.key_mont, l);
    r = add_mod(pow5(t3), r);
  
    let tmp = from_mont(l);

    // 重要：把临时数组拷进本地 var，避免对 let 数组做动态索引
    var h_std: array<u32, 8>;
    h_std[0]=tmp[0]; h_std[1]=tmp[1]; h_std[2]=tmp[2]; h_std[3]=tmp[3];
    h_std[4]=tmp[4]; h_std[5]=tmp[5]; h_std[6]=tmp[6]; h_std[7]=tmp[7];
    
    if (lt256(h_std, cdata.threshold)) {
      let slot = atomicAdd(&wg_cnt, 1u);
      if (slot < WG_CAP) {
        wg_x[slot] = xi;
        wg_y[slot] = yi;
        let base = slot * 8u;
        // 展开 8 次写，避免 h_std[j]
        wg_hash[base + 0u] = h_std[0];
        wg_hash[base + 1u] = h_std[1];
        wg_hash[base + 2u] = h_std[2];
        wg_hash[base + 3u] = h_std[3];
        wg_hash[base + 4u] = h_std[4];
        wg_hash[base + 5u] = h_std[5];
        wg_hash[base + 6u] = h_std[6];
        wg_hash[base + 7u] = h_std[7];
      } else {
        // 溢出直写全局：也展开 8 次
        let w = atomicAdd(&hits_counter.value, 1u);
        hits.data[w].x = xi;
        hits.data[w].y = yi;
        hits.data[w].hash[0] = h_std[0];
        hits.data[w].hash[1] = h_std[1];
        hits.data[w].hash[2] = h_std[2];
        hits.data[w].hash[3] = h_std[3];
        hits.data[w].hash[4] = h_std[4];
        hits.data[w].hash[5] = h_std[5];
        hits.data[w].hash[6] = h_std[6];
        hits.data[w].hash[7] = h_std[7];
      }
    }
  
    workgroupBarrier();
  
    // Flush：仅 Li=0 线程批量落盘
    if (li == 0u) {
      var count = atomicLoad(&wg_cnt);
      if (count > WG_CAP) { count = WG_CAP; }
      if (count > 0u) {
        let base = atomicAdd(&hits_counter.value, count);
        for (var s: u32 = 0u; s < count; s = s + 1u) {
          let dst = base + s;
          hits.data[dst].x = wg_x[s];
          hits.data[dst].y = wg_y[s];
          let off = s * 8u;
          for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            hits.data[dst].hash[j] = wg_hash[off + j];
          }
        }
      }
    }
  }
  