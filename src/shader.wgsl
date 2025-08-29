struct Params {
    base_x: i32,
    base_y: i32,
    side:   i32,
    rounds: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
  };
  
  struct ConstData {
    modulus:   array<u32, 8>,
    r2:        array<u32, 8>,
    key_mont:  array<u32, 8>,
    threshold: array<u32, 8>,
    inv32:     u32,
    _pad3:     u32,
    _pad4:     u32,
    _pad5:     u32,
  };
  
  struct Hit {
    x: i32,
    y: i32,
    hash: array<u32, 8>,
  };
  
  struct RoundKeys { data: array<array<u32, 8>>, };
  struct Counter   { value: atomic<u32>, };
  struct Hits      { data: array<Hit>, };
  
  @group(0) @binding(0) var<storage, read>        params: Params;
  @group(0) @binding(1) var<storage, read>        cdata:  ConstData;
  @group(0) @binding(2) var<storage, read>        rk_mont: RoundKeys;
  @group(0) @binding(3) var<storage, read_write>  hits_counter: Counter;
  @group(0) @binding(4) var<storage, read_write>  hits: Hits;
  
  fn umul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
  
    let p0 = a0 * b0; // 低 32 位部分
    let p1 = a0 * b1; // 交叉项1（16x16 -> 32）
    let p2 = a1 * b0; // 交叉项2（16x16 -> 32）
    let p3 = a1 * b1; // 高 32 位部分
  
    // 组合低 32 位，并记录向高 32 位的进位
    var lo = p0 + (p1 << 16u);
    var carry = select(0u, 1u, lo < p0);
    let add2 = (p2 << 16u);
    let lo2 = lo + add2;
    carry = carry + select(0u, 1u, lo2 < lo);
    lo = lo2;
  
    // 组合高 32 位：p3 + (p1 >> 16) + (p2 >> 16) + 来自低 32 的进位
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
    var aa: array<u32, 8>;
    var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3];
    aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3];
    bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var i: i32 = 7;
    loop {
      if (aa[u32(i)] != bb[u32(i)]) { return aa[u32(i)] < bb[u32(i)]; }
      i = i - 1;
      if (i < 0) { break; }
    }
    return false;
  }
  
  fn sub_n(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var aa: array<u32, 8>;
    var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3];
    aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3];
    bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var out: array<u32, 8>;
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
    var aa: array<u32, 8>;
    var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3];
    aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3];
    bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
    var out: array<u32, 8>;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let r = add3(aa[i], bb[i], carry);
      carry = r.x;
      out[i] = r.y;
    }
    if (!lt256(out, cdata.modulus)) {
      out = sub_n(out, cdata.modulus);
    }
    return out;
  }
  
  // Montgomery 乘法：基数 2^32，n=8
  fn mont_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // 先拷到本地，允许动态索引
    var aa: array<u32, 8>;
    var bb: array<u32, 8>;
    aa[0]=a[0]; aa[1]=a[1]; aa[2]=a[2]; aa[3]=a[3]; aa[4]=a[4]; aa[5]=a[5]; aa[6]=a[6]; aa[7]=a[7];
    bb[0]=b[0]; bb[1]=b[1]; bb[2]=b[2]; bb[3]=b[3]; bb[4]=b[4]; bb[5]=b[5]; bb[6]=b[6]; bb[7]=b[7];
  
    var t: array<u32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
  
    // 外层 8 轮
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      // -------- 累加 a[i]*b 到 t[i..i+8] --------
      var carry_lo: u32 = 0u;
      var carry_hi: u32 = 0u; // 统计 sum_hi 溢出次数
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let prod = umul32(aa[i], bb[j]);        // (hi, lo)
        let r0   = add3(t[i+j], prod.y, carry_lo);
        t[i+j]   = r0.y;
  
        let sum_hi = prod.x + r0.x;
        carry_lo   = sum_hi;
        carry_hi   = carry_hi + select(0u, 1u, sum_hi < prod.x); // 统计进位
      }
      // 把 64 位 carry 加到 t[i+8], t[i+9]，并向后传播
      var idx = i + 8u;
      var s   = t[idx] + carry_lo;
      var c   = select(0u, 1u, s < t[idx]);
      t[idx]  = s;
  
      idx = idx + 1u;
      var s2  = t[idx] + carry_hi;
      var c2  = select(0u, 1u, s2 < t[idx]);
      t[idx]  = s2;
  
      var cprop = c + c2; // 可能是 0/1/2
      while (cprop != 0u) {
        idx      = idx + 1u;
        let s3   = t[idx] + 1u;
        let c3   = select(0u, 1u, s3 == 0u);
        t[idx]   = s3;
        // 消耗一个待传播进位，再加上这次加 1 是否又溢出
        cprop    = cprop - 1u + c3;
      }
  
      // -------- Montgomery 约减：加 mi * p 到 t[i..i+8] --------
      let mi = t[i] * cdata.inv32; // 低 32 位自动截断
  
      carry_lo = 0u;
      carry_hi = 0u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let prod2 = umul32(mi, cdata.modulus[j]);
        let r1    = add3(t[i+j], prod2.y, carry_lo);
        t[i+j]    = r1.y;
  
        let sum_hi2 = prod2.x + r1.x;
        carry_lo    = sum_hi2;
        carry_hi    = carry_hi + select(0u, 1u, sum_hi2 < prod2.x);
      }
      idx = i + 8u;
      s   = t[idx] + carry_lo;
      c   = select(0u, 1u, s < t[idx]);
      t[idx] = s;
  
      idx = idx + 1u;
      s2  = t[idx] + carry_hi;
      c2  = select(0u, 1u, s2 < t[idx]);
      t[idx] = s2;
  
      cprop = c + c2;
      while (cprop != 0u) {
        idx      = idx + 1u;
        let s3   = t[idx] + 1u;
        let c3   = select(0u, 1u, s3 == 0u);
        t[idx]   = s3;
        cprop    = cprop - 1u + c3;
      }
      // 此时按 CIOS 性质有 t[i] == 0（mod 2^32）
    }
  
    // u = t[8..15]
    var u: array<u32, 8>;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) { u[j] = t[j + 8u]; }
    if (!lt256(u, cdata.modulus)) { u = sub_n(u, cdata.modulus); }
    return u;
  }
  
  fn to_mont(a: array<u32, 8>) -> array<u32, 8> {
    return mont_mul(a, cdata.r2);
  }
  
  fn from_mont(x: array<u32, 8>) -> array<u32, 8> {
    var one: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { one[i] = 0u; }
    one[0] = 1u;
    return mont_mul(x, one);
  }
  
  fn i64_to_mont(x: i32) -> array<u32, 8> {
    var v: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { v[i] = 0u; }
    if (x >= 0) {
      v[0] = u32(x);
    } else {
      var tmp: array<u32, 8>;
      for (var i: u32 = 0u; i < 8u; i = i + 1u) { tmp[i] = 0u; }
      let absx = u32(-x);
      tmp[0] = absx;
      v = sub_n(cdata.modulus, tmp);
    }
    return to_mont(v);
  }
  
  fn pow5(x: array<u32, 8>) -> array<u32, 8> {
    let x2 = mont_mul(x, x);
    let x4 = mont_mul(x2, x2);
    return mont_mul(x4, x);
  }
  
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let n   = params.side * params.side;
    if (idx >= n) { return; }
  
    let xi = params.base_x + (idx % params.side);
    let yi = params.base_y + (idx / params.side);
  
    var l: array<u32, 8>;
    var r: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { l[i] = 0u; r[i] = 0u; }
  
    // 注入 x
    let inx = i64_to_mont(xi);
    l = add_mod(l, inx);
  
    // mix
    for (var i: u32 = 0u; i < params.rounds - 1u; i = i + 1u) {
      var t = add_mod(cdata.key_mont, l);
      t = add_mod(t, rk_mont.data[i]);
      let ln = add_mod(pow5(t), r);
      r = l;
      l = ln;
    }
    var t = add_mod(cdata.key_mont, l);
    r = add_mod(pow5(t), r);
  
    // 注入 y
    let iny = i64_to_mont(yi);
    l = add_mod(l, iny);
  
    for (var i: u32 = 0u; i < params.rounds - 1u; i = i + 1u) {
      var t2 = add_mod(cdata.key_mont, l);
      t2 = add_mod(t2, rk_mont.data[i]);
      let ln2 = add_mod(pow5(t2), r);
      r = l;
      l = ln2;
    }
    var t3 = add_mod(cdata.key_mont, l);
    r = add_mod(pow5(t3), r);
  
    // 输出 l（转标准表示）
    let h_std = from_mont(l);
  
    if (lt256(h_std, cdata.threshold)) {
      let w = atomicAdd(&hits_counter.value, 1u);
      hits.data[w].x = xi;
      hits.data[w].y = yi;
      hits.data[w].hash = h_std;
    }
  }
  