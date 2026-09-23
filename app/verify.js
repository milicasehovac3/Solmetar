// =====================================================================================
// verify.js - PROVJERA JAVASCRIPT IMPLEMENTACIJE PREMA PYTHONU
//
// Za Limburg (svih 10.752 testnih sati) i po jednu sedmicu za svaki grad u BiH računa
// procjene sva tri modela istim kodom koji koristi aplikacija (core.js) i poredi ih sa
// procjenama Python/PyTorch implementacije (results/python_reference.json).
// Ispisuje najveće apsolutno odstupanje za svaki model; očekuje se oko 1e-7.
//
// Pokretanje (potreban Node.js):   node app/verify.js   (iz korijenskog foldera projekta)
// Rezultat se upisuje i u results/js_verification.json.
// =====================================================================================
const zlib = require("zlib"), fs = require("fs"), path = require("path");
// putanje u odnosu na ovu datoteku, pa se može pokrenuti iz bilo kojeg foldera
const RES = path.join(__dirname, "..", "results");
const SOLAR = require("./core.js");
const M = JSON.parse(fs.readFileSync(path.join(RES, "app_model.json"), "utf8"));
const R = JSON.parse(fs.readFileSync(path.join(RES, "python_reference.json"), "utf8"));
const E = SOLAR.buildEngine(M), HOUR = 3600000;
const T0 = Date.parse(M.t0.replace(" ", "T") + ":00Z");
const t0 = Date.now(), out = {};

for (const name of Object.keys(R)) {
  // raspakuj ugrađene vremenske podatke lokacije (isti postupak kao u pregledaču)
  const L = M.locations[name];
  const buf = zlib.gunzipSync(Buffer.from(L.meteo, "base64"));
  const ints = new Int32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const f = SOLAR.features(SOLAR.dequantize(ints, M.qvars, M.qscale, M.n_hours), T0, L.lat, L.lon);

  // prvi sat perioda za poređenje
  const ref = R[name], k0 = Math.round((Date.parse(ref.t0.replace(" ", "T") + "Z") - T0) / HOUR);
  const r = { n: ref.n };

  // najveće odstupanje procjene za svaki model
  for (const w of ["ridge", "mlp", "lstm"]) {
    let mx = 0;
    for (let i = 0; i < ref.n; i++) mx = Math.max(mx, Math.abs((E.predictAt(f, k0 + i, w) ?? 0) - ref[w][i]));
    r[w] = mx;
  }
  // provjera i dvije geometrijske osobine (cos zenitnog ugla i indeks prozračnosti)
  let gz = 0, gk = 0;
  for (let i = 0; i < ref.n; i++) {
    gz = Math.max(gz, Math.abs(f.cos_z[k0 + i] - ref.cos_z[i]));
    gk = Math.max(gk, Math.abs(f.k_t[k0 + i] - ref.k_t[i]));
  }
  r.cos_z = gz; r.k_t = gk;
  out[name] = r;
}
out.seconds = (Date.now() - t0) / 1000;
console.log(JSON.stringify(out, null, 1));
fs.writeFileSync(path.join(RES, "js_verification.json"), JSON.stringify(out, null, 1));
