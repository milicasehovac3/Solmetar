// =====================================================================================
// app.js - KORISNIČKI INTERFEJS APLIKACIJE SOLMETAR
//
// Ova datoteka povezuje ekran (HTML elemente iz body_main.html) sa računskim jezgrom
// (core.js). Redoslijed rada:
//   1. učita ugrađene podatke za izabranu lokaciju (ili ih preuzme uživo sa Open-Meteo);
//   2. za svaki od 24 sata izabranog dana pozove SOLAR predictAt za svaki model;
//   3. procjene pomnoži sa instaliranom snagom i nacrta kartice, grafikon i tabelu.
//
// Globalni objekti koje ubacuje build_app.py prije ove datoteke:
//   MODEL - sadržaj results/app_model.json (težine modela, podaci, metrike);
//   REF   - sadržaj results/python_reference.json (Python procjene za provjeru);
//   SOLAR - računsko jezgro iz core.js.
//
// Oznake: kappa = P / P_nom (udio instalirane snage), k = redni broj sata od M.t0.
// =====================================================================================
"use strict";

// ---------- osnovne konstante ----------
const M = MODEL;                          // kraće ime za model i podatke
const E = SOLAR.buildEngine(M);           // "motor" sa funkcijom predictAt za sva tri modela
const HOUR = 3600000;                     // jedan sat u ms
const N = M.n_hours;                      // broj ugrađenih sati po lokaciji (od 1.1.2021.)
const T0 = Date.parse(M.t0.replace(" ", "T") + ":00Z");   // vrijeme prvog ugrađenog sata (UTC)
const LOCS = Object.keys(M.locations);    // Limburg i sedam gradova u BiH
const TZ = "Europe/Sarajevo";             // prikaz po lokalnom vremenu (CET/CEST, isto i u Belgiji)
// Za Limburg se, kao pri treniranju, uprosječuju četiri tačke u provinciji.
const LIMBURG_PTS = { lat: [51.15, 51.10, 50.88, 50.93], lon: [5.20, 5.65, 5.30, 5.68] };
// Imena veličina na Open-Meteo, istim redoslijedom kao M.qvars (G, G_b, G_d, G_n, T_a, RH, v_w, R, S, C).
const OM_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m",
  "relative_humidity_2m", "wind_speed_10m", "precipitation", "snowfall", "cloud_cover"];

// Skraćenica za dohvatanje HTML elementa po id-u: $("loc") umjesto document.getElementById("loc").
const $ = (id) => document.getElementById(id);

// Formatiranje broja na naš način: 1.234,56 (tačka za hiljade, zarez za decimale).
function nf(v, d = 2) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  const s = Math.abs(v).toFixed(d), [a, b] = s.split(".");
  return (v < 0 && +s !== 0 ? "-" : "") + a.replace(/\B(?=(\d{3})+(?!\d))/g, ".") + (b ? "," + b : "");
}

// Tri modela koja se prikazuju: ključ, ime funkcije u jezgru, naziv, boja i izgled linije.
const SER = [
  { key: "lr", which: "ridge", name: "Linearna regresija", color: "var(--s-lr)", dash: "5 4" },
  { key: "mlp", which: "mlp", name: "MLP", color: "var(--s-mlp)", dash: "" },
  { key: "lstm", which: "lstm", name: "LSTM", color: "var(--s-lstm)", dash: "8 3 2 3" },
];

/* =====================================================================
   1. RAD SA VREMENOM
   Modeli rade po UTC vremenu, a korisnik vidi lokalno vrijeme (ljetno/zimsko).
   ===================================================================== */
const fmtParts = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

// Razlika lokalnog vremena i UTC u satima za trenutak ms (1 zimi, 2 ljeti).
function tzOff(ms) {
  const p = fmtParts.formatToParts(new Date(ms)), g = (t) => +p.find(x => x.type === t).value;
  return (Date.UTC(g("year"), g("month") - 1, g("day"), g("hour") % 24, g("minute")) - ms) / HOUR;
}
// Trenutak lokalne ponoći za datum "GGGG-MM-DD", u ms (UTC).
function localMidnight(dstr) {
  const [y, m, d] = dstr.split("-").map(Number);
  return Date.UTC(y, m - 1, d) - tzOff(Date.UTC(y, m - 1, d, 12)) * HOUR;
}
// Današnji datum po lokalnom vremenu, kao "GGGG-MM-DD".
function todayStr() {
  const p = fmtParts.formatToParts(new Date()), g = (t) => p.find(x => x.type === t).value;
  return `${g("year")}-${g("month")}-${g("day")}`;
}
// Datum pomjeren za n dana.
function addDays(dstr, n) {
  const t = new Date(dstr + "T12:00:00Z");
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}
const utcDate = (ms) => new Date(ms).toISOString().slice(0, 10);

/* =====================================================================
   2. UGRAĐENI PODACI
   U HTML-u su podaci zapisani kao tekst base64 -> komprimovano gzipom -> niz cijelih brojeva.
   Ovdje idemo obrnutim redom i rezultat pamtimo (cache), da se svaka lokacija
   raspakuje samo jednom.
   ===================================================================== */
async function gunzipB64(b64, T = Int16Array) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);          // base64 -> bajtovi
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));   // raspakivanje
  return new T(await new Response(stream).arrayBuffer());             // bajtovi -> niz brojeva tipa T
}
// Izmjereni kappa za Limburg čuva se kao cijeli broj * 10000; -1 znači "nema mjerenja".
const kap = (q) => Float64Array.from(q, v => (v < 0 ? NaN : v / 10000));

const cache = {};
// Vraća pripremljene podatke lokacije: osobine f za svih N sati, a za Limburg i mjerenja.
async function getLoc(name) {
  if (cache[name]) return cache[name];
  const L = M.locations[name];
  const raw = SOLAR.dequantize(await gunzipB64(L.meteo, Int32Array), M.qvars, M.qscale, N);
  const o = { f: SOLAR.features(raw, T0, L.lat, L.lon), t0: T0, n: N };
  if (L.kappa) {                               // samo Limburg ima mjerenja i prognozu operatora
    o.kappa = kap(await gunzipB64(L.kappa));
    o.kda = kap(await gunzipB64(L.kappa_da));
  }
  cache[name] = o;
  return o;
}
// Praćena instalirana snaga svih elektrana u Limburgu (MW) za sat k.
// Čuva se samo kada se promijeni: lista parova [sat od kojeg važi, snaga].
function capacityAt(k) {
  const c = M.locations.Limburg.capacity;
  let v = c[0][1];
  for (const [i, mw] of c) { if (i <= k) v = mw; else break; }
  return v;
}

/* =====================================================================
   3. PODACI UŽIVO (Open-Meteo)
   Za dane poslije ugrađenog perioda, za prognozu i na dugme "Osvježi uživo".
   Preuzima se 4 dana prije izabranog dana (LSTM treba 24 sata istorije, S_72 treba
   72 sata) i jedan dan poslije.
   ===================================================================== */
const liveCache = {};
async function fetchLive(name, dstr) {
  const key = name + dstr;
  if (liveCache[key]) return liveCache[key];
  const L = M.locations[name];
  const lat = name === "Limburg" ? LIMBURG_PTS.lat.join(",") : L.lat;
  const lon = name === "Limburg" ? LIMBURG_PTS.lon.join(",") : L.lon;
  const start = addDays(dstr, -4), end = addDays(dstr, 1);
  // Za posljednjih 60 dana i prognozu koristi se obični API, a za stariji period arhiva
  // prognoza (Historical Forecast API), ista vrsta podataka kao pri treniranju.
  const recent = end >= addDays(todayStr(), -60);
  const base = recent ? "https://api.open-meteo.com/v1/forecast" : "https://historical-forecast-api.open-meteo.com/v1/forecast";
  const url = `${base}?latitude=${lat}&longitude=${lon}&hourly=${OM_VARS.join(",")}&start_date=${start}&end_date=${end}&timezone=GMT`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("Open-Meteo " + r.status);
  let js = await r.json();
  if (!Array.isArray(js)) js = [js];           // za jednu tačku API vraća objekat, za više tačaka niz
  // Prosjek po tačkama (za gradove je to samo jedna tačka). Ako neka tačka nema vrijednost, sat je NaN.
  const n = js[0].hourly.time.length, raw = {};
  M.qvars.forEach((c, i) => {
    const a = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      let s = 0, m = 0;
      js.forEach(p => { const v = p.hourly[OM_VARS[i]][k]; if (v !== null && v !== undefined) { s += v; m++; } });
      a[k] = m === js.length ? s / m : NaN;
    }
    raw[c] = a;
  });
  const t0 = Date.parse(js[0].hourly.time[0] + ":00Z");
  // Iste osobine kao za ugrađene podatke, pa modeli ne znaju odakle podaci dolaze.
  const o = { f: SOLAR.features(raw, t0, L.lat, L.lon), t0, n, live: true, fetched: new Date() };
  liveCache[key] = o;
  return o;
}

/* =====================================================================
   4. STANJE APLIKACIJE
   Sve što korisnik izabere čuva se ovdje; svaka promjena poziva render().
   ===================================================================== */
const state = { loc: "Travnik", date: todayStr(), sizeEdited: false, forceLive: false };
// Scenario "šta ako": g = faktor zračenja (1 = 100 %), t = dodatak temperaturi (°C),
// c = dodatak oblačnosti (procentni poeni).
const scen = { g: 1, t: 0, c: 0 };
// Vraća funkciju koja mijenja sirove osobine prema scenariju, ili null ako scenario nije aktivan.
function modifier() {
  if (scen.g === 1 && scen.t === 0 && scen.c === 0) return null;
  const F = M.features, iG = ["G", "G_b", "G_d", "G_n", "G_lag1", "G_lead1"].map(c => F.indexOf(c));
  const iK = F.indexOf("k_t"), iT = F.indexOf("T_a"), iC = F.indexOf("C");
  return (r) => {
    iG.forEach(i => { r[i] *= scen.g; });                       // sve komponente zračenja
    r[iK] = Math.min(1.2, r[iK] * scen.g);                      // i indeks prozračnosti
    r[iT] += scen.t;
    r[iC] = Math.min(100, Math.max(0, r[iC] + scen.c));         // oblačnost ostaje između 0 i 100 %
  };
}

/* =====================================================================
   5. KONTROLE (padajući meniji, polja, dugmad, klizači, kartice)
   ===================================================================== */
function fillLocSelect(sel) {
  sel.innerHTML = LOCS.map(n => `<option value="${n}">${n === "Limburg" ? "Limburg, Belgija (izmjereno)" : M.locations[n].label + ", BiH"}</option>`).join("");
}
fillLocSelect($("loc")); fillLocSelect($("yloc"));
$("loc").value = state.loc; $("yloc").value = state.loc;
// Dozvoljeni datumi: od 2.1.2021. do 7 dana unaprijed (prognoza).
$("date").min = "2021-01-02"; $("date").max = addDays(todayStr(), 7); $("date").value = state.date;

// Instalirana snaga u kW, i jedinice za prikaz (kW/kWh ili MW/MWh).
function sizeKW() { const v = parseFloat($("psize").value); return (Number.isFinite(v) && v > 0 ? v : 1) * ($("punit").value === "MW" ? 1000 : 1); }
function unit() { return $("punit").value === "MW" ? { p: "MW", e: "MWh", f: 1 / 1000 } : { p: "kW", e: "kWh", f: 1 }; }

// Kvadratići za uključivanje/isključivanje serija na grafikonu (za Limburg i mjerenje i Elia).
function setChecks() {
  const lim = state.loc === "Limburg";
  const items = [...SER.map(s => [s.key, s.name, s.color, true])];
  if (lim) items.unshift(["meas", "izmjerena proizvodnja", "var(--measured)", true]), items.push(["elia", "Elia, prognoza dan unaprijed", "var(--s-elia)", true]);
  const prev = {}; document.querySelectorAll("#checks input").forEach(i => { prev[i.id] = i.checked; });   // zapamti prethodni izbor
  $("checks").innerHTML = items.map(([k, n, c, d]) => `<label><input type="checkbox" id="m-${k}" ${(prev["m-" + k] ?? d) ? "checked" : ""}><span class="sw" style="background:${c}"></span>${n}</label>`).join("");
  document.querySelectorAll("#checks input").forEach(i => i.addEventListener("change", render));
}
// Za Limburg se snaga automatski postavlja na praćenu snagu provincije (dok je korisnik ne promijeni).
function syncSize(k0) {
  if (state.loc === "Limburg" && !state.sizeEdited) {
    const cap = capacityAt(Math.max(0, Math.min(N - 1, k0)));
    $("punit").value = "MW"; $("psize").value = cap.toFixed(1);
    $("psize-note").textContent = "Praćena instalirana snaga svih elektrana u provinciji za izabrani dan (Elia). Moguće je unijeti i drugu vrijednost.";
  } else if (state.loc !== "Limburg") {
    $("psize-note").textContent = "Procjena odgovara prosječnom fotonaponskom sistemu u Limburgu, na čijim podacima je model naučen (različiti nagibi i orijentacije panela).";
  }
}

// Događaji: svaka promjena ažurira stanje i ponovo iscrtava prikaz.
$("loc").addEventListener("change", () => {
  const was = state.loc; state.loc = $("loc").value; $("yloc").value = state.loc;
  // pri prelasku sa Limburga na grad vraća se podrazumijevanih 10 kWp
  if (state.loc === "Limburg" || was === "Limburg") { state.sizeEdited = false; if (state.loc !== "Limburg") { $("punit").value = "kW"; $("psize").value = 10; } }
  setChecks(); render();
});
$("psize").addEventListener("input", () => { state.sizeEdited = true; render(); });
$("punit").addEventListener("change", () => { state.sizeEdited = true; render(); });
$("date").addEventListener("change", () => { if ($("date").value) { state.date = $("date").value; render(); } });
$("prev").addEventListener("click", () => { state.date = addDays(state.date, -1); $("date").value = state.date; render(); });
$("next").addEventListener("click", () => { state.date = addDays(state.date, 1); $("date").value = state.date; render(); });
// "Danas" samo vraća datum na današnji dan.
$("today").addEventListener("click", () => { state.date = todayStr(); $("date").value = state.date; render(); });
// "Osvježi uživo" briše zapamćene podatke uživo i tjera novo preuzimanje sa Open-Meteo.
$("live").addEventListener("click", () => { delete liveCache[state.loc + state.date]; state.forceLive = true; render(); });

// Klizači scenarija: id klizača, id oznake vrijednosti, polje u scen, prikaz, pretvaranje.
function bindSlider(id, out, key, fmt, conv) {
  $(id).addEventListener("input", (e) => { scen[key] = conv(+e.target.value); $(out).textContent = fmt(+e.target.value); render(); });
}
bindSlider("s-g", "o-g", "g", v => v + " %", v => v / 100);
bindSlider("s-t", "o-t", "t", v => (v >= 0 ? "+" : "") + v + " °C", v => v);
bindSlider("s-c", "o-c", "c", v => (v >= 0 ? "+" : "") + v + " p.p.", v => v);
$("reset").addEventListener("click", () => {
  [["s-g", 100], ["s-t", 0], ["s-c", 0]].forEach(([id, v]) => { $(id).value = v; $(id).dispatchEvent(new Event("input")); });
});
// Kartice: prikaži izabranu sekciju, sakrij ostale.
document.querySelectorAll("nav.tabs button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("nav.tabs button").forEach(x => { x.setAttribute("aria-selected", x === b); $(x.dataset.view).hidden = x !== b; });
  if (b.id === "t-god") { fillTransferTable(); }
}));

/* =====================================================================
   6. KARTICA "PROCJENA ZA DAN"
   ===================================================================== */
let renderSeq = 0;   // redni broj iscrtavanja: ako stigne noviji zahtjev, stariji se odbacuje

// Odlučuje odakle se uzimaju podaci za izabrani dan i vraća indeks prvog sata tog dana (k0).
//   - ugrađeni podaci, ako dan (sa 24 sata prije i poslije) postoji u njima;
//   - uživo, za današnji i budući dan, za dan poslije ugrađenog perioda ili na "Osvježi uživo";
//   - ako preuzimanje ne uspije, ugrađeni podaci (ako postoje), inače greška.
async function dayData(name, dstr) {
  const lm = localMidnight(dstr);
  const emb = await getLoc(name);
  const k0 = Math.round((lm - T0) / HOUR);
  const embOk = k0 - 24 >= 0 && k0 + 25 <= N - 1;
  const wantLive = state.forceLive || dstr >= todayStr() || !embOk;
  if (wantLive) {
    try {
      const L = await fetchLive(name, dstr);
      return { d: L, k0: Math.round((lm - L.t0) / HOUR), kEmb: embOk ? k0 : null, emb, src: "live" };
    } catch (e) {
      if (!embOk) return { err: e };
    }
  }
  return { d: emb, k0, kEmb: k0, emb, src: wantLive ? "embedded-fallback" : "embedded" };
}

// GLAVNA FUNKCIJA PRIKAZA: računa 24 satne procjene i iscrtava kartice, grafikon i tabelu.
async function render() {
  const my = ++renderSeq;
  const dstr = state.date, name = state.loc;
  const R = await dayData(name, dstr);
  if (my !== renderSeq) return;                // u međuvremenu je korisnik promijenio izbor
  state.forceLive = false;
  if (R.err) {                                 // nema ni ugrađenih ni preuzetih podataka
    $("msg").hidden = false;
    $("msg").innerHTML = `Za ${fmtDate(dstr)} vremenski podaci nisu ugrađeni u aplikaciju (ugrađeni period: 1.1.2021. - ${fmtDate(utcDate(T0 + (N - 26) * HOUR))}), a preuzimanje sa Open-Meteo nije uspjelo. Za datume poslije tog perioda potrebna je internet veza; ako je stranica otvorena u pregledu koji blokira mrežne zahtjeve, datoteku <code>solmetar.html</code> potrebno je otvoriti u pregledaču.`;
    $("kpis").innerHTML = ""; $("chart").innerHTML = ""; $("legend").innerHTML = ""; $("htable").innerHTML = "";
    $("src").textContent = "nema podataka";
    return;
  }
  $("msg").hidden = true;
  syncSize(R.kEmb ?? N - 1);
  const U = unit(), P = sizeKW() * U.f;        // instalirana snaga u jedinici prikaza
  const mod = modifier();
  $("scen-chip").hidden = !mod;
  const lim = name === "Limburg";

  // Jedan red po satu lokalnog dana: procjene svih modela, mjerenje (Limburg) i vremenski podaci.
  const rows = [];
  for (let i = 0; i < 24; i++) {
    const k = R.k0 + i, ms = R.d.t0 + k * HOUR;
    const lh = Math.round((ms + tzOff(ms) * HOUR) / HOUR) % 24;   // lokalni sat za prikaz
    const pred = {}, base = {};
    SER.forEach(s => {
      pred[s.key] = E.predictAt(R.d.f, k, s.which, mod);                            // sa scenarijem
      base[s.key] = mod ? E.predictAt(R.d.f, k, s.which, null) : pred[s.key];      // stvarni uslovi
    });
    const ke = R.kEmb !== null ? R.kEmb + i : null;             // isti sat u ugrađenim podacima
    const meas = lim && ke !== null && ke < N ? R.emb.kappa[ke] : NaN;
    const elia = lim && ke !== null && ke < N ? R.emb.kda[ke] : NaN;
    const g = (c) => R.d.f[c][k];
    rows.push({ lh, pred, base, meas, elia, fsun: g("f_sun"), G: g("G"), Gb: g("G_b"), Gd: g("G_d"), C: g("C"), T: g("T_a"), S72: g("S_72") });
  }
  const srcTxt = R.src === "live" ? `Open-Meteo uživo, preuzeto ${R.d.fetched.toLocaleTimeString("bs-BA", { hour: "2-digit", minute: "2-digit" })}${dstr > todayStr() ? " (prognoza)" : ""}`
    : R.src === "embedded-fallback" ? "ugrađeni podaci (veza sa Open-Meteo nije dostupna)" : "ugrađeni podaci Open-Meteo";
  $("src").textContent = "Izvor: " + srcTxt;
  drawKpis(rows, P, U, lim);
  $("chart-title").textContent = `Satna snaga, ${fmtDate(dstr)} (${lim ? "Limburg" : M.locations[name].label})`;
  drawDay(rows, P, U, lim, !!mod);
  drawTable(rows, P, U, lim);
}
function fmtDate(dstr) { const [y, m, d] = dstr.split("-"); return `${+d}.${+m}.${y}.`; }
const shown = (k) => { const el = $("m-" + k); return el ? el.checked : false; };   // da li je serija uključena

// Kartice iznad grafikona: dnevna energija po modelu. Energija dana = zbir satnih kappa * P
// (snaga tokom jednog sata daje energiju u kWh). Za Limburg i MAE u odnosu na mjerenje.
function drawKpis(rows, P, U, lim) {
  const sum = (f) => rows.reduce((a, r) => { const v = f(r); return a + (Number.isFinite(v) ? v : 0); }, 0);
  let html = "";
  const hasMeas = lim && rows.some(r => Number.isFinite(r.meas));
  if (lim) {
    html += `<div class="kpi"><div class="k"><span class="sw" style="background:var(--measured)"></span>Izmjereno</div><div class="v">${hasMeas ? nf(sum(r => r.meas) * P, 1) : "-"}<small>${U.e}</small></div><div class="d">${hasMeas ? nf(sum(r => r.meas), 2) + " kWh/kWp" : "nema mjerenja"}</div></div>`;
  }
  SER.forEach(s => {
    const e = sum(r => r.pred[s.key]);
    const pairs = rows.filter(r => r.fsun > 0 && Number.isFinite(r.meas) && r.pred[s.key] !== null);   // dnevni sati sa mjerenjem
    const mae = pairs.length ? 100 * pairs.reduce((a, r) => a + Math.abs(r.meas - r.pred[s.key]), 0) / pairs.length : null;
    const eb = sum(r => r.base[s.key]);
    const d = modifier() ? `scenario: ${e - eb >= 0 ? "+" : ""}${nf((e - eb) * P, 1)} ${U.e}` : pairs.length ? `MAE ${nf(mae, 2)} % P<sub>nom</sub>` : `${nf(e, 2)} kWh/kWp`;
    html += `<div class="kpi" style="${shown(s.key) ? "" : "opacity:.45"}"><div class="k"><span class="sw" style="background:${s.color}"></span>${s.name}</div><div class="v">${nf(e * P, 1)}<small>${U.e}</small></div><div class="d">${d}</div></div>`;
  });
  if (lim && rows.some(r => Number.isFinite(r.elia))) {
    html += `<div class="kpi" style="${shown("elia") ? "" : "opacity:.45"}"><div class="k"><span class="sw" style="background:var(--s-elia)"></span>Elia, dan unaprijed</div><div class="v">${nf(sum(r => r.elia) * P, 1)}<small>${U.e}</small></div><div class="d">prognoza operatora</div></div>`;
  }
  $("kpis").innerHTML = html;
}

// Pomoćna funkcija: novi SVG element sa atributima (i tekstom).
function svgEl(tag, attrs, text) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
// "Lijep" gornji kraj ose: najmanji od 1, 1,5, 2, 2,5... puta stepen broja 10 koji je >= v.
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
// Linijski grafikon satne snage, crtan ručno u SVG-u (bez biblioteka).
// x(h) i y(v) pretvaraju sat i snagu u piksele unutar površine grafikona.
function drawDay(rows, P, U, lim, scenario) {
  const svg = $("chart"); svg.innerHTML = "";
  const W = 760, H = 320, m = { l: 54, r: 16, t: 12, b: 34 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const series = SER.filter(s => shown(s.key));
  // najveća prikazana vrijednost određuje visinu ose
  let ymax = 0.05 * P;
  rows.forEach(r => {
    if (lim && shown("meas") && Number.isFinite(r.meas)) ymax = Math.max(ymax, r.meas * P);
    if (lim && shown("elia") && Number.isFinite(r.elia)) ymax = Math.max(ymax, r.elia * P);
    series.forEach(s => { [r.pred[s.key], scenario ? r.base[s.key] : null].forEach(v => { if (v != null) ymax = Math.max(ymax, v * P); }); });
  });
  ymax = niceMax(ymax * 1.02);
  const x = (h) => m.l + (h / 23) * iw, y = (v) => m.t + ih - (v / ymax) * ih;
  // mreža i oznake na osama
  for (let i = 0; i <= 5; i++) {
    const v = ymax * i / 5;
    svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "gridl" }));
    svg.appendChild(svgEl("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, nf(v, v >= 10 || v === 0 ? 0 : v >= 1 ? 1 : 2)));
  }
  for (let h = 0; h <= 23; h += 3) svg.appendChild(svgEl("text", { x: x(h), y: H - 12, "text-anchor": "middle" }, String(h).padStart(2, "0") + ":00"));
  svg.appendChild(svgEl("text", { x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle" }, `P (${U.p})`));
  // SVG putanja kroz satne vrijednosti: "M" počinje liniju, "L" je nastavlja; nepoznata vrijednost prekida liniju.
  const path = (vals) => { let d = "", pen = false; vals.forEach((v, h) => { if (v === null || !Number.isFinite(v)) { pen = false; return; } d += (pen ? "L" : "M") + x(h).toFixed(1) + " " + y(v * P).toFixed(1) + " "; pen = true; }); return d; };
  if (lim && shown("meas")) {                  // izmjerena proizvodnja: blago obojena površina i linija
    const pts = rows.map((r, h) => Number.isFinite(r.meas) ? [x(h), y(r.meas * P)] : null).filter(Boolean);
    if (pts.length) svg.appendChild(svgEl("path", { d: `M${pts[0][0]} ${y(0)} ` + pts.map(p => `L${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ") + ` L${pts[pts.length - 1][0]} ${y(0)} Z`, fill: "var(--measured)", "fill-opacity": 0.08, stroke: "none" }));
    svg.appendChild(svgEl("path", { d: path(rows.map(r => r.meas)), fill: "none", stroke: "var(--measured)", "stroke-width": 2.2 }));
  }
  if (lim && shown("elia")) svg.appendChild(svgEl("path", { d: path(rows.map(r => r.elia)), fill: "none", stroke: "var(--s-elia)", "stroke-width": 1.8, "stroke-dasharray": "2 3" }));
  // uz aktivan scenario: blijede linije za stvarne uslove
  if (scenario) series.forEach(s => svg.appendChild(svgEl("path", { d: path(rows.map(r => r.base[s.key])), fill: "none", stroke: s.color, "stroke-width": 1.2, "stroke-opacity": 0.4, "stroke-dasharray": s.dash })));
  series.forEach(s => svg.appendChild(svgEl("path", { d: path(rows.map(r => r.pred[s.key])), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash, "stroke-linejoin": "round" })));
  // legenda ispod grafikona
  const lg = [];
  if (lim && shown("meas")) lg.push(`<span><span class="sw" style="background:var(--measured)"></span>izmjereno</span>`);
  series.forEach(s => lg.push(`<span><span class="sw" style="background:${s.color}"></span>${s.name}${scenario ? " (scenario)" : ""}</span>`));
  if (lim && shown("elia")) lg.push(`<span><span class="sw" style="background:var(--s-elia)"></span>Elia, dan unaprijed</span>`);
  if (scenario) lg.push(`<span style="opacity:.7">blijedo: procjene za stvarne uslove</span>`);
  $("legend").innerHTML = lg.join("");
}
// Tabela satnih podataka (otvara se ispod grafikona).
function drawTable(rows, P, U, lim) {
  let t = `<thead><tr><th>sat</th><th>G (W/m²)</th><th>G<sub>b</sub></th><th>G<sub>d</sub></th><th>C (%)</th><th>T<sub>a</sub> (°C)</th><th>κ̂ MLP (%)</th>${SER.map(s => `<th>${s.name.replace("Linearna regresija", "Lin. reg.")} (${U.p})</th>`).join("")}${lim ? `<th>izmjereno (${U.p})</th>` : ""}</tr></thead><tbody>`;
  rows.forEach(r => {
    t += `<tr class="${r.fsun > 0 ? "" : "night"}"><td>${String(r.lh).padStart(2, "0")}:00</td><td>${nf(r.G, 0)}</td><td>${nf(r.Gb, 0)}</td><td>${nf(r.Gd, 0)}</td><td>${nf(r.C, 0)}</td><td>${nf(r.T, 1)}</td><td>${nf(r.pred.mlp == null ? null : 100 * r.pred.mlp, 1)}</td>${SER.map(s => `<td>${nf(r.pred[s.key] == null ? null : r.pred[s.key] * P, 2)}</td>`).join("")}${lim ? `<td>${nf(r.meas * P, 2)}</td>` : ""}</tr>`;
  });
  $("htable").innerHTML = t + "</tbody>";
}

/* =====================================================================
   7. KARTICA "GODIŠNJA PROIZVODNJA"
   MLP procjena za svaki sat izabrane godine, sabrana po mjesecima i upoređena sa PVGIS.
   ===================================================================== */
const MN = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "avg", "sep", "okt", "nov", "dec"];
(function fillYears() {
  const last = +todayStr().slice(0, 4);
  let h = ""; for (let y = last; y >= 2021; y--) h += `<option value="${y}">${y}.${y === last ? " (do danas)" : ""}</option>`;
  $("year").innerHTML = h; $("year").value = String(Math.min(last, 2025));
})();
$("yloc").addEventListener("change", () => { $("loc").value = $("yloc").value; $("loc").dispatchEvent(new Event("change")); });
$("yrun").addEventListener("click", runYear);
async function runYear() {
  const name = $("yloc").value, yr = +$("year").value;
  $("yrun").disabled = true;
  const L = await getLoc(name);
  // raspon sati izabrane godine u ugrađenim podacima (tekuća godina samo do danas)
  const k1 = Math.max(0, Math.round((Date.UTC(yr, 0, 1) - T0) / HOUR));
  const kEnd = Math.min(N, Math.round((Date.UTC(yr + 1, 0, 1) - T0) / HOUR), Math.round((Date.now() - T0) / HOUR));
  const mon = new Array(12).fill(0), meas = new Array(12).fill(0), measN = new Array(12).fill(0);
  let tot = 0, totMeas = 0, hasMeas = false;
  for (let k = k1; k < kEnd; k++) {
    const mo = new Date(T0 + k * HOUR).getUTCMonth();
    const v = E.predictAt(L.f, k, "mlp", null) ?? 0;        // kWh po kWp za taj sat
    mon[mo] += v; tot += v;
    if (L.kappa && Number.isFinite(L.kappa[k])) { meas[mo] += L.kappa[k]; totMeas += L.kappa[k]; measN[mo]++; hasMeas = true; }
  }
  const T = M.transfer[name], pv = T.PVGIS_E_m;
  const partial = kEnd < Math.round((Date.UTC(yr + 1, 0, 1) - T0) / HOUR);
  const S = sizeKW();
  let h = `<div class="kpi"><div class="k"><span class="sw" style="background:var(--s-mlp)"></span>MLP, ${yr}.${partial ? " (do danas)" : ""}</div><div class="v">${nf(tot, 0)}<small>kWh/kWp</small></div><div class="d">za ${nf(S, 1)} kWp: ${nf(tot * S, 0)} kWh</div></div>`;
  h += `<div class="kpi"><div class="k"><span class="sw" style="background:var(--sun)"></span>PVGIS, prosječna godina</div><div class="v">${nf(T.PVGIS_E_y, 0)}<small>kWh/kWp</small></div><div class="d">optimalni ugao, gubici 14 %</div></div>`;
  if (hasMeas) h += `<div class="kpi"><div class="k"><span class="sw" style="background:var(--measured)"></span>Izmjereno, ${yr}.</div><div class="v">${nf(totMeas, 0)}<small>kWh/kWp</small></div><div class="d">elektrane u Limburgu</div></div>`;
  const pyv = T.MLP[String(yr)];
  h += `<div class="kpi"><div class="k">Odnos MLP / PVGIS</div><div class="v">${partial ? "-" : nf(tot / T.PVGIS_E_y, 2)}</div><div class="d">${pyv !== undefined ? "Python: " + nf(pyv, 1) + " kWh/kWp" : "godina nije u analizi rada"}</div></div>`;
  $("ykpis").innerHTML = h;
  const ser = [["mlp", mon, "var(--s-mlp)", "MLP (izabrana godina)"], ["pv", pv, "var(--sun)", "PVGIS (prosječna godina)"]];
  if (hasMeas) ser.unshift(["meas", meas, "var(--measured)", "izmjereno"]);
  drawBars($("ychart"), $("ylegend"), MN, ser, "kWh/kWp");
  $("yrun").disabled = false;
}
// Stubičasti grafikon u SVG-u: labels su grupe (mjeseci), ser su serije [ključ, vrijednosti, boja, naziv].
function drawBars(svg, legend, labels, ser, ylab) {
  svg.innerHTML = "";
  const W = 760, H = 260, m = { l: 52, r: 10, t: 10, b: 30 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  let mx = 0; ser.forEach(s => s[1].forEach(v => { mx = Math.max(mx, v); }));
  mx = niceMax(mx * 1.02); const y = (v) => m.t + ih - v / mx * ih;
  for (let i = 0; i <= 5; i++) { const v = mx * i / 5; svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "gridl" })); svg.appendChild(svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, nf(v, 0))); }
  svg.appendChild(svgEl("text", { x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle" }, ylab));
  const gw = iw / labels.length, bw = gw * 0.8 / ser.length;   // širina grupe i jednog stubića
  labels.forEach((lab, i) => {
    ser.forEach(([, vals, c], j) => svg.appendChild(svgEl("rect", { x: m.l + i * gw + gw * 0.1 + j * bw, y: y(vals[i]), width: Math.max(1, bw - 1), height: Math.max(0, y(0) - y(vals[i])), fill: c, "fill-opacity": 0.88 })));
    svg.appendChild(svgEl("text", { x: m.l + i * gw + gw / 2, y: H - 10, "text-anchor": "middle" }, lab));
  });
  legend.innerHTML = ser.map(([, , c, n]) => `<span><span class="sw" style="background:${c};height:10px;width:10px"></span>${n}</span>`).join("");
}
// Tabela prenosa modela na gradove u BiH (rezultati iz Pythona, transfer.json).
function fillTransferTable() {
  const avg = (o) => { const v = Object.values(o); return v.reduce((a, b) => a + b, 0) / v.length; };
  let t = `<thead><tr><th>Lokacija</th><th>PVGIS</th><th>MLP</th><th>LSTM</th><th>MLP / PVGIS</th><th>MLP / MLP Limburg</th><th>PVGIS / PVGIS Limburg</th></tr></thead><tbody>`;
  const LM = avg(M.transfer.Limburg.MLP), LP = M.transfer.Limburg.PVGIS_E_y;
  LOCS.forEach(n => {
    const T = M.transfer[n], mm = avg(T.MLP), ml = avg(T.LSTM);
    t += `<tr><td>${n === "Limburg" ? "Limburg" : M.locations[n].label}</td><td>${nf(T.PVGIS_E_y, 0)}</td><td>${nf(mm, 0)}</td><td>${nf(ml, 0)}</td><td>${nf(mm / T.PVGIS_E_y, 2)}</td><td>${nf(mm / LM, 2)}</td><td>${nf(T.PVGIS_E_y / LP, 2)}</td></tr>`;
  });
  $("ttable").innerHTML = t + "</tbody>";
}

/* =====================================================================
   8. KARTICA "EVALUACIJA I PROVJERA"
   Ponovo računa sve testne sate u Limburgu (1.7.2025. - 21.9.2026.) u pregledaču,
   računa MAE, RMSE, MBE i R² (samo dnevni sati) i poredi svaku procjenu sa Python
   procjenom iz REF. Računa se u komadima od 150 sati (setTimeout), da ekran ne zastane.
   ===================================================================== */
let evalDone = false;
$("run").addEventListener("click", runEval);
async function runEval() {
  if (evalDone) return;
  $("run").disabled = true;
  const L = await getLoc("Limburg"), ref = REF.Limburg;
  const kr = Math.round((Date.parse(ref.t0.replace(" ", "T") + "Z") - T0) / HOUR), n = ref.n;   // prvi testni sat
  // sabirači: se = zbir kvadrata grešaka, ae = zbir apsolutnih grešaka, be = zbir grešaka (pristrasnost),
  // mx = najveće odstupanje od Pythona
  const acc = {}; SER.forEach(s => { acc[s.key] = { se: 0, ae: 0, be: 0, mx: 0 }; });
  const el = { se: 0, ae: 0, be: 0 };
  const ys = [], mon = {};
  const t0 = performance.now();
  let i = 0;
  const step = () => {
    const end = Math.min(n, i + 150);
    for (; i < end; i++) {
      const k = kr + i, y = L.kappa[k], day = L.f.f_sun[k] > 0 && Number.isFinite(y);
      const mo = new Date(T0 + k * HOUR).toISOString().slice(0, 7);
      mon[mo] = mon[mo] || { y: 0, lr: 0, mlp: 0, lstm: 0, elia: 0 };
      SER.forEach(s => {
        const p = E.predictAt(L.f, k, s.which, null) ?? 0, a = acc[s.key];
        a.mx = Math.max(a.mx, Math.abs(p - ref[s.which][i]));
        if (day) { a.se += (y - p) ** 2; a.ae += Math.abs(y - p); a.be += p - y; mon[mo][s.key] += p; }
      });
      if (day) {
        ys.push(y); mon[mo].y += y;
        const pe = L.kda[k]; el.se += (y - pe) ** 2; el.ae += Math.abs(y - pe); el.be += pe - y; mon[mo].elia += pe;
      }
    }
    $("bar").style.width = (100 * i / n).toFixed(1) + "%";
    $("runinfo").textContent = `obrađeno ${nf(i, 0)} / ${nf(n, 0)} sati`;
    if (i < n) setTimeout(step, 0); else finish();
  };
  const finish = () => {
    // R² = 1 - zbir kvadrata grešaka / zbir kvadrata odstupanja od prosjeka
    const m = ys.length, ym = ys.reduce((a, b) => a + b, 0) / m, sst = ys.reduce((a, v) => a + (v - ym) ** 2, 0);
    const py = M.metrics_python, pyName = { lr: "Linearna regresija", mlp: "MLP", lstm: "LSTM" };
    const row = (name, a, pyv, mx) => `<tr><td>${name}</td><td>${nf(100 * a.ae / m, 3)}</td><td>${nf(100 * Math.sqrt(a.se / m), 3)}</td><td>${nf(100 * a.be / m, 3)}</td><td>${nf(1 - a.se / sst, 4)}</td><td>${nf(pyv, 3)}</td><td>${mx === null ? "-" : `<span class="okmark">${mx.toExponential(1).replace(".", ",")}</span>`}</td></tr>`;
    let t = `<thead><tr><th>Model</th><th>MAE (%)</th><th>RMSE (%)</th><th>MBE (%)</th><th>R²</th><th>RMSE u Pythonu (%)</th><th>maks. |JS - Python| (κ)</th></tr></thead><tbody>`;
    SER.forEach(s => { t += row(s.name, acc[s.key], py[pyName[s.key]].RMSE, acc[s.key].mx); });
    t += row("Elia, dan unaprijed", el, py["Elia, dan unaprijed"].RMSE, null);
    $("etable").innerHTML = t + "</tbody>";
    const worst = Math.max(...SER.map(s => acc[s.key].mx));
    $("runinfo").innerHTML = `Gotovo za ${nf((performance.now() - t0) / 1000, 1)} s, ${nf(m, 0)} dnevnih sati. Greške su u procentima instalirane snage. ` +
      (worst < 1e-4 ? `<span class="okmark">Procjene u pregledaču podudaraju se sa Python implementacijom.</span>` : `<span style="color:var(--warn)">Odstupanje od Python implementacije veće od očekivanog.</span>`);
    const keys = Object.keys(mon).sort();
    drawBars($("mchart"), $("mlegend"), keys.map(k => MN[+k.slice(5) - 1] + " " + k.slice(2, 4)),
      [["y", keys.map(k => mon[k].y), "var(--measured)", "izmjereno"], ["mlp", keys.map(k => mon[k].mlp), "var(--s-mlp)", "MLP"],
       ["lstm", keys.map(k => mon[k].lstm), "var(--s-lstm)", "LSTM"], ["elia", keys.map(k => mon[k].elia), "var(--s-elia)", "Elia, dan unaprijed"]], "kWh/kWp");
    evalDone = true;
  };
  step();
}

/* =====================================================================
   9. POKRETANJE: nacrtaj kvadratiće serija i prvi prikaz (danas, Travnik, 10 kWp).
   ===================================================================== */
setChecks();
render();
