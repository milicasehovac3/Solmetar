// =====================================================================================
// core.js - RAČUNSKO JEZGRO APLIKACIJE SOLMETAR
//
// Ova datoteka ne crta ništa na ekranu. Ona radi samo "matematiku":
//   1. iz sirovih satnih vremenskih podataka (Open-Meteo) računa 18 ulaznih osobina
//      (solarna geometrija, indeks prozračnosti, susjedni sati...), tačno kao Python modul
//      solar_nn/prep.py;
//   2. izvršava tri istrenirana modela (linearna regresija, MLP i LSTM) sa težinama
//      izvezenim iz Pythona (solar_nn/export_app.py), tačno kao PyTorch.
//
// Rezultat svakog modela je procijenjeni faktor iskorištenja kappa = P / P_nom
// (udio instalirane snage, broj od 0 do 1). Snaga u kW je kappa * instalirana snaga.
//
// Sve je upakovano u jedan objekat SOLAR (tzv. "modul" obrazac), da imena funkcija
// ne bi smetala ostatku aplikacije. Ista datoteka radi i u pregledaču i u Node.js
// (verify.js je koristi za provjeru prema Pythonu).
// =====================================================================================
"use strict";

const SOLAR = (() => {
  const RAD = Math.PI / 180;   // pretvaranje stepeni u radijane
  const G_SC = 1367.0;         // solarna konstanta (W/m²)
  const HOUR = 3600000;        // jedan sat u milisekundama (JavaScript vrijeme je u ms)

  // Open-Meteo zračenje i padavine sa oznakom t+1 opisuju PROTEKLI sat (od t do t+1).
  // Zato se ove veličine pomjeraju za jedan sat unazad (INTERVAL).
  // Temperatura, vlažnost, vjetar i oblačnost su trenutne vrijednosti, pa se za sat
  // uzima prosjek početka i kraja sata (INSTANT).
  const INTERVAL = ["G", "G_b", "G_d", "G_n", "R", "S"];
  const INSTANT = ["T_a", "RH", "v_w", "C"];

  // ------------------------------------------------------------------
  // 1. SOLARNA GEOMETRIJA (formule (1)-(5) i (20)-(21) u radu)
  // ------------------------------------------------------------------

  // Deklinacija Sunca (Cooper): ugao između Sunca i ravni ekvatora za dan u godini doy.
  const declination = (doy) => RAD * 23.45 * Math.sin(RAD * (360 / 365) * (284 + doy));

  // Jednačina vremena (Spencer, 1971), u minutama: razlika pravog i srednjeg solarnog vremena.
  function equationOfTime(doy) {
    const B = RAD * (360 / 365) * (doy - 1);   // ugao dana Γ
    return 229.18 * (0.000075 + 0.001868 * Math.cos(B) - 0.032077 * Math.sin(B)
      - 0.014615 * Math.cos(2 * B) - 0.040890 * Math.sin(2 * B));
  }

  // Kosinus zenitnog ugla Sunca za trenutak tUtcHours (sati po UTC), dan doy i lokaciju.
  // Vraća [cos_z, omega], gdje je omega satni ugao (0 u solarno podne).
  function cosZenith(tUtcHours, doy, lat, lon) {
    const tst = tUtcHours + lon / 15 + equationOfTime(doy) / 60;   // pravo solarno vrijeme H
    const omega = RAD * 15 * (tst - 12);                           // satni ugao
    const phi = RAD * lat, d = declination(doy);
    const cz = Math.sin(phi) * Math.sin(d) + Math.cos(phi) * Math.cos(d) * Math.cos(omega);
    return [cz, omega];
  }

  // Redni broj dana u godini (1. januar = 1) za vrijeme ms (milisekunde, UTC).
  function dayOfYear(ms) {
    const d = new Date(ms);
    const start = Date.UTC(d.getUTCFullYear(), 0, 1);
    const day = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    return Math.round((day - start) / 86400000) + 1;
  }

  // ------------------------------------------------------------------
  // 2. PRIPREMA OSOBINA (prevod funkcija align_meteo i features iz prep.py)
  //
  // raw: objekat sa nizovima {G, G_b, G_d, G_n, T_a, RH, v_w, R, S, C}, jedna vrijednost
  //      po satu, sa oznakama vremena kao na Open-Meteo (UTC);
  // t0:  vrijeme prvog sata u ms;  lat, lon: lokacija.
  // Vraća objekat f u kojem je svaka osobina niz dužine N (f.G[k], f.cos_z[k], ...).
  // Vrijednost NaN znači "nije poznato" (npr. prvi sat nema prethodni sat).
  // ------------------------------------------------------------------
  function features(raw, t0, lat, lon) {
    const N = raw.G.length, f = {};
    const arr = () => new Float64Array(N).fill(NaN);   // novi niz popunjen sa NaN

    // 2a. Poravnanje: veličine intervala pomjeri za sat unazad...
    INTERVAL.forEach(c => {
      const a = arr();
      for (let k = 0; k < N - 1; k++) a[k] = raw[c][k + 1];
      f[c] = a;
    });
    // ...a trenutne veličine zamijeni prosjekom početka i kraja sata.
    INSTANT.forEach(c => {
      const a = arr();
      for (let k = 0; k < N - 1; k++) a[k] = 0.5 * (raw[c][k] + raw[c][k + 1]);
      f[c] = a;
    });

    // 2b. Izvedene osobine, sat po sat.
    ["cos_z", "G_0", "f_sun", "w_sin", "k_t", "S_72", "G_lag1", "G_lead1"].forEach(c => { f[c] = arr(); });
    for (let k = 0; k < N; k++) {
      const ms = t0 + k * HOUR, doy = dayOfYear(ms), hr = new Date(ms).getUTCHours();
      const E0 = 1 + 0.033 * Math.cos(RAD * 360 * doy / 365);   // korekcija udaljenosti Zemlja-Sunce

      // Položaj Sunca se mijenja tokom sata, pa se računa u 4 tačke (svakih 15 minuta)
      // i uprosječuje. f_sun je udio sata u kojem je Sunce iznad horizonta.
      let cz = 0, up = 0;
      for (let j = 0; j < 4; j++) {
        const c = cosZenith(hr + (j + 0.5) / 4, doy, lat, lon)[0];
        cz += Math.max(0, c);
        up += c > 0 ? 1 : 0;
      }
      f.cos_z[k] = cz / 4;                       // srednji cos zenitnog ugla (0 noću)
      f.G_0[k] = G_SC * E0 * f.cos_z[k];         // vanatmosfersko zračenje (W/m²)
      f.f_sun[k] = up / 4;
      // sin satnog ugla u sredini sata: razlikuje jutro (negativno) od popodneva (pozitivno)
      f.w_sin[k] = f.f_sun[k] > 0 ? Math.sin(cosZenith(hr + 0.5, doy, lat, lon)[1]) : 0;

      // Indeks prozračnosti k_t = G / G_0 (formula (22)): koliko zračenja prolazi kroz atmosferu.
      // Kad je Sunce vrlo nisko (G_0 <= 10) postavlja se na 0, a ograničen je na [0; 1,2].
      const g = f.G[k];
      f.k_t[k] = (f.G_0[k] > 10 && Number.isFinite(g)) ? Math.min(1.2, Math.max(0, g / f.G_0[k])) : 0;

      // S_72: ukupni snijeg u posljednja 72 sata (snijeg na panelima smanjuje proizvodnju).
      let s = 0, n = 0;
      for (let j = Math.max(0, k - 71); j <= k; j++) if (Number.isFinite(f.S[j])) { s += f.S[j]; n++; }
      f.S_72[k] = n ? s : NaN;

      // Zračenje prethodnog i narednog sata (pomaže kod naglih promjena oblačnosti).
      f.G_lag1[k] = k > 0 ? f.G[k - 1] : NaN;
      f.G_lead1[k] = k < N - 1 ? f.G[k + 1] : NaN;
    }
    return f;
  }

  // ------------------------------------------------------------------
  // 3. MODELI
  //
  // buildEngine(M) prima objekat M (sadržaj app_model.json) sa težinama modela i
  // vraća funkcije za procjenu. Težine se jednom pretvore u brze nizove (Float64Array).
  // ------------------------------------------------------------------
  function buildEngine(M) {
    const F = M.features;          // imena 18 osobina, istim redoslijedom kao u Pythonu
    const SF = M.seq_features;     // 15 osobina koje koristi LSTM
    const K = F.length;
    // Parametri standardizacije (formula (14)): x_std = (x - srednja vrijednost) / std. devijacija,
    // izračunati samo na podacima za treniranje.
    const mean = Float64Array.from(M.scaler_mean), scale = Float64Array.from(M.scaler_scale);
    const seqIdx = SF.map(c => F.indexOf(c));   // pozicije LSTM osobina u punom vektoru

    // Matrica se čuva kao jedan ravan niz d, red po red (r redova, c kolona).
    const toMat = (a) => ({ r: a.length, c: a[0].length, d: Float64Array.from(a.flat()) });
    const toVec = (a) => Float64Array.from(a);

    // Afina transformacija z = W x + b (osnovna operacija svakog sloja mreže).
    function affine(W, b, x) {
      const out = new Float64Array(W.r);
      for (let i = 0; i < W.r; i++) {
        let s = b[i];
        const o = i * W.c;                       // početak i-tog reda u ravnom nizu
        for (let j = 0; j < W.c; j++) s += W.d[o + j] * x[j];
        out[i] = s;
      }
      return out;
    }
    const sig = (z) => 1 / (1 + Math.exp(-z));                 // sigmoidna funkcija
    const relu = (z) => { for (let i = 0; i < z.length; i++) z[i] = z[i] > 0 ? z[i] : 0; return z; };
    const clip01 = (v) => Math.min(1, Math.max(0, v));         // proizvodnja je između 0 i P_nom

    // --- 3a. Linearna regresija (formula (24)): kappa = b + suma(w_k * x_k) ---
    const rc = toVec(M.ridge.coef), ri = M.ridge.intercept;
    const ridge = (x) => {
      let s = ri;
      for (let k = 0; k < K; k++) s += rc[k] * x[k];
      return clip01(s);
    };

    // --- 3b. MLP (formula (10)): 18 -> 256 -> 128 -> 64 -> 1, aktivacija ReLU ---
    // U PyTorch rječniku težine se zovu "net.0.weight", "net.3.weight", "net.6.weight" i
    // "net.9.weight"; sortiramo ih po broju sloja i uz svaku težinu uzimamo odgovarajući
    // pomak (bias). Dropout se koristi samo pri treniranju, pa ga ovdje nema.
    const wk = Object.keys(M.mlp.state).filter(k => k.endsWith(".weight"))
      .sort((a, b) => parseInt(a.split(".")[1]) - parseInt(b.split(".")[1]));
    const layers = wk.map(k => ({ W: toMat(M.mlp.state[k]), b: toVec(M.mlp.state[k.replace("weight", "bias")]) }));
    function mlp(x) {
      let a = x;
      layers.forEach((L, l) => {
        const z = affine(L.W, L.b, a);
        if (l < layers.length - 1) relu(z);     // ReLU na skrivenim slojevima, izlazni sloj je linearan
        a = z;
      });
      return clip01(a[0]);
    }

    // --- 3c. LSTM (formule (15)-(17)): 2 sloja po 32 ćelije, prozor od 24 sata ---
    // PyTorch ima dva pomaka (bias_ih i bias_hh); pošto se uvijek sabiraju, spajamo ih u jedan.
    const H = M.lstm.hidden, NL = M.lstm.layers, S = M.lstm.state, lstmL = [];
    for (let l = 0; l < NL; l++) {
      const bih = S[`lstm.bias_ih_l${l}`], bhh = S[`lstm.bias_hh_l${l}`];
      lstmL.push({
        Wih: toMat(S[`lstm.weight_ih_l${l}`]),   // težine ulaza
        Whh: toMat(S[`lstm.weight_hh_l${l}`]),   // težine prethodnog skrivenog stanja
        b: Float64Array.from(bih.map((v, i) => v + bhh[i])),
      });
    }
    // "Glava" mreže: posljednje skriveno stanje (32) -> 32 (ReLU) -> 1
    const head1 = { W: toMat(S["head.0.weight"]), b: toVec(S["head.0.bias"]) };
    const head2 = { W: toMat(S["head.2.weight"]), b: toVec(S["head.2.bias"]) };

    // seq: niz od 24 standardizovana vektora osobina (od najstarijeg do trenutnog sata).
    function lstm(seq) {
      // uzmi samo 15 LSTM osobina; nepoznate vrijednosti (NaN) zamijeni sa 0 (= prosjek)
      let inputs = seq.map(x => Float64Array.from(seqIdx.map(k => (Number.isNaN(x[k]) ? 0 : x[k]))));
      const g = new Float64Array(4 * H);         // predaktivacije četiri kapije
      for (let l = 0; l < NL; l++) {
        const L = lstmL[l];
        const h = new Float64Array(H);           // skriveno stanje h, na početku 0
        const c = new Float64Array(H);           // stanje ćelije c (dugoročna memorija), na početku 0
        const outs = [];
        for (let t = 0; t < inputs.length; t++) {   // prolaz kroz 24 sata, redom
          const xt = inputs[t];
          // g = W_ih x_t + W_hh h_(t-1) + b, za sve četiri kapije odjednom
          for (let i = 0; i < 4 * H; i++) {
            let s = L.b[i];
            const oi = i * L.Wih.c, oh = i * H;
            for (let j = 0; j < L.Wih.c; j++) s += L.Wih.d[oi + j] * xt[j];
            for (let j = 0; j < H; j++) s += L.Whh.d[oh + j] * h[j];
            g[i] = s;
          }
          // Redoslijed kapija u PyTorch-u je: ulazna (i), zaboravljanja (f), kandidat (g), izlazna (o).
          for (let j = 0; j < H; j++) {
            const ig = sig(g[j]);
            const fg = sig(g[H + j]);
            const cg = Math.tanh(g[2 * H + j]);
            const og = sig(g[3 * H + j]);
            c[j] = fg * c[j] + ig * cg;          // zaboravi dio starog, dodaj dio novog
            h[j] = og * Math.tanh(c[j]);         // novo skriveno stanje
          }
          outs.push(Float64Array.from(h));
        }
        inputs = outs;                           // izlazi prvog sloja su ulazi drugog sloja
      }
      // procjena iz skrivenog stanja posljednjeg (trenutnog) sata
      const z = relu(affine(head1.W, head1.b, inputs[inputs.length - 1]));
      return clip01(affine(head2.W, head2.b, z)[0]);
    }

    // Standardizovani vektor osobina za sat k.
    // mod je opciona funkcija koja mijenja sirove osobine prije standardizacije
    // (koristi je scenario "šta ako", npr. zračenje * 0,7).
    function vec(f, k, mod) {
      const r = F.map(c => f[c][k]);
      if (mod) mod(r, F);
      return r.map((v, i) => (v - mean[i]) / scale[i]);
    }

    // GLAVNA FUNKCIJA: procjena kappa za sat k modelom which ("ridge", "mlp" ili "lstm").
    // Vraća 0 noću, a null ako za taj sat nedostaju podaci.
    function predictAt(f, k, which, mod) {
      if (!(f.f_sun[k] > 0)) return 0;                     // noć: proizvodnja je 0
      const x = vec(f, k, mod);
      if (x.some(v => !Number.isFinite(v))) return null;
      if (which === "ridge") return ridge(x);
      if (which === "mlp") return mlp(x);
      // LSTM treba i 23 prethodna sata (prozor od M.window = 24 sata)
      if (k < M.window - 1) return null;
      const seq = [];
      for (let t = k - M.window + 1; t <= k; t++) {
        const xs = vec(f, t, mod);
        if (seqIdx.some(i => !Number.isFinite(xs[i]))) return null;
        seq.push(xs);
      }
      return lstm(seq);
    }
    return { predictAt, ridge, mlp, lstm, vec };
  }

  // ------------------------------------------------------------------
  // 4. DEKODIRANJE UGRAĐENIH PODATAKA
  // Podaci su u HTML-u sačuvani kao cijeli brojevi (int32) da bi zauzimali manje prostora:
  // vrijednost = cijeli broj / skala (npr. temperatura 12,35 °C sa skalom 40 je 494).
  // ints sadrži sve veličine jednu za drugom, svaka dužine N.
  // ------------------------------------------------------------------
  function dequantize(ints, qvars, qscale, N) {
    const raw = {};
    qvars.forEach((c, i) => {
      const a = new Float64Array(N);
      for (let k = 0; k < N; k++) a[k] = ints[i * N + k] / qscale[i];
      raw[c] = a;
    });
    return raw;
  }

  // Ovo je sve što ostatak aplikacije vidi iz ovog modula.
  return { features, buildEngine, dequantize, cosZenith, equationOfTime, dayOfYear, HOUR };
})();

// Omogućava korištenje iste datoteke u Node.js (require("./core.js")), za verify.js.
if (typeof module !== "undefined") module.exports = SOLAR;
