# ☀️ Solmetar - Neuronske mreže za procjenu proizvodnje solarne energije
### Praktično programsko rješenje uz završni rad
**Tema rada:** Neuronske mreže za procjenu proizvodnje solarne energije na osnovu vremenskih podataka
**Autor:** Radović Milica (broj indeksa 1764/25)
**Mentor:** Prof. dr Omerović Maid
**Univerzitet u Travniku, Fakultet za tehničke studije, Inženjerska informatika** · Travnik, septembar 2026.

---

## 📌 Pregled projekta
Projekat pokazuje cijeli postupak procjene satne proizvodnje fotonaponskih (solarnih) elektrana iz vremenskih podataka pomoću neuronskih mreža:

1. **Podaci:** izmjerena proizvodnja svih solarnih elektrana u belgijskoj provinciji Limburg (Elia Open Data, 1.1.2021. - 21.9.2026.) i satni vremenski podaci Open-Meteo za Limburg i 7 gradova u BiH.
2. **Modeli:** višeslojni perceptron (MLP) i LSTM mreža (PyTorch), upoređeni sa persistencijom, linearnom regresijom, gradijentnim pojačanjem i zvaničnom prognozom operatora Elia.
3. **Provjera:** hronološka podjela podataka, pet treniranja svake mreže, Diebold-Mariano test, važnost osobina i ablacija.
4. **Primjena u BiH:** model naučen u Belgiji primijenjen na Travnik, Sarajevo, Mostar, Banju Luku, Tuzlu, Bihać i Trebinje i provjeren procjenom PVGIS.
5. **Aplikacija Solmetar:** jedna HTML datoteka koja izvršava sve modele direktno u pregledaču, bez instalacije.

Kompletan rad je u folderu [`rad/`](rad/).

---

## 🎯 Ključne karakteristike

| Parametar / osobina | Specifikacija u radu | Implementacija u kodu |
| :--- | :--- | :--- |
| **Ciljna veličina** | faktor iskorištenja κ = P / P<sub>nom</sub> (udio instalirane snage) | `prep.load_limburg()` |
| **Ulazi** | 18 osobina nezavisnih od lokacije (zračenje, vrijeme, položaj Sunca, susjedni sati) | `prep.FEATURES` |
| **Solarna geometrija** | Cooper (deklinacija), Spencer (jednačina vremena), pravo solarno vrijeme | `prep.geometry()` |
| **Podjela podataka** | trening do 30.6.2024., validacija do 30.6.2025., test 1.7.2025. - 21.9.2026. | `prep.split()` |
| **MLP** | 18 → 256 → 128 → 64 → 1, ReLU, dropout 0,2 | **46.081 parametar** (`models.MLP`) |
| **LSTM** | 2 sloja × 32 ćelije, prozor od 24 sata | **15.809 parametara** (`models.LSTMReg`) |
| **Učenje** | Adam, MSE gubitak, grupa od 256 uzoraka, rano zaustavljanje, L2 = 10<sup>-5</sup> | `models.train_nn()` |
| **Ponovljivost** | 5 sjemena generatora slučajnih brojeva po mreži | `train.py` |
| **Statistička provjera** | Diebold-Mariano test, Newey-West varijansa, 24 pomjeraja | `metrics.diebold_mariano()` |
| **Aplikacija** | isti modeli u JavaScriptu, odstupanje od Pythona < 3·10<sup>-7</sup> | `app/core.js`, `app/verify.js` |

---

## 📊 Rezultati (testni period, 5.940 dnevnih sati)

| Model | MAE (%) | RMSE (%) | R² | Indeks uspješnosti (%) |
| :--- | ---: | ---: | ---: | ---: |
| Persistencija (juče u isto vrijeme) | 7,26 | 11,54 | 0,612 | 0,0 |
| Linearna regresija | 4,30 | 6,14 | 0,890 | 46,8 |
| Gradijentno pojačanje | 3,78 | 5,95 | 0,897 | 48,4 |
| **MLP** | **3,89** | **5,69** | **0,906** | **50,7** |
| **LSTM** | **3,65** | **5,59** | **0,909** | **51,6** |
| *Elia, prognoza dan unaprijed* | *3,08* | *4,77* | *0,934* | *58,7* |

Greške su u procentima instalirane snage. Za 7 gradova u BiH MLP procjenjuje godišnju proizvodnju sa prosječnim odstupanjem od **4,3 %** u odnosu na PVGIS.

**Hipoteze:** H1 potvrđena (mreže značajno tačnije od jednostavnijih modela), H2 nije potvrđena (LSTM nije pouzdano bolji od MLP, p = 0,28), H3 potvrđena za MLP (sva odstupanja < 10 %).

---

## 📐 Matematička formulacija u kodu

**Faktor iskorištenja i procjena snage:**
$$\kappa_t = \frac{P_t}{P_{nom,t}}, \qquad \hat{P}_t = \hat{\kappa}_t \cdot P_{nom}$$

**Zenitni ugao Sunca i indeks prozračnosti:**
$$\cos\theta_z = \sin\varphi\sin\delta + \cos\varphi\cos\delta\cos\omega, \qquad k_t = \frac{G}{G_0}$$

**MLP (prolaz unaprijed):**
$$\mathbf{a}^{(l)} = \max\left(0,\ \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \qquad \hat{\kappa} = \mathbf{W}^{(L)}\mathbf{a}^{(L-1)} + b^{(L)}$$

**LSTM ćelija:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, \qquad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**Metrike:**
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\kappa_i - \hat{\kappa}_i)^2}, \qquad \text{SS} = 100\left(1 - \frac{\text{RMSE}}{\text{RMSE}_{pers}}\right)$$

Sve formule (32) sa objašnjenjima nalaze se u radu.

---

## 📁 Struktura repozitorija

```
Solmetar/
├── rad/
│   ├── Zavrsni_rad_Radovic_Milica.pdf     <- Završni rad (PDF, konačna verzija)
│   └── Zavrsni_rad_Radovic_Milica.docx    <- Završni rad (Word, konačna verzija)
├── app/
│   ├── solmetar.html        <- Gotova aplikacija (otvara se dvostrukim klikom)
│   ├── core.js              <- Računsko jezgro: osobine, linearna regresija, MLP i LSTM u JavaScriptu
│   ├── app.js               <- Korisnički interfejs: izbor lokacije, dana i snage, grafikoni
│   ├── head.html, body_main.html, logo.svg   <- Izgled i struktura stranice
│   ├── build_app.py         <- Spaja sve dijelove u jednu HTML datoteku
│   └── verify.js            <- Provjera: JavaScript naspram Pythona
├── solar_nn/
│   ├── prep.py              <- Učitavanje podataka, solarna geometrija, 18 osobina, podjela
│   ├── data.py              <- Standardizacija i 24-satni prozori za LSTM
│   ├── models.py            <- MLP, LSTM i petlja treniranja (PyTorch)
│   ├── metrics.py           <- MAE, RMSE, MBE, R² i Diebold-Mariano test
│   ├── train.py             <- Kompletan eksperiment: pretraga, treniranje, evaluacija
│   ├── transfer.py          <- Primjena na 7 gradova u BiH i poređenje sa PVGIS
│   ├── export_app.py        <- Izvoz modela i podataka za aplikaciju
│   ├── figures.py           <- Izrada svih slika iz rada
│   └── numbers.py           <- Ispis brojeva korištenih u tekstu rada
├── data/v2/
│   ├── limburg_pv.csv       <- Proizvodnja i prognoze za Limburg (Elia, satne vrijednosti)
│   ├── meteo_all.csv.gz     <- Vremenski podaci Open-Meteo (Limburg i 7 gradova u BiH)
│   └── bih_pvgis.csv        <- PVGIS procjene za gradove u BiH
├── results/                 <- Istrenirani modeli (mlp.pt, lstm.pt, baselines.pkl), metrike, procjene
├── figures/                 <- Sve slike iz rada (web/ = slike preuzete sa interneta, sa izvorima u radu)
├── tests/
│   └── test_solar_nn.py     <- Automatizovani testovi (14 provjera)
├── requirements.txt         <- Potrebne Python biblioteke
└── README.md                <- Ova dokumentacija
```

---

## 🚀 Uputstvo za pokretanje

### 1. Aplikacija Solmetar (bez instalacije)
Preuzmite `app/solmetar.html` i otvorite je dvostrukim klikom u bilo kojem savremenom pregledaču (Chrome, Edge, Firefox, Safari).

- Za dane od 1.1.2021. do 23.9.2026. internet nije potreban.
- Za kasnije dane i prognozu za 7 dana aplikacija preuzima podatke sa Open-Meteo.

Na interfejsu možete:
- izabrati lokaciju (Limburg ili grad u BiH), instaliranu snagu (kWp ili MWp) i dan
- vidjeti satnu procjenu i dnevnu energiju za linearnu regresiju, MLP i LSTM
- za Limburg uporediti procjenu sa izmjerenom proizvodnjom i prognozom operatora
- isprobati scenario "šta ako" (zračenje, temperatura, oblačnost)
- izračunati godišnju proizvodnju i pokrenuti evaluaciju svih testnih sati u pregledaču

### 2. Ponavljanje eksperimenta (Python 3.10 ili noviji)
U terminalu, iz korijenskog foldera repozitorija:
```bash
pip install -r requirements.txt
python -m solar_nn.train        # kompletan eksperiment (~35 min na procesoru), upisuje results/
python -m solar_nn.transfer     # primjena na gradove u BiH i poređenje sa PVGIS
python -m solar_nn.figures      # slike u figures/
```
Istrenirani modeli su već u folderu `results/`, pa za aplikaciju i testove treniranje nije potrebno.

### 3. Ponovna izgradnja aplikacije
```bash
python -m solar_nn.export_app   # izvoz modela i podataka
python app/build_app.py         # spajanje u app/solmetar.html
node app/verify.js              # provjera JavaScript implementacije (potreban Node.js)
```

---

## 🧪 Automatizovani testovi
```bash
python -m unittest -v tests.test_solar_nn
```
Očekivani izlaz:
```text
Ran 14 tests in ~5s

OK
```
Testovi provjeravaju:
1. Deklinaciju Sunca (Cooper)
2. Jednačinu vremena (Spencer)
3. Solarnu geometriju za Travnik (podnevna elevacija oko 69° 21. juna)
4. Broj ulaznih osobina (18 za MLP, 15 po satu za LSTM)
5. Indeks prozračnosti k<sub>t</sub> u intervalu [0; 1,2]
6. Poravnanje vremenskih podataka sa satom proizvodnje
7. Opseg faktora iskorištenja κ
8. Hronološku podjelu bez preklapanja
9. Broj parametara: **MLP 46.081, LSTM 15.809**
10. Učitavanje sačuvanih modela i procjene u intervalu [0, 1]
11. Metrike MAE, RMSE, MBE i R²
12. Diebold-Mariano test
13. Podudarnost sačuvanih rezultata sa Tabelom 5 u radu
14. Podudarnost aplikacije Solmetar sa Python implementacijom

---

## 📚 Izvori podataka i licence
- **Elia Open Data**, skup ods032: proizvodnja fotonaponskih elektrana u Belgiji - https://opendata.elia.be/explore/dataset/ods032/
- **Open-Meteo** (Zippenfenig, 2023), licenca CC BY 4.0 - https://open-meteo.com
- **PVGIS 5.3**, Evropska komisija, JRC - https://re.jrc.ec.europa.eu/pvg_tools/
- Slike u `figures/web/`: Our World in Data (CC BY), Global Solar Atlas / Svjetska banka i Solargis (CC BY 4.0), Wikimedia Commons (CC BY-SA 3.0 i 4.0); tačni izvori navedeni su ispod slika u radu.
