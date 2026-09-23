"""
build_app.py - SASTAVLJANJE APLIKACIJE SOLMETAR U JEDNU HTML DATOTEKU

Aplikacija se razvija u više manjih datoteka, a korisniku se isporučuje jedna datoteka
(solmetar.html) koja radi bez servera i bez instalacije. Ova skripta ih spaja ovim redom:

    head.html        stilovi (CSS): boje, raspored, izgled kartica i grafikona
    body_main.html   HTML struktura: zaglavlje sa logom, kartice, polja, dugmad, prazni grafikoni
    core.js          računsko jezgro: osobine i modeli (SOLAR)
    MODEL, REF       težine modela i podaci (results/app_model.json) i Python procjene
                     za provjeru (results/python_reference.json), kao JavaScript konstante
    app.js           korisnički interfejs: povezuje kontrole, jezgro i grafikone

Nastaju dvije datoteke:
    solmetar_artifact.html  samo sadržaj stranice (za objavu kao Claude artefakt)
    solmetar.html           potpuna HTML stranica sa ikonom, za otvaranje u pregledaču

Pokretanje (iz korijenskog foldera projekta):
    python -m solar_nn.export_app     # prvo izvoz modela i podataka
    python app/build_app.py           # zatim sastavljanje aplikacije
"""
import base64
from pathlib import Path

A = Path(__file__).resolve().parent          # folder app/
R = A.parent / "results"                     # folder results/ sa izvezenim modelima


def read(p):
    return p.read_text(encoding="utf-8")


head = read(A / "head.html")
main = read(A / "body_main.html")
core = read(A / "core.js")
app = read(A / "app.js")
model = read(R / "app_model.json")
ref = read(R / "python_reference.json")

# Redoslijed je važan: core.js i podaci moraju biti učitani prije app.js, koji ih koristi.
body = (head + main
        + "\n<script>\n" + core + "\n</script>\n"
        + "<script>\nconst MODEL = " + model + ";\nconst REF = " + ref + ";\n</script>\n"
        + "<script>\n" + app + "\n</script>\n")
(A / "solmetar_artifact.html").write_text(body, encoding="utf-8")

# Potpuna stranica: logo kao ikona kartice pregledača (ugrađen kao base64, bez posebne datoteke).
icon = base64.b64encode((A / "logo.svg").read_bytes()).decode()
full = ('<!DOCTYPE html>\n<html lang="bs">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,' + icon + '">\n'
        '<style>[hidden]{display:none!important} body{margin:0}</style>\n</head>\n<body>\n'
        + body + '\n</body>\n</html>\n')
(A / "solmetar.html").write_text(full, encoding="utf-8")
print("solmetar.html:", round(len(full) / 1e6, 2), "MB")
