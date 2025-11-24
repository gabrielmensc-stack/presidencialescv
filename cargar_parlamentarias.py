import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd

DB_PATH = Path("data.db")
BASE_ELECCIONES = Path("..") / "bases_datos_ad" / "Elecciones"


def normalizar(txt: str) -> str:
    import unicodedata

    if txt is None:
        return ""
    s = str(txt).upper().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


PARTY_ABBR_OVERRIDES = {
    "PARTIDO SOCIALISTA DE CHILE": "PS",
    "PARTIDO POR LA DEMOCRACIA": "PPD",
    "PARTIDO DEMOCRATA CRISTIANO": "PDC",
    "PARTIDO DEMOCRATA CRISTIANO (CHILE)": "PDC",
    "RENOVACION NACIONAL": "RN",
    "UNION DEMOCRATA INDEPENDIENTE": "UDI",
    "EVOLUCION POLITICA": "EVOPOLI",
    "REVOLUCION DEMOCRATICA": "RD",
    "CONVERGENCIA SOCIAL": "CS",
    "PARTIDO LIBERAL DE CHILE": "PL",
    "PARTIDO COMUNISTA DE CHILE": "PC",
    "PARTIDO REPUBLICANO DE CHILE": "REP",
    "PARTIDO CONSERVADOR CRISTIANO": "PCC",
    "PARTIDO HUMANISTA": "PH",
    "PARTIDO ECOLOGISTA VERDE": "PEV",
    "PARTIDO DE LA GENTE": "PDG",
    "FEDERACION REGIONALISTA VERDE SOCIAL": "FRVS",
    "PARTIDO RADICAL DE CHILE": "PR",
    "PARTIDO REGIONALISTA INDEPENDIENTE DEMOCRATA": "PRI",
    "PARTIDO NACIONAL CIUDADANO": "PNC",
    "CENTRO UNIDO": "CU",
    "CIUDADANOS": "CIU",
    "NUEVO TIEMPO": "NT",
}


def partido_to_code(raw: str) -> str:
    if raw is None:
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    # Si viene como "INDEPENDIENTE PARTIDO X", intenta quedarse con el partido
    norm = normalizar(raw_str)
    if norm.startswith("INDEPENDIENTE "):
        norm = norm.replace("INDEPENDIENTE ", "", 1).strip()
        if not norm:
            return "IND"
    if norm in PARTY_ABBR_OVERRIDES:
        return PARTY_ABBR_OVERRIDES[norm]
    tokens = norm.split()
    if tokens:
        return tokens[-1]
    return norm


def load_coalition_lookup(conn) -> dict:
    try:
        df = pd.read_sql_query("SELECT partido, coalicion FROM coalicion_partido", conn)
        return {normalizar(row["partido"]): row["coalicion"] for _, row in df.iterrows()}
    except Exception:
        return {}


def detect_header_row(path: Path, sheet: int = 0) -> int:
    # Busca fila que contenga la palabra Votos
    df = pd.read_excel(path, sheet_name=sheet, header=None, nrows=12)
    for i, row in df.iterrows():
        if row.isin(["Votos", "VOTOS"]).any():
            return i
    return 0


def process_file(path: Path, tipo: str, anio: int, conn):
    header_row = detect_header_row(path)
    df = pd.read_excel(path, sheet_name=0, header=header_row)
    cargo_map = {"diputados": "DIPUTADO", "senadores": "SENADOR", "alcaldes": "ALCALDE"}
    cargo_ref = cargo_map.get(tipo, tipo.upper())
    df = df[df["Cargo"].astype(str).str.upper().str.contains(cargo_ref, na=False)]

    # Renombrar columnas para alinearlas a resultados_servel
    rename_map = {
        "Nro.Regi\u00f3n": "region_num",
        "Nro Regi\u00f3n": "region_num",
        "Regi\u00f3n": "region",
        "Circunscripci\u00f3n senatorial": "circ_sen",
        "Distrito": "distrito",
        "Comuna": "comuna",
        "Lista": "lista",
        "Pacto": "pacto",
        "Partido": "partido",
        "Nro.voto": "nro_voto",
        "Nro.Voto": "nro_voto",
        "Nombres": "nombres",
        "Primer apellido": "apellido1",
        "Segundo apellido": "apellido2",
        "Votos": "votos",
    }
    df = df.rename(columns=rename_map)
    cols_needed = ["region_num", "region", "circ_sen", "distrito", "comuna", "lista", "pacto", "partido", "nro_voto", "nombres", "apellido1", "apellido2", "votos"]
    for col in cols_needed:
        if col not in df.columns:
            df[col] = None

    df["region_num"] = pd.to_numeric(df["region_num"], errors="coerce").astype("Int64")
    df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0).astype(int)
    df["nro_voto"] = pd.to_numeric(df["nro_voto"], errors="coerce").astype("Int64")

    df["partido_code"] = df["partido"].apply(partido_to_code)
    coal_lookup = load_coalition_lookup(conn)
    df["coalicion"] = df["partido_code"].apply(lambda p: coal_lookup.get(normalizar(p), None))

    df["candidato"] = (
        df[["nombres", "apellido1", "apellido2"]]
        .fillna("")
        .apply(lambda row: " ".join([str(x).strip() for x in row if str(x).strip()]), axis=1)
    )
    df.loc[df["candidato"] == "", "candidato"] = pd.NA

    elect_col = "electo_nominado" if "electo_nominado" in df.columns else None
    electo_series = pd.Series(0, index=df.index)
    if elect_col:
        electo_series = pd.to_numeric(df[elect_col], errors="coerce").fillna(0).astype(int)
    else:
        electo_series = df["Cargo"].astype(str).str.upper().str.contains(cargo_ref, na=False).astype(int)

    registros = df[
        [
            "region_num",
            "region",
            "circ_sen",
            "distrito",
            "comuna",
            "lista",
            "pacto",
            "partido_code",
            "nro_voto",
            "candidato",
            "votos",
        ]
    ].copy()
    registros["anio"] = anio
    registros["tipo"] = tipo
    registros["vuelta"] = 1
    registros["apellido1"] = df["apellido1"].fillna("").astype(str)
    registros["apellido2"] = df["apellido2"].fillna("").astype(str)
    registros["electo"] = electo_series
    registros = registros.rename(columns={"partido_code": "partido"})

    registros = registros[
        [
            "anio",
            "tipo",
            "vuelta",
            "region_num",
            "region",
            "circ_sen",
            "distrito",
            "comuna",
            "lista",
            "pacto",
            "partido",
            "nro_voto",
            "apellido1",
            "apellido2",
            "candidato",
            "votos",
            "electo",
        ]
    ]
    return registros


def load_all(targets: List[Tuple[str, int, str]]):
    if not DB_PATH.exists():
        raise SystemExit("No se encontró data.db en el directorio actual.")
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.execute("PRAGMA busy_timeout=60000;")
    # Asegura columna electo si no existía
    cols = {row[1] for row in conn.execute("PRAGMA table_info(resultados_servel)").fetchall()}
    if "electo" not in cols:
        try:
            conn.execute("ALTER TABLE resultados_servel ADD COLUMN electo INTEGER")
        except Exception:
            pass
    try:
        for rel_path, anio, tipo in targets:
            file_path = BASE_ELECCIONES / rel_path
            if not file_path.exists():
                print(f"Archivo no encontrado: {file_path}")
                continue
            print(f"Ingresando {tipo} {anio} desde {file_path.name}")
            registros = process_file(file_path, tipo, anio, conn)
            conn.execute("DELETE FROM resultados_servel WHERE tipo=? AND anio=?", (tipo, anio))
            registros.to_sql("resultados_servel", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    objetivos = [
        ("diputados/2017_11_Diputados_Datos_Eleccion/2017_11_Diputados_Datos_Eleccion.xlsx", 2017, "diputados"),
        ("diputados/2021_11_Diputados_Datos_Eleccion/2021_11_Diputados_Datos_Eleccion.xlsx", 2021, "diputados"),
        ("senadores/2017_11_Senatorial_Datos_Eleccion/2017_11_Senatorial_Datos_Eleccion.xlsx", 2017, "senadores"),
        ("alcaldes/2016_10_Alcaldes_DatosEleccion/2016_10_Alcaldes_DatosEleccion.xlsx", 2016, "alcaldes"),
    ]
    load_all(objetivos)
