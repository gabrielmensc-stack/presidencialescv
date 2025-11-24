import json
import os
import sqlite3
from pathlib import Path
import base64
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk

DB_PATH = "data.db"
BASE_DIR = Path(__file__).parent
GEO_REG = BASE_DIR / "geodata" / "regiones_chile.geojson"
GEO_COM = BASE_DIR / "geodata" / "comunas_chile.geojson"
PHOTO_DIR = BASE_DIR / "static" / "candidatos"
PARQUET_DIR = BASE_DIR / ".cache"
PARQUET_FILE = PARQUET_DIR / "v_presidencial_coalicion.parquet"

PALETTE = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#dd3497", "#636363", "#e0a500", "#16a085"]
ISLANDS = {"ISLA DE PASCUA", "JUAN FERNANDEZ"}
PHOTO_ALIASES = {
    "MARCO ENRIQUEZ OMINAMI": ["meo", "enriquez ominami"],
    "MARCO ENRIQUEZ-OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "MARCO ANTONIO ENRIQUEZ-OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "MARCO ANTONIO ENRIQUEZ OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "MARCO ENRIQUEZ-OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "FRANCO PARISI FERNANDEZ": ["parisi"],
    "EVELYN MATTHEI FORNET": ["matthei"],
    "JOSE ANTONIO KAST RIST": ["kast", "jak", "jose antonio kast"],
    "JOSE ANTONIO KAST": ["kast", "jak"],
    "JEANNETTE JARA ROMAN": ["jara"],
    "HAROLD MAYNE NICHOLLS SECUL": ["maynenichols", "mayne nicholls", "mayne nicholls secul"],
    "HAROLD MAYNE NICHOLLS": ["maynenichols", "mayne nicholls"],
    "HAROLD MAYNE-NICHOLLS": ["maynenichols", "mayne nicholls"],
    "HAROLD MAYNE-NICHOLLS SECUL": ["maynenichols", "mayne nicholls"],
    "JOHANNES KAISER BARENTS VON HONHAGEN": ["kaiser"],
    "JOHANNES KAISER BARENTS VON HOHENHAGEN": ["kaiser", "johannes kaiser"],
    "JOHANNES HOHENHAGEN": ["kaiser", "johannes kaiser"],
    "EDUARDO ANTONIO ARTES BRICHETTI": ["artes"],
    "SEBASTIAN PINERA ECHENIQUE": ["pinera"],
    "RICARDO LAGOS ESCOBAR": ["lagos"],
    "HAROLD SECUL": ["harold mayne nicholls", "secul", "maynenichols", "mayne nicholls", "mayne nicholls secul"],
}
REGION_CENTER_FALLBACK = {
    "Arica y Parinacota": (-69.98, -18.48),
    "Tarapaca": (-69.33, -20.21),
    "Antofagasta": (-69.83, -23.65),
    "Atacama": (-70.57, -27.37),
    "Coquimbo": (-70.98, -29.90),
    "Valparaiso": (-71.62, -33.04),
    "Metropolitana de Santiago": (-70.66, -33.45),
    "Libertador General Bernardo O'Higgins": (-71.17, -34.58),
    "Maule": (-71.32, -35.43),
    "Nuble": (-72.10, -36.62),
    "Biobio": (-72.76, -37.47),
    "La Araucania": (-72.67, -38.74),
    "Los Rios": (-72.97, -39.86),
    "Los Lagos": (-72.93, -41.47),
    "Aysen del General Carlos Ibanez del Campo": (-72.31, -45.58),
    "Magallanes y de la Antartica Chilena": (-72.77, -53.16),
}

CHAMBER_TOTAL_SEATS = 155
SENATE_TOTAL_SEATS = 50
SENATE_SEATS_EN_JUEGO_2021 = {
    # Supuestos basados en los cupos que estuvieron en disputa en 2021
    "Antofagasta": 3,
    "Coquimbo": 3,
    "Libertador General Bernardo O'Higgins": 3,
    "Biobio": 3,
    "Los Lagos": 3,
    "Magallanes y de la Antartica Chilena": 2,
    "Metropolitana de Santiago": 5,
    "Los Rios": 2,
    "Nuble": 3,
}
# Si luego se cargan los cupos que no iban a elección, usar aquí el desglose por coalición
SENATE_HOLDOVER_COALITIONS = {}

COALITION_ALIASES = {
    "CHILE PODEMOS": "Chile Podemos +",
    "CHILE PODEMOS +": "Chile Podemos +",
    "CHILE PODEMOS MAS": "Chile Podemos +",
    "CHILE VAMOS": "Chile Vamos",
    "APRUEBO DIGNIDAD": "Apruebo Dignidad",
    "NUEVO PACTO SOCIAL": "Nuevo Pacto Social",
    "DIGNIDAD AHORA": "Dignidad Ahora",
    "FRENTE SOCIAL CRISTIANO": "Frente Social Cristiano",
    "PARTIDO DE LA GENTE": "Partido de la Gente",
    "INDEPENDIENTES UNIDOS": "Independientes Unidos",
    "INDEPENDIENTES": "Independientes",
    "UNION PATRIOTICA": "Union Patriotica",
    "PARTIDO ECOLOGISTA VERDE": "Partido Ecologista Verde",
}

COALITION_COLORS = {
    "Chile Podemos +": "#3182bd",
    "Chile Vamos": "#3182bd",
    "Nuevo Pacto Social": "#e6550d",
    "Apruebo Dignidad": "#31a354",
    "Partido de la Gente": "#e0a500",
    "Frente Social Cristiano": "#756bb1",
    "Independientes": "#636363",
    "Independientes Unidos": "#636363",
    "Dignidad Ahora": "#16a085",
    "Partido Ecologista Verde": "#0ea95c",
    "Union Patriotica": "#dd3497",
}

COALITION_SHORT = {
    "Chile Podemos +": "Chile Podemos",
    "Chile Vamos": "Chile Vamos",
    "Nuevo Pacto Social": "N. Pacto Social",
    "Apruebo Dignidad": "Apruebo Dignidad",
    "Partido de la Gente": "PDG",
    "Frente Social Cristiano": "F. Social Crist.",
    "Independientes Unidos": "Ind. Unidos",
    "Independientes": "Independientes",
    "Dignidad Ahora": "Dignidad Ahora",
    "Partido Ecologista Verde": "PEV",
    "Union Patriotica": "UPA",
}


def ensure_sample_db():
    """Create a small sample DB if the expected view is missing."""
    db_path = Path(DB_PATH)
    sample_csv = BASE_DIR / "sample_data" / "v_presidencial_coalicion.csv"
    has_view = False
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name='v_presidencial_coalicion'")
            has_view = cur.fetchone() is not None
        except Exception:
            has_view = False
        finally:
            try:
                conn.close()
            except Exception:
                pass
    if has_view or not sample_csv.exists():
        return False


def ensure_views():
    """Ensure required views exist on the real database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_comuna_region AS
                SELECT DISTINCT
                    comuna,
                    region,
                    region_num
                FROM resultados_servel
                WHERE comuna IS NOT NULL
                  AND region IS NOT NULL;

            CREATE VIEW IF NOT EXISTS v_presidencial_comuna AS
                SELECT
                    anio,
                    vuelta,
                    'TRICEL' AS fuente,
                    region_num,
                    region,
                    comuna_num,
                    comuna,
                    candidato,
                    partido,
                    votos,
                    votos_nulos,
                    votos_blancos,
                    total_votos,
                    inscritos
                FROM resultados_tricel
                UNION ALL
                SELECT
                    anio,
                    vuelta,
                    'SERVEL' AS fuente,
                    region_num,
                    region,
                    NULL       AS comuna_num,
                    comuna,
                    candidato,
                    partido,
                    votos,
                    NULL       AS votos_nulos,
                    NULL       AS votos_blancos,
                    NULL       AS total_votos,
                    NULL       AS inscritos
                FROM resultados_servel
                WHERE tipo = 'presidencial'
                UNION ALL
                SELECT
                    s.anio,
                    s.vuelta,
                    'SERVEL_2025' AS fuente,
                    cr.region_num,
                    cr.region,
                    NULL       AS comuna_num,
                    s.comuna,
                    s.candidato,
                    NULL       AS partido,
                    s.votos,
                    NULL       AS votos_nulos,
                    NULL       AS votos_blancos,
                    NULL       AS total_votos,
                    NULL       AS inscritos
                FROM servel_presid_2025_comuna s
                LEFT JOIN v_comuna_region cr
                       ON s.comuna = cr.comuna;

            CREATE VIEW IF NOT EXISTS v_presidencial_coalicion AS
                SELECT
                    v.anio,
                    v.vuelta,
                    v.fuente,
                    v.region_num,
                    v.region,
                    v.comuna_num,
                    v.comuna,
                    v.candidato,
                    v.partido,
                    cp.coalicion,
                    v.votos,
                    v.votos_nulos,
                    v.votos_blancos,
                    v.total_votos,
                    v.inscritos
                FROM v_presidencial_comuna v
                LEFT JOIN coalicion_partido cp
                  ON v.partido = cp.partido
                 AND v.anio BETWEEN cp.anio_inicio AND cp.anio_fin;
            """
        )
        create_indexes(conn)
        conn.close()
        return True
    except Exception as e:
        st.error(f"No se pudieron crear las vistas requeridas: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return False
    try:
        df_sample = pd.read_csv(sample_csv)
        conn = sqlite3.connect(db_path)
        df_sample.to_sql("v_presidencial_coalicion", conn, if_exists="replace", index=False)
        conn.close()
        return True
    except Exception as e:
        st.error(f"No se pudo crear base de datos de ejemplo: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return False

def create_indexes(conn):
    """Crear índices usados en filtros para acelerar consultas."""
    existing_tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    index_statements = []
    if "resultados_tricel" in existing_tables:
        index_statements.extend(
            [
                "CREATE INDEX IF NOT EXISTS idx_tricel_anio_region ON resultados_tricel(anio, region)",
                "CREATE INDEX IF NOT EXISTS idx_tricel_comuna ON resultados_tricel(comuna)",
                "CREATE INDEX IF NOT EXISTS idx_tricel_candidato ON resultados_tricel(candidato)",
            ]
        )
    if "resultados_servel" in existing_tables:
        index_statements.extend(
            [
                "CREATE INDEX IF NOT EXISTS idx_servel_tipo_anio ON resultados_servel(tipo, anio)",
                "CREATE INDEX IF NOT EXISTS idx_servel_region ON resultados_servel(region)",
                "CREATE INDEX IF NOT EXISTS idx_servel_comuna ON resultados_servel(comuna)",
                "CREATE INDEX IF NOT EXISTS idx_servel_candidato ON resultados_servel(candidato)",
            ]
        )
    if "servel_presid_2025_comuna" in existing_tables:
        index_statements.extend(
            [
                "CREATE INDEX IF NOT EXISTS idx_servel_2025_anio ON servel_presid_2025_comuna(anio)",
                "CREATE INDEX IF NOT EXISTS idx_servel_2025_comuna ON servel_presid_2025_comuna(comuna)",
                "CREATE INDEX IF NOT EXISTS idx_servel_2025_candidato ON servel_presid_2025_comuna(candidato)",
            ]
        )
    for stmt in index_statements:
        try:
            conn.execute(stmt)
        except Exception:
            # Si la tabla faltara o la DB está bloqueada, seguimos sin bloquear la app
            pass
    conn.commit()


def normalizar(txt: str) -> str:
    import unicodedata

    if txt is None:
        return ""
    s = str(txt).upper().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())



def normalizar_region(nombre: str) -> str:
    n = normalizar(nombre)
    if n.startswith("DEL "):
        n = n[4:]
    if n.startswith("DE "):
        n = n[3:]
    if "VALPAR" in n:
        return "Valparaiso"
    if "NUBLE" in n or "ÑUBLE" in n or "�UBLE" in n:
        return "Nuble"
    mapping = {
        "REGION DE TARAPACA": "Tarapaca",
        "TARAPACA": "Tarapaca",
        "REGION DE ANTOFAGASTA": "Antofagasta",
        "ANTOFAGASTA": "Antofagasta",
        "REGION DE ATACAMA": "Atacama",
        "ATACAMA": "Atacama",
        "REGION DE COQUIMBO": "Coquimbo",
        "COQUIMBO": "Coquimbo",
        "REGION DE VALPARAISO": "Valparaiso",
        "VALPARAISO": "Valparaiso",
        "VALPARAISO REGION": "Valparaiso",
        "REGION DE VALPARAISO REGION": "Valparaiso",
        "REGION METROPOLITANA DE SANTIAGO": "Metropolitana de Santiago",
        "METROPOLITANA DE SANTIAGO": "Metropolitana de Santiago",
        "LIBERTADOR GENERAL BERNARDO O HIGGINS": "Libertador General Bernardo O'Higgins",
        "REGION DEL LIBERTADOR BERNARDO O HIGGINS": "Libertador General Bernardo O'Higgins",
        "REGION DEL LIBERTADOR GENERAL BERNARDO O HIGGINS": "Libertador General Bernardo O'Higgins",
        "REGION DEL MAULE": "Maule",
        "MAULE": "Maule",
        "REGION DEL BIO BIO": "Biobio",
        "REGION DEL BIO-BIO": "Biobio",
        "BIOBIO": "Biobio",
        "REGION DE LA ARAUCANIA": "La Araucania",
        "LA ARAUCANIA": "La Araucania",
        "REGION DE LOS RIOS": "Los Rios",
        "LOS RIOS": "Los Rios",
        "REGION DE LOS LAGOS": "Los Lagos",
        "LOS LAGOS": "Los Lagos",
        "REGION DE AYSEN DEL GRAL.IBANEZ DEL CAMPO": "Aysen del General Carlos Ibanez del Campo",
        "REGION DE AYSEN DEL GRAL IBANEZ DEL CAMPO": "Aysen del General Carlos Ibanez del Campo",
        "AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO": "Aysen del General Carlos Ibanez del Campo",
        "REGION DE MAGALLANES Y ANTARTICA CHILENA": "Magallanes y de la Antartica Chilena",
        "MAGALLANES Y DE LA ANTARTICA CHILENA": "Magallanes y de la Antartica Chilena",
        "REGION DE ARICA Y PARINACOTA": "Arica y Parinacota",
        "ARICA Y PARINACOTA": "Arica y Parinacota",
    }
    return mapping.get(n, " ".join(w.capitalize() for w in n.split()))


PARTY_ABBR_OVERRIDES = {
    "PARTIDO SOCIALISTA DE CHILE": "PS",
    "PARTIDO POR LA DEMOCRACIA": "PPD",
    "PARTIDO DEMOCRATA CRISTIANO": "PDC",
    "PARTIDO DEMOCRATA CRISTIANO (CHILE)": "PDC",
    "RENOVACION NACIONAL": "RN",
    "UNION DEMOCRATA INDEPENDIENTE": "UDI",
    "UNI�'N DEM�'CRATA INDEPENDIENTE": "UDI",
    "EVOLUCION POLITICA": "EVOPOLI",
    "EVOLUCI�'N POL�'TICA": "EVOPOLI",
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
    "FEDERACI�'N REGIONALISTA VERDE SOCIAL": "FRVS",
    "PARTIDO RADICAL DE CHILE": "PR",
    "PARTIDO REGIONALISTA INDEPENDIENTE DEMOCRATA": "PRI",
    "PARTIDO NACIONAL CIUDADANO": "PNC",
    "CENTRO UNIDO": "CU",
}


def partido_to_code(partido: str) -> str:
    key = normalizar(partido)
    if not key:
        return ""
    if key in PARTY_ABBR_OVERRIDES:
        return PARTY_ABBR_OVERRIDES[key]
    tokens = key.split()
    if tokens:
        return tokens[-1]
    return key


def coalicion_from_fields(pacto: str, partido: str, coalition_lookup: dict) -> str:
    pacto_norm = normalizar(pacto)
    if pacto_norm:
        for alias_norm, label in COALITION_ALIASES.items():
            if pacto_norm.startswith(alias_norm):
                return label
        return pacto.strip()
    code = partido_to_code(partido)
    if code and code in coalition_lookup:
        return coalition_lookup[code]
    if code == "IND":
        return "Independientes"
    return pacto or partido or ""


@st.cache_data
def cargar_datos():
    """Carga datos desde Parquet cacheado si está vigente; de lo contrario lee de SQLite y refresca el cache."""
    db_mtime = Path(DB_PATH).stat().st_mtime if Path(DB_PATH).exists() else 0
    pq_mtime = PARQUET_FILE.stat().st_mtime if PARQUET_FILE.exists() else 0

    @st.cache_data(show_spinner=False)
    def _read_parquet(path_str: str):
        return pd.read_parquet(path_str)

    @st.cache_data(show_spinner=False)
    def _read_sql(db_path: str):
        conn = sqlite3.connect(db_path)
        df_sql = pd.read_sql_query("SELECT * FROM v_presidencial_coalicion", conn)
        conn.close()
        return df_sql

    df = pd.DataFrame()
    # Usar Parquet si es más nuevo que la base de datos
    if PARQUET_FILE.exists() and pq_mtime >= db_mtime:
        try:
            df = _read_parquet(str(PARQUET_FILE))
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        try:
            df = _read_sql(DB_PATH)
            if not df.empty:
                try:
                    PARQUET_DIR.mkdir(exist_ok=True)
                    df.to_parquet(PARQUET_FILE, index=False)
                except Exception:
                    # Si no se puede escribir cache, no interrumpimos la carga
                    pass
        except Exception as e:
            st.error(f"No se pudo leer la vista: {e}")
            return pd.DataFrame()

    if df.empty:
        return df
    df["anio"] = df["anio"].astype(int)
    df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0).astype(int)
    for col in ["region", "comuna", "candidato", "partido", "fuente", "coalicion"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["region"] = df["region"].apply(normalizar_region)
    df["region_norm"] = df["region"].apply(normalizar_region)
    df["comuna_norm"] = df["comuna"].apply(normalizar)
    df["candidato_norm"] = df["candidato"].apply(normalizar)
    return df


@st.cache_data(show_spinner=False)
def cargar_parlamentarias():
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            anio,
            tipo,
            vuelta,
            region,
            circ_sen,
            distrito,
            comuna,
            pacto,
            lista,
            partido,
            candidato,
            votos
        FROM resultados_servel
        WHERE tipo IN ('senadores','diputados')
        """,
        conn,
    )
    coalition_lookup = {}
    try:
        df_map = pd.read_sql_query("SELECT partido, coalicion FROM coalicion_partido", conn)
        coalition_lookup = {row["partido"]: row["coalicion"] for _, row in df_map.iterrows()}
    except Exception:
        coalition_lookup = {}
    conn.close()
    if df.empty:
        return df
    df["anio"] = df["anio"].astype(int)
    df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0).astype(int)
    for col in ["region", "comuna", "pacto", "lista", "partido", "candidato"]:
        df[col] = df[col].astype(str).fillna("").str.strip()
        df.loc[df[col].str.lower() == "nan", col] = ""
    df["region_norm"] = df["region"].apply(normalizar_region)
    df["coalicion"] = df.apply(lambda r: coalicion_from_fields(r["pacto"], r["partido"], coalition_lookup), axis=1)
    df["coalicion_norm"] = df["coalicion"].apply(normalizar)
    df["comuna_norm"] = df["comuna"].apply(normalizar)
    return df


def dhondt_allocation(votes_by_coalition: dict, seats: int):
    allocation = {c: 0 for c in votes_by_coalition}
    if seats <= 0 or not votes_by_coalition:
        return allocation
    for _ in range(seats):
        quotients = []
        for coal, votos in votes_by_coalition.items():
            divisor = allocation.get(coal, 0) + 1
            quotients.append((votos / divisor, coal))
        if not quotients:
            break
        _, winner = max(quotients, key=lambda x: (x[0], x[1]))
        allocation[winner] = allocation.get(winner, 0) + 1
    return allocation


def asignar_escanos_por_zona(df: pd.DataFrame, zona_col: str, cupos_map: dict):
    registros = []
    for zona, chunk in df.groupby(zona_col):
        cupos = cupos_map.get(normalizar(zona), cupos_map.get(normalizar_region(zona), 0))
        if not cupos:
            continue
        votos_coal = chunk.groupby("coalicion", as_index=False)["votos"].sum()
        votos_dict = {row["coalicion"]: row["votos"] for _, row in votos_coal.iterrows() if row["votos"] > 0}
        asignacion = dhondt_allocation(votos_dict, cupos)
        for coal, seats in asignacion.items():
            if seats:
                registros.append({"zona": zona, "coalicion": coal, "escanos": seats, "votos": votos_dict.get(coal, 0)})
    if not registros:
        return pd.DataFrame(columns=["zona", "coalicion", "escanos", "votos"])
    return pd.DataFrame(registros)


def format_compact_votes(votes: int) -> str:
    if votes >= 1_000_000:
        return f"{votes/1_000_000:.1f} M"
    if votes >= 1_000:
        return f"{votes/1_000:.1f} K"
    return f"{votes}"


def build_seat_grid(filled: dict, total_slots: int, palette: dict, outlined: dict | None = None):
    seats = []
    palette_order = {name: idx for idx, name in enumerate(palette.keys())}
    sorted_coal = sorted(filled.items(), key=lambda x: (palette_order.get(x[0], 999), -x[1], x[0]))
    outlined = outlined or {}
    used_coals = set()
    for coal, qty in sorted_coal:
        color = palette.get(coal, "#7d8697")
        for _ in range(qty):
            seats.append({"coalicion": coal, "color": color, "outline": False})
        if outlined.get(coal):
            for _ in range(outlined[coal]):
                seats.append({"coalicion": coal, "color": color, "outline": True})
            used_coals.add(coal)
    # Outlines de coaliciones que no tenían escaños nuevos
    for coal, qty in outlined.items():
        if coal in used_coals:
            continue
        color = palette.get(coal, "#7d8697")
        for _ in range(qty):
            seats.append({"coalicion": coal, "color": color, "outline": True})
    while len(seats) < total_slots:
        seats.append({"coalicion": None, "color": "#7d8697", "outline": True})
    return seats[:total_slots]


def render_seat_grid(seats, title: str, election_year: int | None = None):
    st.markdown(f"#### {title}")
    rows = 4
    cols = max(1, (len(seats) + rows - 1) // rows)
    padded = list(seats)
    while len(padded) < rows * cols:
        padded.append({"coalicion": None, "color": "#7d8697", "outline": True})
    html_parts = [f"<div class='seat-grid-table' style='display:grid;grid-template-columns:repeat({cols}, 18px);grid-auto-rows:18px;gap:6px;'>"]
    for seat in padded:
        if seat["outline"]:
            style = f"border:2px solid {seat['color']};background:transparent;"
        else:
            style = f"background:{seat['color']};border:2px solid {seat['color']};"
        html_parts.append(f"<span class='seat' style='{style}' title='{seat.get('coalicion','')}'></span>")
    html_parts.append("</div>")
    if election_year:
        legend = f"""
        <div style='display:flex;gap:16px;align-items:center;margin-top:8px;color:#cdd6e5;font-size:12px;'>
          <span style='display:flex;align-items:center;gap:6px;'>
            <span class='seat' style='width:14px;height:14px;border-radius:50%;background:#6b7280;border:2px solid #6b7280;'></span>
            Electo en {election_year}
          </span>
          <span style='display:flex;align-items:center;gap:6px;'>
            <span class='seat' style='width:14px;height:14px;border-radius:50%;background:transparent;border:2px solid #a8b1c2;'></span>
            No fue a eleccion (periodo previo)
          </span>
        </div>
        """
        html_parts.append(legend)
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def render_seat_semicircle(seats, title: str, election_year: int | None = None, legend_html: str | None = None):
    st.markdown(f"#### {title}")
    total = len(seats)
    rows = 8 if total > 120 else 7 if total > 90 else 6 if total > 60 else 5 if total > 30 else 4
    dot_size = 16
    gap = 12  # separación extra entre círculos
    inner_radius = 24
    outer_radius = inner_radius + (rows - 1) * (dot_size + gap)
    width = 2 * (outer_radius + dot_size)
    height = outer_radius + dot_size * 2
    cx = width / 2
    cy = outer_radius + dot_size

    radii = [outer_radius - i * (dot_size + gap) for i in range(rows)]
    weight_sum = sum(radii)
    base_counts = [max(3, round(total * r / weight_sum)) for r in radii]
    diff = total - sum(base_counts)
    idx_diff = 0
    while diff != 0:
        base_counts[idx_diff % rows] += 1 if diff > 0 else -1
        diff = total - sum(base_counts)
        idx_diff += 1

    positions = []
    for r, seats_in_row in zip(radii, base_counts):
        if seats_in_row <= 0:
            continue
        steps = seats_in_row - 1 if seats_in_row > 1 else 1
        angles = [math.pi - (math.pi * i / steps) for i in range(seats_in_row)]
        for ang in angles:
            x = cx + r * math.cos(ang)
            y = cy - r * math.sin(ang)
            positions.append((ang, -r, x, y))  # sort by angle, then outer to inner

    positions.sort(key=lambda t: (t[0], t[1]))
    html_parts = [f"<div style='position:relative;width:{width}px;height:{height}px;margin-bottom:14px;'>"]
    for seat, (_, _, x, y) in zip(seats, positions):
        if seat["outline"]:
            style = f"border:2px solid {seat['color']};background:transparent;"
        else:
            style = f"background:{seat['color']};border:2px solid {seat['color']};"
        html_parts.append(
            f"<span class='seat' style='position:absolute;left:{x}px;top:{y}px;transform:translate(-50%,-50%);{style}' title='{seat.get('coalicion','')}'></span>"
        )
    html_parts.append("</div>")
    if legend_html:
        html_parts.append(legend_html.strip())
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def render_coalition_summary(seat_counts: dict, df_votes: pd.DataFrame, palette: dict):
    if not seat_counts:
        st.info("Sin asignación de escaños para mostrar.")
        return
    data = []
    for coal, seats in seat_counts.items():
        votos = int(df_votes[df_votes["coalicion"] == coal]["votos"].sum())
        top_parties = (
            df_votes[df_votes["coalicion"] == coal]
            .groupby("partido", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
        )
        partidos_txt = ", ".join(top_parties.head(5)["partido"].tolist())
        display = COALITION_SHORT.get(coal, coal)
        if len(display) > 18:
            tokens = display.split()
            display = " ".join(tokens[:2]) if len(tokens) >= 2 else display[:18]
        data.append({"coalicion": coal, "display": display, "escanos": seats, "votos": votos, "partidos": partidos_txt})
    data = sorted(data, key=lambda x: (-x["escanos"], -x["votos"]))
    cols_per_row = 3
    for i in range(0, len(data), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, data[i : i + cols_per_row]):
            color = palette.get(item["coalicion"], "#7d8697")
            col.markdown(
                f"""
                <div style="padding:8px 10px;border-radius:12px;background:#111a29;border:1px solid #1f2b3e;">
                  <div style="font-size:22px;font-weight:800;color:{color};">{item['escanos']}</div>
                  <div style="font-weight:700;margin-top:-6px;display:flex;align-items:center;gap:8px;">
                    <span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:{color};border:2px solid {color};"></span>
                    {item['display']}
                  </div>
                  <div style="color:#9fb3ce;font-size:13px;margin:4px 0 6px 0;">{format_compact_votes(item['votos'])} votos</div>
                  <div style="color:#ccd6e6;font-size:12px;line-height:1.3;">{item['partidos']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _distribute_outlines(seats_by_coal: dict, missing: int):
    if missing <= 0 or not seats_by_coal:
        return {}
    total = sum(seats_by_coal.values()) or 1
    provisional = {coal: max(0, round(missing * val / total)) for coal, val in seats_by_coal.items()}
    diff = missing - sum(provisional.values())
    ordered = sorted(seats_by_coal.items(), key=lambda x: -x[1])
    idx = 0
    while diff != 0 and ordered:
        coal = ordered[idx % len(ordered)][0]
        provisional[coal] = provisional.get(coal, 0) + (1 if diff > 0 else -1)
        diff = missing - sum(provisional.values())
        idx += 1
    return {k: v for k, v in provisional.items() if v > 0}


def build_legend(fill_label: str, outline_label: str | None = None, fill_color: str = "#4b5563", outline_color: str = "#9ca3af") -> str:
    parts = [
        "<div style='display:flex;gap:16px;align-items:center;margin-top:-4px;color:#cdd6e5;font-size:12px;'>",
        f"<span style='display:flex;align-items:center;gap:6px;'><span class='seat' style='width:14px;height:14px;border-radius:50%;background:{fill_color};border:2px solid {fill_color};'></span>{fill_label}</span>",
    ]
    if outline_label:
        parts.append(
            f"<span style='display:flex;align-items:center;gap:6px;'><span class='seat' style='width:14px;height:14px;border-radius:50%;background:transparent;border:2px solid {outline_color};'></span>{outline_label}</span>"
        )
    parts.append("</div>")
    return "\n".join(parts)

def _distribute_outlines(seats_by_coal: dict, missing: int):
    if missing <= 0 or not seats_by_coal:
        return {}
    total = sum(seats_by_coal.values()) or 1
    provisional = {coal: max(0, round(missing * val / total)) for coal, val in seats_by_coal.items()}
    diff = missing - sum(provisional.values())
    ordered = sorted(seats_by_coal.items(), key=lambda x: -x[1])
    idx = 0
    while diff != 0 and ordered:
        coal = ordered[idx % len(ordered)][0]
        provisional[coal] = provisional.get(coal, 0) + (1 if diff > 0 else -1)
        diff = missing - sum(provisional.values())
        idx += 1
    return {k: v for k, v in provisional.items() if v > 0}




def ganador_por_geo(df, geo_col):
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby([geo_col, "candidato"], as_index=False)["votos"].sum()
    idx = agg.groupby(geo_col)["votos"].idxmax()
    return agg.loc[idx]


def assign_color(name, cmap):
    if name not in cmap:
        cmap[name] = PALETTE[len(cmap) % len(PALETTE)]
    return cmap[name]


def build_palette(df: pd.DataFrame):
    """Palette estable para el conjunto filtrado (anio/vuelta) y consistente mapa/informe."""
    if df.empty or "candidato" not in df.columns:
        return {}, {}
    ranking = (
        df.groupby("candidato_norm", as_index=False)["votos"]
        .sum()
        .sort_values(["votos", "candidato_norm"], ascending=[False, True])
    )
    palette_norm = {row["candidato_norm"]: PALETTE[i % len(PALETTE)] for i, row in ranking.iterrows()}
    palette_orig = {}
    for cand in df["candidato"].dropna().unique():
        c_norm = normalizar(cand)
        if c_norm in palette_norm:
            palette_orig[cand] = palette_norm[c_norm]
    return palette_orig, palette_norm


@st.cache_data(show_spinner=False)
def build_photo_map():
    photo_map = {}
    if PHOTO_DIR.exists():
        for path in PHOTO_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                stem_norm = normalizar(path.stem)
                if stem_norm not in photo_map:  # keep first found
                    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
                    ext = path.suffix.lstrip(".") or "png"
                    photo_map[stem_norm] = {"path": str(path), "b64": b64, "ext": ext}
    return photo_map


def get_photo_path(name_norm: str, photo_map: dict):
    # Intentos: nombre + apellido, dos últimas palabras, solo apellido, solo nombre
    tokens = name_norm.split()
    keys = [name_norm]
    if len(tokens) >= 2:
        keys.append(" ".join(tokens[:2]))
        keys.append(" ".join(tokens[-2:]))
        keys.append("".join(tokens[-2:]))  # pegado
    if tokens:
        keys.append(tokens[-1])
        keys.append(tokens[0])
    # Aliases conocidos
    aliases = PHOTO_ALIASES.get(name_norm, [])
    keys.extend([normalizar(a) for a in aliases])
    for k in keys:
        if k in photo_map:
            return photo_map[k]
    # Búsqueda parcial por inclusión
    for k, info in photo_map.items():
        if k in name_norm or name_norm in k:
            return info
    return None


def short_name(full_name: str) -> str:
    overrides_raw = {
        "JOSE ANTONIO KAST RIST": "Jose Kast",
        "JOSE ANTONIO KAST": "Jose Kast",
        "MARCO ENRIQUEZ-OMINAMI": "Marco Enriquez-Ominami",
        "MARCO ENRIQUEZ OMINAMI": "Marco Enriquez-Ominami",
        "MARCO ENRIQUEZ OMINAMI GUMUCIO": "Marco Enriquez-Ominami",
        "MARCO ANTONIO ENRIQUEZ OMINAMI GUMUCIO": "Marco Enriquez-Ominami",
        "MARCO ANTONIO ENRIQUEZ-OMINAMI GUMUCIO": "Marco Enriquez-Ominami",
        "MARCO ENRIQUEZ-OMINAMI GUMUCIO": "Marco Enriquez-Ominami",
        "MARCO GUMUCIO": "Marco Enriquez-Ominami",
        "HAROLD MAYNE NICHOLLS SECUL": "Harold Mayne-Nicholls",
        "HAROLD MAYNE NICHOLLS": "Harold Mayne-Nicholls",
        "HAROLD SECUL": "Harold Mayne-Nicholls",
        "JOHANNES KAISER BARENTS VON HONHAGEN": "Johannes Kaiser",
        "JOHANNES KAISER BARENTS VON HOHENHAGEN": "Johannes Kaiser",
        "JOHANNES HOHENHAGEN": "Johannes Kaiser",
        "EDUARDO ANTONIO ARTES BRICHETTI": "Eduardo Artes",
        "EVELYN MATTHEI FORNET": "Evelyn Matthei",
        "JEANNETTE JARA ROMAN": "Jeanette Jara",
        "JEANETTE JARA ROMAN": "Jeanette Jara",
        "FRANCO PARISI FERNANDEZ": "Franco Parisi",
    }
    overrides = {normalizar(k): v for k, v in overrides_raw.items()}
    key = normalizar(full_name)
    if key in overrides:
        return overrides[key]
    parts = full_name.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"
    return full_name


def img_to_data_uri(path: str, size: int = 64) -> str:
    try:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = Path(path).suffix.lstrip(".") or "png"
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ""


def render_ranking_cards(ranking: pd.DataFrame, photo_map: dict, palette_local: dict, title: str = "Ranking"):
    if ranking.empty:
        st.info("Sin datos para esta selección.")
        return
    max_votes = ranking["votos"].max() or 1
    st.markdown(f"#### {title}")
    for _, r in ranking.iterrows():
        col_img, col_bar = st.columns([1, 5])
        img_path = get_photo_path(r["candidato_norm"], photo_map)
        color = palette_local.get(r["candidato"], "#3182bd")
        if img_path:
            col_img.markdown(
                f"<div style='width:56px;height:56px;border-radius:50%;overflow:hidden;border:3px solid {color};'>"
                f"<img src='data:image/{img_path['ext']};base64,{img_path['b64']}' style='width:100%;height:100%;object-fit:cover;' />"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            col_img.markdown("&nbsp;")
        pct_width = max(8, min(100, (r["votos"] / max_votes) * 100))
        color = palette_local.get(r["candidato"], "#3182bd")
        # Texto fuera si la barra es pequeña
        if pct_width < 35:
            bar_html = f"""
            <div style='display:flex;align-items:center;gap:8px;'>
              <div style='background:{color};width:{pct_width}%;min-width:8%;border-radius:10px;padding:6px 10px;'>&nbsp;</div>
              <div style='color:#e8edf5;font-weight:600; white-space: nowrap;'>
                {short_name(r['candidato'])} — {r['votos']:,} votos
              </div>
            </div>
            """
        else:
            bar_html = f"""
            <div style='display:flex;align-items:center;'>
              <div style='background:{color};width:{pct_width}%;min-width:8%;border-radius:10px;padding:6px 10px;color:#0e1624;font-weight:600; white-space: nowrap;'>
                {short_name(r['candidato'])} — {r['votos']:,} votos
              </div>
            </div>
            """
        col_bar.markdown(bar_html, unsafe_allow_html=True)


def hex_to_rgba(hex_color: str, alpha: int = 200):
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


@st.cache_resource(show_spinner=False)
def load_geojsons():
    """Carga GeoJSON una sola vez por sesión."""
    reg_geo = json.load(open(GEO_REG, "r", encoding="utf-8"))
    com_geo = json.load(open(GEO_COM, "r", encoding="utf-8"))
    return reg_geo, com_geo


def _geometry_bbox(features):
    mins = [180, 90]
    maxs = [-180, -90]
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        stack = list(coords) if isinstance(coords, list) else []
        while stack:
            item = stack.pop()
            if not item:
                continue
            if isinstance(item[0], (float, int)):
                lon, lat = item[:2]
                mins[0] = min(mins[0], lon)
                mins[1] = min(mins[1], lat)
                maxs[0] = max(maxs[0], lon)
                maxs[1] = max(maxs[1], lat)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    if mins[0] == 180:
        return (-71, -35), mins, maxs
    center = ((mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2)
    return center, mins, maxs



def preparar_capas(df, reg_geo, com_geo, region_sel_norm=None, palette_local=None, palette_norm=None):
    cmap = dict(palette_local or {})
    cmap_norm = dict(palette_norm or {})
    reg_win = ganador_por_geo(df, "region_norm")
    reg_lookup = {row["region_norm"]: (row["candidato"], row["votos"]) for _, row in reg_win.iterrows()}
    reg_features = []
    valpo_override = None
    override_path = BASE_DIR / "geodata" / "valparaiso_from_shp.geojson"
    if override_path.exists():
        try:
            valpo_geo = json.load(open(override_path, "r", encoding="utf-8"))
            if valpo_geo.get("features"):
                valpo_override = valpo_geo["features"][0]
        except Exception:
            valpo_override = None

    for feat in reg_geo["features"]:
        props = feat.get("properties", {})
        name_norm = normalizar_region(props.get("Region", ""))
        props["color"] = hex_to_rgba("#c0c0c0")
        props["Region_display"] = props.get("Region", "")

        if name_norm == "Valparaiso" and valpo_override:
            feat = valpo_override
            props = feat.get("properties", {})
            props["Region"] = "Valparaiso"
            props["color"] = hex_to_rgba("#c0c0c0")
            props["Region_display"] = props.get("Region", "")

        cand, votos = reg_lookup.get(name_norm, (None, None))
        if cand:
            col = cmap.get(cand) or cmap_norm.get(normalizar(cand)) or assign_color(cand, cmap)
            props["winner"] = cand
            props["votos"] = int(votos)
            props["color"] = hex_to_rgba(col)

        if name_norm == "Aysen del General Carlos Ibanez del Campo":
            props["Region_display"] = "Aysen"

        feat["properties"] = props
        reg_features.append(feat)

    layers = []
    if not region_sel_norm:
        layers.append({"id": "regiones", "features": reg_features})
        center, _, _ = _geometry_bbox(reg_features) if reg_features else ((-71, -35), None, None)
        if center == (-71, -35) and reg_features:
            lons = []
            lats = []
            for f in reg_features:
                rn = normalizar_region(f.get("properties", {}).get("Region", ""))
                if rn in REGION_CENTER_FALLBACK:
                    lon, lat = REGION_CENTER_FALLBACK[rn]
                    lons.append(lon)
                    lats.append(lat)
            if lons and lats:
                center = (sum(lons) / len(lons), sum(lats) / len(lats))
    else:
        df_region = df[df["region_norm"] == region_sel_norm]
        com_win = ganador_por_geo(df_region, "comuna_norm")
        com_lookup = {row["comuna_norm"]: (row["candidato"], row["votos"]) for _, row in com_win.iterrows()}
        com_features = []
        for feat in com_geo["features"]:
            props = feat.get("properties", {})
            if normalizar_region(props.get("Region", "")) != region_sel_norm:
                continue
            name_norm = normalizar(props.get("Comuna", ""))
            if name_norm in ISLANDS:
                continue
            props["color"] = hex_to_rgba("#c0c0c0")
            props["Region_display"] = props.get("Region", "")
            cand, votos = com_lookup.get(name_norm, (None, None))
            if cand:
                col = cmap.get(cand) or cmap_norm.get(normalizar(cand)) or assign_color(cand, cmap)
                props["winner"] = cand
                props["votos"] = int(votos)
                props["color"] = hex_to_rgba(col)
            feat["properties"] = props
            com_features.append(feat)
        layers.append({"id": "comunas", "features": com_features})
        center, _, _ = _geometry_bbox(com_features) if com_features else ((-71, -35), None, None)
        if center == (-71, -35):
            center = REGION_CENTER_FALLBACK.get(region_sel_norm, center)
    return layers, center

def render_mapa(layers, center, region_sel_norm=None):
    default_center = (-70.6548, -33.4366)  # Santiago por defecto
    if region_sel_norm:
        cx, cy = center if center else default_center
        zoom_level = 6.5
    else:
        cx, cy = default_center
        zoom_level = 3.9  # más abierto para ver todo Chile
    view = pdk.ViewState(latitude=cy, longitude=cx, zoom=zoom_level, pitch=0, bearing=0)
    deck_layers = []
    for lyr in layers:
        deck_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                {"type": "FeatureCollection", "features": lyr["features"]},
                id=lyr["id"],
                opacity=0.85,
                stroked=True,
                filled=True,
                get_fill_color="[properties.color[0], properties.color[1], properties.color[2], properties.color[3]]",
                get_line_color=[30, 40, 60, 220],
                lineWidthUnits="pixels",
                lineWidthScale=1,
                lineWidthMinPixels=1.2,
                pickable=True,
                auto_highlight=True,
            )
        )
    tooltip = {
        "html": "<b>Region:</b> {Region_display}{Region}<br/><b>Comuna:</b> {Comuna}<br/><b>Ganador:</b> {winner}<br/><b>Votos:</b> {votos}",
        "style": {"backgroundColor": "#0e1624", "color": "white"},
    }
    deck = pdk.Deck(layers=deck_layers, initial_view_state=view, map_style=None, tooltip=tooltip)
    st.pydeck_chart(deck, width="stretch", key=f"deck_{region_sel_norm or 'all'}")


def barra_total(ranking, palette_local):
    if ranking.empty:
        return
    total = ranking["votos"].sum()
    ranking = ranking.copy()
    ranking["percent"] = ranking["votos"] / total * 100 if total else 0
    ranking["color"] = ranking["candidato"].map(palette_local).fillna("#c0c0c0")
    fig = go.Figure()
    for _, r in ranking.iterrows():
        fig.add_bar(
            x=[r["percent"]],
            y=["Total"],
            orientation="h",
            marker_color=r["color"],
            hovertemplate=f"{r['candidato']}<br>{r['percent']:.1f}%<br>{r['votos']:,} votos<br>",
            text=f"{r['percent']:.1f}% ({r['votos']:,})",
            textposition="inside",
            textfont={"color": "#111111", "size": 12},
            name=str(r["candidato"]),
            showlegend=False,
        )
    fig.update_layout(barmode="stack", xaxis={"title": "100%", "range": [0, 100]}, height=180, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")



def main():
    st.set_page_config(page_title="Resultados Presidenciales (Kepler)", layout="wide")
    st.markdown(
        """
        <style>
        body, .main, .stApp {
            background: #0e1624;
            color: #e8edf5;
        }
        h1, h2, h3, h4, h5 { color: #f2f6ff; }
        .stMetric, .stButton>button, .stPlotlyChart {
            border-radius: 14px !important;
        }
        .stMetric {
            background: #1a2536;
            padding: 10px 14px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.16);
            color: #f2f6ff;
        }
        .keplergl-widget { border-radius: 14px !important; overflow: hidden; box-shadow: 0 10px 24px rgba(0,0,0,0.25); }
        /* Hover suave */
        .stButton>button:hover, .stDataFrame:hover, .stPlotlyChart:hover {
            box-shadow: 0 10px 24px rgba(0,0,0,0.2);
            transform: translateY(-1px);
            transition: all 180ms ease;
        }
        .seat { width:16px; height:16px; border-radius:50%; display:inline-block; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Logo + titulo alineados
    logo_path = BASE_DIR / "static" / "Logo.png"
    col_logo, col_title = st.columns([0.2, 1])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=60)
    with col_title:
        st.title("Resultados Presidenciales (Kepler)")

    sample_loaded = ensure_sample_db()
    if sample_loaded:
        st.info("Se cargaron datos de ejemplo desde sample_data/v_presidencial_coalicion.csv porque no se encontro data.db con la vista requerida.")
    else:
        views_ok = ensure_views()
        if views_ok:
            st.caption("Vistas creadas/verificadas en data.db.")

    df_pres = cargar_datos()
    df_parl = cargar_parlamentarias()
    tipo_eleccion = st.sidebar.selectbox("Eleccion", ["Presidencial", "Senadores", "Diputados", "Alcaldes"])
    vista_datos = st.sidebar.selectbox("Datos a mostrar", ["Resultados", "Participacion"])

    if tipo_eleccion == "Presidencial":
        if df_pres.empty:
            st.warning("No hay datos en v_presidencial_coalicion.")
            return
        anios = sorted(df_pres["anio"].unique())
        anio_sel = st.sidebar.selectbox("Anio", anios, index=anios.index(max(anios)))
        df = df_pres[df_pres["anio"] == anio_sel]
        vueltas_disp = sorted(v for v in df["vuelta"].dropna().unique())
        if vueltas_disp:
            vuelta_default_idx = len(vueltas_disp) - 1
            vuelta_sel = st.sidebar.selectbox("Vuelta", vueltas_disp, index=vuelta_default_idx)
            df = df[df["vuelta"] == vuelta_sel]
        palette_year, palette_year_norm = build_palette(df)

        if vista_datos == "Participacion":
            st.info("Vista de participacion en construccion; se mostraran resultados mientras tanto.")

        regiones_disp = sorted({normalizar_region(r) for r in df["region"].dropna().unique()})
        region_sel = st.sidebar.selectbox("Region (opcional para ver comunas)", ["(todas)"] + regiones_disp, index=0, key="sel_region")
        region_sel_norm = None if region_sel == "(todas)" else normalizar_region(region_sel)
        comuna_sel_norm = None
        if region_sel_norm:
            comunas_disp = sorted({normalizar(c) for c in df[df["region_norm"] == region_sel_norm]["comuna"].dropna().unique()})
            comuna_sel = st.sidebar.selectbox("Comuna (opcional)", ["(todas)"] + comunas_disp, index=0, key="sel_comuna")
            comuna_sel_norm = None if comuna_sel == "(todas)" else normalizar(comuna_sel)

        if not GEO_REG.exists() or not GEO_COM.exists():
            st.error("GeoJSON no encontrados en geodata/")
            return
        try:
            reg_geo, com_geo = load_geojsons()
        except Exception as e:
            st.error(f"No se pudieron cargar GeoJSON: {e}")
            return

        col_info, col_map = st.columns([1, 1.2])
        with col_map:
            st.markdown("##### Selecciona o haz clic en la lista para centrar")
            layers, center = preparar_capas(df, reg_geo, com_geo, region_sel_norm, palette_year, palette_year_norm)
            render_mapa(layers, center, region_sel_norm)

        with col_info:
            df_sel = df[df["region_norm"] == region_sel_norm] if region_sel_norm else df
            if comuna_sel_norm:
                df_sel = df_sel[df_sel["comuna_norm"] == comuna_sel_norm]
            ranking = (
                df_sel.groupby("candidato", as_index=False)["votos"]
                .sum()
                .sort_values("votos", ascending=False)
            )
            ranking["candidato_norm"] = ranking["candidato"].apply(normalizar)
            palette_local = {}
            for i, row in ranking.iterrows():
                c = row["candidato"]
                c_norm = row["candidato_norm"]
                color = palette_year.get(c) or palette_year_norm.get(c_norm) or PALETTE[i % len(PALETTE)]
                palette_local[c] = color
            photo_map = build_photo_map()

            st.markdown("### Ganadores")
            top2 = ranking.head(2)
            cols = st.columns(min(2, len(top2)))
            for col, (_, row) in zip(cols, top2.iterrows()):
                img_path = get_photo_path(row["candidato_norm"], photo_map)
                if img_path:
                    col.markdown(
                        f"<div style='width:110px;height:110px;border-radius:50%;overflow:hidden;border:4px solid {palette_local.get(row['candidato'], '#3182bd')};'>"
                        f"<img src='data:image/{img_path['ext']};base64,{img_path['b64']}' style='width:100%;height:100%;object-fit:cover;' />"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                col.metric(short_name(row["candidato"]), f"{row['votos']:,} votos")

            st.markdown("### Barra total (100%)")
            barra_total(ranking, palette_local)
            st.markdown("### Ranking completo")
            render_ranking_cards(ranking, photo_map, palette_local, title="Ranking")
        return

    df_parl_tipo = df_parl[df_parl["tipo"] == tipo_eleccion.lower()]
    if df_parl_tipo.empty:
        st.warning("Aun no hay datos cargados para esta eleccion.")
        return
    anios = sorted(df_parl_tipo["anio"].unique())
    anio_sel = st.sidebar.selectbox("Anio", anios, index=anios.index(max(anios)))
    df_parl_tipo = df_parl_tipo[df_parl_tipo["anio"] == anio_sel]
    if vista_datos == "Participacion":
        st.info("Vista de participacion en construccion; se mostraran resultados mientras tanto.")

    palette_coal = dict(COALITION_COLORS)
    extras = [c for c in df_parl_tipo["coalicion"].unique() if c and c not in palette_coal]
    for i, coal in enumerate(sorted(extras)):
        palette_coal[coal] = PALETTE[i % len(PALETTE)]

    if tipo_eleccion == "Senadores":
        if anio_sel == 2021:
            cupos_map = {normalizar_region(k): v for k, v in SENATE_SEATS_EN_JUEGO_2021.items()}
            asignacion = asignar_escanos_por_zona(df_parl_tipo, "region_norm", cupos_map)
            seats_by_coal = asignacion.groupby("coalicion")["escanos"].sum().to_dict()
            missing = max(0, SENATE_TOTAL_SEATS - sum(seats_by_coal.values()))
            outlines = _distribute_outlines(seats_by_coal, missing)
            seats = build_seat_grid(seats_by_coal, SENATE_TOTAL_SEATS, palette_coal, outlined=outlines)
            prev_year = anio_sel - 4 if anio_sel else None
            legend_html = build_legend(f"Electo en {anio_sel}", f"Electo el año {prev_year}")
            render_seat_semicircle(seats, f"{SENATE_TOTAL_SEATS} Senadores", election_year=anio_sel, legend_html=legend_html)
            render_coalition_summary(seats_by_coal, df_parl_tipo, palette_coal)
            st.caption("Los escaños no sometidos a elección se muestran delineados con el color de su coalición estimada.")
        else:
            votos_coal = df_parl_tipo.groupby("coalicion", as_index=False)["votos"].sum()
            seats_by_coal = dhondt_allocation({row["coalicion"]: row["votos"] for _, row in votos_coal.iterrows() if row["votos"] > 0}, SENATE_TOTAL_SEATS)
            seats = build_seat_grid(seats_by_coal, SENATE_TOTAL_SEATS, palette_coal)
            legend_html = build_legend(f"Electo en {anio_sel}")
            render_seat_semicircle(seats, f"{SENATE_TOTAL_SEATS} Senadores", election_year=anio_sel, legend_html=legend_html)
            render_coalition_summary(seats_by_coal, df_parl_tipo, palette_coal)
            st.caption("Distribucion aproximada con D'Hondt nacional al no contar con cupos por circunscripcion para este anio.")
    elif tipo_eleccion == "Diputados":
        votos_coal = df_parl_tipo.groupby("coalicion", as_index=False)["votos"].sum()
        seats_by_coal = dhondt_allocation({row["coalicion"]: row["votos"] for _, row in votos_coal.iterrows() if row["votos"] > 0}, CHAMBER_TOTAL_SEATS)
        seats = build_seat_grid(seats_by_coal, CHAMBER_TOTAL_SEATS, palette_coal)
        legend_html = build_legend(f"Electo en {anio_sel}")
        render_seat_semicircle(seats, f"{CHAMBER_TOTAL_SEATS} Diputados", election_year=anio_sel, legend_html=legend_html)
        render_coalition_summary(seats_by_coal, df_parl_tipo, palette_coal)
        st.caption("Distribucion aproximada usando D'Hondt nacional mientras no haya detalle por distrito.")
    else:
        winners = (
            df_parl_tipo.sort_values("votos", ascending=False)
            .groupby("comuna", as_index=False)
            .first()
        )
        seats_by_coal = winners.groupby("coalicion")["comuna"].count().to_dict()
        total_alcaldias = sum(seats_by_coal.values())
        seats = build_seat_grid(seats_by_coal, total_alcaldias, palette_coal)
        render_seat_semicircle(seats, f"{total_alcaldias} Alcaldias", election_year=anio_sel)
        render_coalition_summary(seats_by_coal, df_parl_tipo, palette_coal)


if __name__ == "__main__":
    main()
