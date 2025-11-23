import json
import os
import sqlite3
from pathlib import Path
import base64

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

PALETTE = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#dd3497", "#636363", "#e0a500", "#16a085"]
ISLANDS = {"ISLA DE PASCUA", "JUAN FERNANDEZ"}
PHOTO_ALIASES = {
    "MARCO ENRIQUEZ OMINAMI": ["meo", "enriquez ominami"],
    "MARCO ENRIQUEZ OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "MARCO ANTONIO ENRIQUEZ OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "MARCO ENRIQUEZ OMINAMI GUMUCIO": ["meo", "enriquez ominami"],
    "FRANCO PARISI FERNANDEZ": ["parisi"],
    "EVELYN MATTHEI FORNET": ["matthei"],
    "JOSE ANTONIO KAST RIST": ["kast", "jak", "jose antonio kast"],
    "JOSE ANTONIO KAST": ["kast", "jak"],
    "JEANNETTE JARA ROMAN": ["jara"],
    "HAROLD MAYNE NICHOLLS SECUL": ["maynenichols", "mayne nicholls", "mayne nicholls secul"],
    "HAROLD MAYNE NICHOLLS": ["maynenichols", "mayne nicholls"],
    "JOHANNES KAISER BARENTS VON HONHAGEN": ["kaiser"],
    "EDUARDO ANTONIO ARTES BRICHETTI": ["artes"],
    "SEBASTIAN PINERA ECHENIQUE": ["pinera"],
    "RICARDO LAGOS ESCOBAR": ["lagos"],
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


@st.cache_data
def cargar_datos():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM v_presidencial_coalicion", conn)
        conn.close()
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
    overrides = {
        "JOSE ANTONIO KAST RIST": "José Kast",
        "JOSE ANTONIO KAST": "José Kast",
        "MARCO ENRIQUEZ OMINAMI": "Marco Enríquez-Ominami",
        "MARCO ANTONIO ENRIQUEZ OMINAMI GUMUCIO": "Marco Enríquez-Ominami",
        "MARCO ENRIQUEZ OMINAMI GUMUCIO": "Marco Enríquez-Ominami",
        "HAROLD MAYNE NICHOLLS SECUL": "Harold Mayne-Nicholls",
        "HAROLD MAYNE NICHOLLS": "Harold Mayne-Nicholls",
        "JOHANNES KAISER BARENTS VON HONHAGEN": "Johannes Kaiser",
        "EDUARDO ANTONIO ARTES BRICHETTI": "Eduardo Artés",
        "EVELYN MATTHEI FORNET": "Evelyn Matthei",
        "JEANNETTE JARA ROMAN": "Jeanette Jara",
        "JEANETTE JARA ROMAN": "Jeanette Jara",
        "FRANCO PARISI FERNANDEZ": "Franco Parisi",
    }
    key = full_name.upper()
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



def preparar_capas(df, reg_geo, com_geo, region_sel_norm=None):
    cmap = {}
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
            col = assign_color(cand, cmap)
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
                col = assign_color(cand, cmap)
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
                get_line_color=[255, 255, 255, 220],
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Logo + título alineados
    logo_path = BASE_DIR / "static" / "Logo.png"
    col_logo, col_title = st.columns([0.2, 1])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=60)
    with col_title:
        st.title("Resultados Presidenciales (Kepler)")

    sample_loaded = ensure_sample_db()
    if sample_loaded:
        st.info("Se cargaron datos de ejemplo desde sample_data/v_presidencial_coalicion.csv porque no se encontró data.db con la vista requerida.")
    else:
        views_ok = ensure_views()
        if views_ok:
            st.caption("Vistas creadas/verificadas en data.db.")

    df = cargar_datos()
    if df.empty:
        st.warning("No hay datos en v_presidencial_coalicion.")
        return

    # Filtros principales
    tipo_eleccion = st.sidebar.selectbox("Elección", ["Presidencial", "Senadores", "Diputados"])
    vista_datos = st.sidebar.selectbox("Datos a mostrar", ["Resultados", "Participación"])
    anios = sorted(df["anio"].unique())
    anio_sel = st.sidebar.selectbox("Año", anios, index=anios.index(max(anios)))
    df = df[df["anio"] == anio_sel]

    # Por ahora sólo tenemos presidenciales cargadas
    if tipo_eleccion != "Presidencial":
        st.warning("Por ahora sólo hay datos de presidenciales; los demás tipos se agregarán en pasos siguientes.")
    # Participación aún sin implementar
    if vista_datos == "Participación":
        st.info("Vista de participación en construcción; se mostrarán resultados mientras tanto.")

    regiones_disp = sorted({normalizar_region(r) for r in df["region"].dropna().unique()})
    region_sel = st.sidebar.selectbox("Región (opcional para ver comunas)", ["(todas)"] + regiones_disp, index=0, key="sel_region")
    region_sel_norm = None if region_sel == "(todas)" else normalizar_region(region_sel)
    comuna_sel_norm = None
    if region_sel_norm:
        comunas_disp = sorted({normalizar(c) for c in df[df["region_norm"] == region_sel_norm]["comuna"].dropna().unique()})
        comuna_sel = st.sidebar.selectbox("Comuna (opcional)", ["(todas)"] + comunas_disp, index=0, key="sel_comuna")
        comuna_sel_norm = None if comuna_sel == "(todas)" else normalizar(comuna_sel)

    if not GEO_REG.exists() or not GEO_COM.exists():
        st.error("GeoJSON no encontrados en geodata/")
        return
    reg_geo = json.load(open(GEO_REG, "r", encoding="utf-8"))
    com_geo = json.load(open(GEO_COM, "r", encoding="utf-8"))

    # Layout: panel izquierda, mapa derecha
    col_info, col_map = st.columns([1, 1.2])

    with col_map:
        st.markdown("##### Selecciona o haz clic en la lista para centrar")
        # Nota: PyDeck en Streamlit no expone el click de mapa, así que usamos el selector como trigger de centrado
        layers, center = preparar_capas(df, reg_geo, com_geo, region_sel_norm)
        render_mapa(layers, center, region_sel_norm)

    with col_info:
        # Panel resumen (simple): top candidatos total o por selección
        df_sel = df[df["region_norm"] == region_sel_norm] if region_sel_norm else df
        if comuna_sel_norm:
            df_sel = df_sel[df_sel["comuna_norm"] == comuna_sel_norm]
        ranking = (
            df_sel.groupby("candidato", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
        )
        ranking["candidato_norm"] = ranking["candidato"].apply(normalizar)
        palette_local = {row["candidato"]: PALETTE[i % len(PALETTE)] for i, row in ranking.iterrows()}
        photo_map = build_photo_map()

        st.markdown("### Ganadores")
        top2 = ranking.head(2)
        cols = st.columns(min(2, len(top2)))
        for col, (_, row) in zip(cols, top2.iterrows()):
            img_path = get_photo_path(row["candidato_norm"], photo_map)
            if img_path:
                col.markdown(
                    f"<div style='width:110px;height:110px;border-radius:50%;overflow:hidden;border:4px solid {palette_local.get(row['candidato'], '#3182bd')};'>"
                    f"<img src='data:image/{img_path['ext']};base64,{img_path['b64']}' "
                    f"style='width:100%;height:100%;object-fit:cover;' />"
                    "</div>",
                    unsafe_allow_html=True,
                )
            col.metric(short_name(row["candidato"]), f"{row['votos']:,} votos")

        st.markdown("### Barra total (100%)")
        barra_total(ranking, palette_local)

        # Ranking con fotos al lado
        st.markdown("### Ranking completo")
        render_ranking_cards(ranking, photo_map, palette_local, title="Ranking")


if __name__ == "__main__":
    main()
