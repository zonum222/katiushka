import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io, base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="GeoVisualizador de Paillaco", layout="wide")
st.title("🌎 GeoVisualizador de Paillaco")
st.write("Aplicación desarrollada con Streamlit. Seleccione las capas que desea visualizar.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# PASO 1: Paletas cartográficas
# ─────────────────────────────────────────────────────────────

# Geología: colores inspirados en SERNAGEOMIN
PALETA_GEOLOGIA = [
    "#F5C842", "#A8D5A2", "#7EC8C8", "#D4A0F0", "#F5A06E",
    "#6EB5E0", "#E07070", "#A0C878", "#F0D080", "#88BBDD",
    "#CC99BB", "#88CC88", "#DDAA66", "#66AACC", "#CC7766",
    "#AABB55", "#88AABB", "#DDBB88", "#99CCAA", "#BB99CC",
]

# Transporte: jerarquía vial estándar (TIPO → color+grosor)
ESTILOS_TIPO_TRANSPORTE = {
    "autopista":        {"color": "#CC0000", "weight": 6},
    "ruta":             {"color": "#E05000", "weight": 4},
    "camino":           {"color": "#FF8800", "weight": 3},
    "pavimentado":      {"color": "#E05000", "weight": 4},
    "ripio":            {"color": "#DAA520", "weight": 2},
    "tierra":           {"color": "#8B6914", "weight": 2},
    "sendero":          {"color": "#A0522D", "weight": 1},
    "ferrocarril":      {"color": "#333333", "weight": 3},
    "ferroviaria":      {"color": "#333333", "weight": 3},
    "default":          {"color": "#888888", "weight": 2},
}

# Hidrología
ESTILOS_TIPO_HIDRO = {
    "rio":      {"color": "#1565C0", "weight": 3.5},
    "estero":   {"color": "#1E88E5", "weight": 2.0},
    "quebrada": {"color": "#64B5F6", "weight": 1.5},
    "canal":    {"color": "#00ACC1", "weight": 1.5},
    "lago":     {"color": "#0D47A1", "weight": 1.5},
    "laguna":   {"color": "#1976D2", "weight": 1.5},
    "default":  {"color": "#2196F3", "weight": 1.5},
}

# DEM: colormap hipsométrico estándar
COLORMAP_DEM = [
    (0.00, "#006400"),   # verde oscuro - mínimo
    (0.15, "#228B22"),   # verde bosque
    (0.30, "#9ACD32"),   # verde amarillento
    (0.45, "#DAA520"),   # dorado
    (0.60, "#CD853F"),   # marrón claro
    (0.75, "#8B4513"),   # marrón oscuro
    (0.88, "#D2B48C"),   # arena
    (1.00, "#FFFAFA"),   # blanco nieve - máximo
]

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones de color dinámico desde columna
# ─────────────────────────────────────────────────────────────

def construir_mapa_colores(serie, paleta):
    """
    Dado una columna (serie pandas), asigna un color
    de la paleta a cada valor único.
    Retorna: dict {valor -> color_hex}
    """
    valores = sorted(serie.dropna().unique().tolist())
    return {
        str(v): paleta[i % len(paleta)]
        for i, v in enumerate(valores)
    }

def crear_style_categorico(color_map, col, fill=True, weight=0.6):
    """
    Genera función de estilo folium para capa categórica.
    col: nombre de la columna a usar
    fill: True para polígonos, False para líneas
    """
    def style_fn(feature):
        val   = str(feature["properties"].get(col, ""))
        color = color_map.get(val, "#AAAAAA")
        if fill:
            return {
                "fillColor":   color,
                "color":       "#333333",
                "weight":      weight,
                "fillOpacity": 0.72,
            }
        else:
            return {
                "color":   color,
                "weight":  3,
                "opacity": 0.9,
            }
    return style_fn

def crear_style_transporte(color_map_tipo):
    """
    Estilo para transporte: primero busca coincidencia
    con palabras clave, luego usa el color del mapa dinámico.
    """
    def style_fn(feature):
        tipo = str(feature["properties"].get("TIPO", "")).lower()
        # Coincidencia con palabras clave estándar
        for key, vals in ESTILOS_TIPO_TRANSPORTE.items():
            if key in tipo:
                return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.9}
        # Fallback: color dinámico por valor exacto
        color = color_map_tipo.get(
            str(feature["properties"].get("TIPO", "")),
            ESTILOS_TIPO_TRANSPORTE["default"]["color"]
        )
        return {"color": color, "weight": 2, "opacity": 0.9}
    return style_fn

def crear_style_hidrologia():
    def style_fn(feature):
        tipo = str(feature["properties"].get("tipo", "")).lower()
        for key, vals in ESTILOS_TIPO_HIDRO.items():
            if key in tipo:
                return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.9}
        return {"color": ESTILOS_TIPO_HIDRO["default"]["color"],
                "weight": ESTILOS_TIPO_HIDRO["default"]["weight"],
                "opacity": 0.9}
    return style_fn

# ─────────────────────────────────────────────────────────────
# PASO 3: Funciones de leyenda HTML
# ─────────────────────────────────────────────────────────────

def leyenda_categorica_html(titulo, color_map, icono="🔲", posicion_top="10px", posicion_right="10px"):
    """
    Genera HTML de leyenda con cuadrados de color.
    posicion_top / posicion_right: para apilar varias leyendas.
    """
    items = ""
    for etiqueta, color in sorted(color_map.items()):
        items += f"""
        <div style="display:flex;align-items:center;margin:3px 0;">
          <div style="background:{color};width:16px;height:16px;
                      border:1px solid #555;margin-right:7px;
                      border-radius:2px;flex-shrink:0;"></div>
          <span style="font-size:11px;color:#222;">{etiqueta}</span>
        </div>"""

    return f"""
    <div style="
        position: fixed;
        top:   {posicion_top};
        right: {posicion_right};
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #bbb;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        max-height: 280px;
        overflow-y: auto;
        min-width: 170px;
        font-family: Arial, sans-serif;">
      <b style="font-size:12px;">{icono} {titulo}</b>
      <hr style="margin:5px 0;border-color:#ddd;">
      {items}
    </div>"""

def leyenda_dem_html(dem_min, dem_max, posicion_top="10px", posicion_right="10px"):
    """
    Barra de gradiente continuo con valores de elevación mínimo/máximo.
    """
    # Construir CSS gradient desde la paleta
    stops = ", ".join([f"{color} {int(pct*100)}%" for pct, color in COLORMAP_DEM])
    gradient = f"linear-gradient(to top, {stops})"

    return f"""
    <div style="
        position: fixed;
        top:   {posicion_top};
        right: {posicion_right};
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #bbb;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        min-width: 130px;
        font-family: Arial, sans-serif;">
      <b style="font-size:12px;">🏔️ Elevación (m)</b>
      <hr style="margin:5px 0;border-color:#ddd;">
      <div style="display:flex;align-items:stretch;gap:8px;">
        <div style="
            width:22px;
            height:150px;
            background:{gradient};
            border:1px solid #888;
            border-radius:3px;
            flex-shrink:0;">
        </div>
        <div style="display:flex;flex-direction:column;
                    justify-content:space-between;font-size:11px;color:#333;">
          <span><b>{int(dem_max)} m</b></span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.75)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.50)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.25)} m</span>
          <span><b>{int(dem_min)} m</b></span>
        </div>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────
# PASO 4: Raster → ImageOverlay con colormap DEM
# ─────────────────────────────────────────────────────────────

def aplicar_colormap_dem(band, nodata):
    """Aplica colormap hipsométrico al array de elevación."""
    posiciones = [p for p, _ in COLORMAP_DEM]
    colores    = [c for _, c in COLORMAP_DEM]
    cmap = mcolors.LinearSegmentedColormap.from_list("dem", list(zip(posiciones, colores)))

    mascara = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
    valid   = band[~mascara]

    dem_min = float(valid.min()) if len(valid) > 0 else 0
    dem_max = float(valid.max()) if len(valid) > 0 else 1

    norm  = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
    rgba  = cmap(norm(band))  # shape (H, W, 4)

    # Transparencia en nodata
    rgba[mascara, 3] = 0
    rgba[~mascara, 3] = 0.82

    img_array = (rgba * 255).astype(np.uint8)
    return img_array, dem_min, dem_max

def raster_a_overlay(raster_path, es_dem=False):
    with rasterio.open(raster_path) as src:

        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.zeros((src.count, height, width), dtype=np.float32)
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=data[i - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )
            bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        else:
            data = src.read().astype(np.float32)
            bounds_wgs84 = src.bounds

        nodata  = src.nodata
        dem_min = dem_max = None

        if es_dem:
            img_array, dem_min, dem_max = aplicar_colormap_dem(data[0], nodata)
        else:
            if src.count >= 3:
                rgb = data[:3].copy()
            else:
                rgb = np.stack([data[0]] * 3)

            for i in range(3):
                band  = rgb[i]
                mask  = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
                valid = band[~mask]
                if len(valid) > 0:
                    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                    rgb[i] = np.clip((band - mn) / (mx - mn + 1e-10), 0, 1)
                rgb[i][mask] = 0

            base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            alpha = np.full((base.shape[0], base.shape[1]), 200, dtype=np.uint8)
            if nodata is not None:
                alpha[data[0] == nodata] = 0
            img_array = np.dstack([base, alpha])

        img_pil = Image.fromarray(img_array)
        buf     = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        bounds = [
            [bounds_wgs84[1], bounds_wgs84[0]],
            [bounds_wgs84[3], bounds_wgs84[2]],
        ]
        return img_b64, bounds, dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# Cargar archivos
# ─────────────────────────────────────────────────────────────

archivos_vec = (
    list(DATA.glob("*.gpkg")) +
    list(DATA.glob("*.shp")) +
    list(DATA.glob("*.geojson"))
)

capas = {}
for archivo in archivos_vec:
    nombre = archivo.stem.replace("_", " ")
    try:
        gdf = gpd.read_file(archivo)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        capas[nombre] = gdf
    except Exception as e:
        st.warning(f"No fue posible cargar {archivo.name}: {e}")

archivos_raster = (
    list(DATA.glob("*.tif")) +
    list(DATA.glob("*.tiff")) +
    list(DATA.glob("*.img"))
)
rasters = {archivo.stem.replace("_", " "): archivo for archivo in archivos_raster}

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.title("Capas disponibles")
st.sidebar.subheader("🗺️ Vectores")
capas_activas = [n for n in capas if st.sidebar.checkbox(n, value=True, key=f"vec_{n}")]

st.sidebar.subheader("🛰️ Rasters")
rasters_activos = [n for n in rasters if st.sidebar.checkbox(n, value=False, key=f"rst_{n}")]

# ─────────────────────────────────────────────────────────────
# Mapa base
# ─────────────────────────────────────────────────────────────

centro = [-40.05, -72.87]
for gdf in capas.values():
    if len(gdf) > 0:
        c = gdf.unary_union.centroid
        centro = [c.y, c.x]
        break

m = folium.Map(location=centro, zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron",    name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 5: Agregar rasters
# ─────────────────────────────────────────────────────────────

leyendas_html = []       # acumular para agregar al final
offset_top    = 10       # px, para apilar leyendas verticalmente

for nombre in rasters_activos:
    nombre_low = nombre.lower()
    es_dem     = any(k in nombre_low for k in ["dem", "dtm", "elevacion", "mde"])

    try:
        with st.spinner(f"Cargando raster: {nombre}..."):
            img_b64, bounds, dem_min, dem_max = raster_a_overlay(rasters[nombre], es_dem=es_dem)

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=bounds,
            opacity=0.80,
            name=f"🛰 {nombre}",
        ).add_to(m)

        if es_dem and dem_min is not None:
            leyendas_html.append(
                leyenda_dem_html(dem_min, dem_max,
                                 posicion_top=f"{offset_top}px",
                                 posicion_right="10px")
            )
            offset_top += 220

    except Exception as e:
        st.warning(f"No fue posible cargar raster '{nombre}': {e}")

# ─────────────────────────────────────────────────────────────
# PASO 6: Agregar vectores con estilos y leyendas
# ─────────────────────────────────────────────────────────────

for nombre in capas_activas:
    gdf        = capas[nombre]
    nombre_low = nombre.lower()

    # ── Geología ─────────────────────────────────────────────
    if "geolog" in nombre_low:
        col = "AMBIENTE" if "AMBIENTE" in gdf.columns else "Ambiente"

        if col in gdf.columns:
            color_map = construir_mapa_colores(gdf[col], PALETA_GEOLOGIA)
            style_fn  = crear_style_categorico(color_map, col, fill=True)
            tooltip   = folium.GeoJsonTooltip(fields=[col])
        else:
            color_map = {}
            style_fn  = None
            tooltip   = folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))

        folium.GeoJson(
            gdf,
            name=f"🪨 {nombre}",
            style_function=style_fn,
            tooltip=tooltip,
        ).add_to(m)

        if color_map:
            leyendas_html.append(
                leyenda_categorica_html(
                    titulo="Geología (Ambiente)",
                    color_map=color_map,
                    icono="🪨",
                    posicion_top=f"{offset_top}px",
                    posicion_right="10px",
                )
            )
            offset_top += min(60 + len(color_map) * 23, 300) + 10

    # ── Hidrología ────────────────────────────────────────────
    elif "hidric" in nombre_low or "hidro" in nombre_low:
        cols_disp = [c for c in ["tipo", "LENGTH", "Nombre"] if c in gdf.columns]

        folium.GeoJson(
            gdf,
            name=f"💧 {nombre}",
            style_function=crear_style_hidrologia(),
            tooltip=folium.GeoJsonTooltip(fields=cols_disp) if cols_disp else None,
        ).add_to(m)

    # ── Transporte ────────────────────────────────────────────
    elif "transport" in nombre_low:
        col_tipo = "TIPO" if "TIPO" in gdf.columns else "tipo"

        if col_tipo in gdf.columns:
            color_map_t = construir_mapa_colores(gdf[col_tipo], [
                v["color"] for v in ESTILOS_TIPO_TRANSPORTE.values()
            ])
            style_fn = crear_style_transporte(color_map_t)
            tooltip  = folium.GeoJsonTooltip(fields=[col_tipo])
        else:
            style_fn = crear_style_transporte({})
            tooltip  = folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))
            color_map_t = {}

        folium.GeoJson(
            gdf,
            name=f"🛣️ {nombre}",
            style_function=style_fn,
            tooltip=tooltip,
        ).add_to(m)

        if color_map_t:
            leyendas_html.append(
                leyenda_categorica_html(
                    titulo="Transporte (Tipo)",
                    color_map=color_map_t,
                    icono="🛣️",
                    posicion_top=f"{offset_top}px",
                    posicion_right="10px",
                )
            )
            offset_top += min(60 + len(color_map_t) * 23, 300) + 10

    # ── Resto ─────────────────────────────────────────────────
    else:
        folium.GeoJson(
            gdf,
            name=nombre,
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:-1])),
        ).add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 7: Inyectar todas las leyendas al mapa
# ─────────────────────────────────────────────────────────────

for html in leyendas_html:
    m.get_root().html.add_child(folium.Element(html))

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=700)

# ─────────────────────────────────────────────────────────────
# Info sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.write(f"Capas vectoriales: **{len(capas)}**")
st.sidebar.write(f"Rasters:           **{len(rasters)}**")
st.sidebar.write(f"Activas:           **{len(capas_activas) + len(rasters_activos)}**")