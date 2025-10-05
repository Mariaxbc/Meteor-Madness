FILE_NAME = 'sbdb_query_results (2).csv' 
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import warnings
import math
import io 
# --- IMPORTAÇÕES PARA ASTRODINÂMICA ---
from poliastro.bodies import Sun
from poliastro.twobody.orbit import Orbit
from astropy import units as u
from astropy.time import Time
# --- FIM DAS IMPORTAÇÕES ---

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", message="The unit 'ua' has been deprecated")

# --- ÚNICA DEFINIÇÃO DO CAMINHO DO ARQUIVO ---
# --- FIM DA DEFINIÇÃO DO CAMINHO ---

# ===============================================================
# 🛠️ FUNÇÕES DE CÁLCULO DE IMPACTO 
# ===============================================================

def estimar_densidade_por_albedo(albedo):
    """Estima a densidade (kg/m³) com base no albedo."""
    if pd.isna(albedo): return 2500
    elif albedo > 0.35: return 2800
    elif albedo > 0.20: return 3200
    elif albedo > 0.10: return 7000
    elif albedo > 0.03: return 1800
    else: return 1000

def calcular_impacto(diameter_km, velocity_kms, density_kgm3, angulo=45):
    """Calcula a energia e o tamanho da cratera de impacto."""
    try:
        d = diameter_km * 1000
        v = velocity_kms * 1000
        ρ = density_kgm3
        η = math.sin(math.radians(angulo))

        massa = (4/3) * math.pi * (d/2)**3 * ρ 
        energia_j = 0.5 * massa * (v**2) * η
        energia_megatons = energia_j / 4.184e15
        diametro_cratera_km = (1.8 * (energia_j ** 0.22)) / 1000

        return {"Energia_Mt": energia_megatons, "Cratera_km": diametro_cratera_km}

    except Exception:
        return {"Energia_Mt": np.nan, "Cratera_km": np.nan}

def calcular_zonas_destruicao_simples(cratera_km):
    """Calcula as zonas de destruição em km a partir do diâmetro da cratera."""
    if pd.isna(cratera_km): return {"Zona_Total_km": np.nan, "Zona_Parcial_km": np.nan}
        
    raio_cratera_km = cratera_km / 2

    return {
        "Zona_Total_km": raio_cratera_km * 2,
        "Zona_Parcial_km": raio_cratera_km * 5,
    }
# --- FIM DAS FUNÇÕES DE CÁLCULO ---


# --- FUNÇÕES DE PROCESSAMENTO DE DADOS (Mantidas) ---

def clean_diameter(d):
    if pd.isna(d): return np.nan
    s = str(d).strip().replace(',', '.')
    match = re.search(r'^\s*(\d+\.?\d*)', s)
    if match: return float(match.group(1))
    match_range = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', s)
    if match_range: return float(match_range.group(1))
    return np.nan

def clean_year_range(year_str):
    if pd.isna(year_str): return np.nan, np.nan
    s = str(year_str).strip()
    parts = s.split('-')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit(): return int(parts[0]), int(parts[1])
    return np.nan, np.nan

def calculate_orbit_line(a, e, i_rad, om_rad, w_rad, num_points=300):
    nu_arr = np.linspace(0, 2 * np.pi, num_points); r_arr = a * (1 - e**2) / (1 + e * np.cos(nu_arr)); u_arr = w_rad + nu_arr
    cos_u, sin_u = np.cos(u_arr), np.sin(u_arr); cos_om, sin_om = np.cos(om_rad), np.sin(om_rad); cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    x_rot = r_arr * (cos_om * cos_u - sin_om * sin_u * cos_i); y_rot = r_arr * (sin_om * cos_u + cos_om * sin_u * cos_i); z_rot = r_arr * (sin_u * sin_i)
    return x_rot, y_rot, z_rot

def calculate_asteroid_position(row):
    required_fields = ['a', 'e', 'i', 'om', 'w', 'ma'];
    if any(pd.isna(row[f]) for f in required_fields): raise ValueError("Dados insuficientes para cálculo Poliastro.")
    try: obstime = Time(row['epoch_mjd'], format='mjd');
    except: obstime = Time('J2000', scale='tdb')
    try:
        orb = Orbit.from_classical(attractor=Sun, a=row['a'] * u.AU, ecc=row['e'] * u.one, inc=row['i'] * u.deg, raan=row['om'] * u.deg, argp=row['w'] * u.deg, nu=row['ma'] * u.deg, epoch=obstime)
        r = orb.r.to(u.AU).value; return r[0], r[1], r[2]
    except Exception:
        return np.nan, np.nan, np.nan 

def get_ps_color(ps_value):
    if pd.isna(ps_value): return '#7f8c8d'
    if ps_value >= -4.0: return '#e74c3c'
    elif ps_value >= -8.0: return '#f1c40f'
    else: return '#2ecc71'


# --- FUNÇÃO CRÍTICA DE CARREGAMENTO DE DADOS (REVISADA) ---
def get_initial_data(file_path_arg, num_rows=150):
    """Lê o arquivo CSV no novo formato e PRÉ-CALCULA o impacto."""
    try:
        # --- LEITURA ROBUSTA ---
        df = pd.read_csv(file_path_arg, sep=r'\s*;\s*', skipinitialspace=True, header=0, 
                         nrows=num_rows + 1, engine='python')
        
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'full_name': 'Full_Name', 'm': 'diameter_original_text',
                                'Date/Time': 'Impact_Date', 'Vel km/s': 'Velocity'}).copy()
        
    except Exception: return [], pd.DataFrame({'diameter_num': [0], 'a': [1], 'Min_Year': [2025], 'Max_Year': [2025], 'PS max': [0]}), 10.0

    numeric_cols = ['a', 'e', 'i', 'om', 'w', 'ma', 'epoch_mjd', 'Velocity', 'PS max', 'albedo']
    for col in numeric_cols:
         df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    
    # Processamento de Dados
    df['diameter_num'] = df['diameter_original_text'].apply(clean_diameter)
    df['Diameter_Text'] = df['diameter_original_text'].astype(str)
    df['first_obs'] = df['first_obs'].astype(str); df['last_obs'] = df['last_obs'].astype(str)
    df['class'] = df['class'].astype(str).str.strip(); df['albedo_num'] = df['albedo'].astype(float)
    df['Impact_Date'] = df['Impact_Date'].astype(str).str.strip()
    
    df[['Min_Year', 'Max_Year']] = df['Years'].apply(lambda x: pd.Series(clean_year_range(x)))

    # --- PRÉ-CÁLCULO DE IMPACTO NO DATAFRAME ---
    df['densidade'] = df['albedo_num'].apply(estimar_densidade_por_albedo)
    
    impact_results = df.apply(
        lambda row: calcular_impacto(row['diameter_num'] / 1000, row['Velocity'], row['densidade']), axis=1, result_type='expand'
    )
    df = pd.concat([df, impact_results], axis=1)
    
    # Calcula zonas de destruição
    df['Zona_Total_km'] = df['Cratera_km'].apply(lambda x: calcular_zonas_destruicao_simples(x)['Zona_Total_km'])
    df['Zona_Parcial_km'] = df['Cratera_km'].apply(lambda x: calcular_zonas_destruicao_simples(x)['Zona_Parcial_km'])
    # ----------------------------------------

    df = df.dropna(subset=['a', 'i', 'om']).copy()
    
    df['afhelion'] = df['a'] * (1 + df['e']); max_extent = df['afhelion'].max()
    if pd.isna(max_extent) or max_extent < 1.1: max_extent = 2.0 
    
    traces = []
    AU_TERRA = 1.0; EARTH_POSITION_HELIOCENTRIC = np.array([AU_TERRA, 0.0, 0.0])

    # Traces Fixas
    t_earth = np.linspace(0, 2 * np.pi, 300)
    x_earth_orb_geocentric = 1.0 * np.cos(t_earth) - EARTH_POSITION_HELIOCENTRIC[0]; y_earth_orb_geocentric = 1.0 * np.sin(t_earth) - EARTH_POSITION_HELIOCENTRIC[1]; z_earth_orb_geocentric = np.zeros_like(t_earth)
    x_sun_geocentric = 0.0 - EARTH_POSITION_HELIOCENTRIC[0]; y_sun_geocentric = 0.0 - EARTH_POSITION_HELIOCENTRIC[1]; z_sun_geocentric = 0.0 - EARTH_POSITION_HELIOCENTRIC[2]

    traces.extend([
        go.Scatter3d(x=x_earth_orb_geocentric, y=y_earth_orb_geocentric, z=z_earth_orb_geocentric, mode='lines', line=dict(color='blue', dash='dash'), name='Órbita da Terra', hoverinfo='none', customdata=[['reference']], showlegend=True),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='blue', size=5), name='Terra', hoverinfo='none', customdata=[['reference']], showlegend=True),
        go.Scatter3d(x=[x_sun_geocentric], y=[y_sun_geocentric], z=[z_sun_geocentric], mode='markers', marker=dict(color='orange', size=8), name='Sol', hoverinfo='none', customdata=[['reference']], showlegend=True),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=1, color='pink'), name='Órbitas de Asteroides', showlegend=True)
    ])


    # 2. Órbitas e Posições dos Asteroides
    for index, row in df.iterrows():
        
        if row['a'] <= 0 or row['e'] >= 1.0 or pd.isna(row['w']): continue 
        
        try:
            x_orbit_h, y_orbit_h, z_orbit_h = calculate_orbit_line(row['a'], row['e'], np.deg2rad(row['i']), np.deg2rad(row['om']), np.deg2rad(row['w']))
            x_pos_h, y_pos_h, z_pos_h = calculate_asteroid_position(row)
            
            x_orbit_g = x_orbit_h - EARTH_POSITION_HELIOCENTRIC[0]; y_orbit_g = y_orbit_h - EARTH_POSITION_HELIOCENTRIC[1]; z_orbit_g = z_orbit_h - EARTH_POSITION_HELIOCENTRIC[2]
            x_pos_g = x_pos_h - EARTH_POSITION_HELIOCENTRIC[0]; y_pos_g = y_pos_h - EARTH_POSITION_HELIOCENTRIC[1]; z_pos_g = z_pos_h - EARTH_POSITION_HELIOCENTRIC[2]
            
            trace_name = row['Full_Name'].strip(); diameter_text = row['Diameter_Text']
            marker_color = get_ps_color(row['PS max'])
            
            # customdata FINAL (Inclui todos os 13 campos, incluindo os calculados)
            trace_data = [row['diameter_num'], diameter_text, row['first_obs'], row['last_obs'], 
                          row['Min_Year'], row['Max_Year'], row['Impact_Date'], row['Velocity'], 
                          row['PS max'], row['class'], row['albedo_num'],
                          row['Zona_Total_km'], row['Zona_Parcial_km']] 

            # Trace da Órbita do Asteroide (linha)
            traces.append(go.Scatter3d(
                x=x_orbit_g, y=y_orbit_g, z=z_orbit_g, mode='lines', line=dict(width=1.5, color='pink'), name=f"Órbita {trace_name}",
                customdata=[trace_data], 
                # --- HOVER TEMPLATE COMPLETO (FINAL) ---
                hovertemplate=(
                    f"Órbita: {trace_name}<br>Diâmetro (m): {diameter_text}<br>"
                    f"Risco (PS max): {row['PS max']:.2f}<br>"
                    f"Classe: {row['class']}<br>"
                    f"Primeira Obs: {row['first_obs']}<br>"
                    f"Última Obs: {row['last_obs']}<br>"
                    f"Impacto: {row['Impact_Date']}<br>"
                    f"Velocidade: {row['Velocity']:.2f} km/s<br>"
                    f"--- ZONAS DE DESTRUIÇÃO (Simuladas) ---<br>"
                    f"Cratera Estimada: {row['Cratera_km']:.2f} km<br>"
                    f"Destruição Total: {row['Zona_Total_km']:.2f} km<br>"
                    f"Destruição Parcial: {row['Zona_Parcial_km']:.2f} km<extra></extra>"
                ),
                visible=True, showlegend=False 
            ))

            # Trace da Posição Atual do Asteroide (marcador)
            traces.append(go.Scatter3d(
                x=[x_pos_g], y=[y_pos_g], z=[z_pos_g], mode='markers', marker=dict(size=6, color=marker_color, symbol='circle'),
                name=f"POSIÇÃO {trace_name}", customdata=[trace_data],
                hovertemplate=(
                    f"POSIÇÃO: {trace_name}<br>Diâmetro (m): {diameter_text}<br>"
                    f"Risco (PS max): {row['PS max']:.2f}<br>"
                    f"Classe: {row['class']}<br>"
                    f"Primeira Obs: {row['first_obs']}<br>"
                    f"Última Obs: {row['last_obs']}<br>"
                    f"Impacto: {row['Impact_Date']}<br>"
                    f"Velocidade: {row['Velocity']:.2f} km/s<br>"
                    f"--- ZONAS DE DESTRUIÇÃO (Simuladas) ---<br>"
                    f"Cratera Estimada: {row['Cratera_km']:.2f} km<br>"
                    f"Destruição Total: {row['Zona_Total_km']:.2f} km<br>"
                    f"Destruição Parcial: {row['Zona_Parcial_km']:.2f} km<extra></extra>"
                ),
                visible=True, showlegend=False 
            ))

        except Exception:
            continue
            
    return traces, df, max_extent

# --- INICIALIZAÇÃO E APLICAÇÃO DASH (Mantida) ---

ASTEROID_TRACES, ASTEROID_DF, MAX_ORBIT_EXTENT = get_initial_data(FILE_NAME, num_rows=100)

# [Configuração dos SLIDERS E LIMITES mantida]
AXIS_LIMIT = MAX_ORBIT_EXTENT * 1.05

MAX_DIAM = ASTEROID_DF['diameter_num'].max(); MIN_DIAM = ASTEROID_DF['diameter_num'].min()
if pd.isna(MAX_DIAM) or pd.isna(MIN_DIAM): MIN_DIAM, MAX_DIAM = 0, 1000
else:
    margin_d = (MAX_DIAM - MIN_DIAM) * 0.05; MIN_DIAM = max(0, int(MIN_DIAM - margin_d)); MAX_DIAM = int(MAX_DIAM + margin_d)
SLIDER_MARKS_D = {MIN_DIAM: f'{MIN_DIAM}m', MAX_DIAM: f'{MAX_DIAM}m'}

if 'Min_Year' in ASTEROID_DF.columns and not ASTEROID_DF['Min_Year'].empty:
    GLOBAL_MIN_YEAR = ASTEROID_DF['Min_Year'].min(); GLOBAL_MAX_YEAR = ASTEROID_DF['Max_Year'].max()
else: GLOBAL_MIN_YEAR, GLOBAL_MAX_YEAR = 2025, 2150
if pd.isna(GLOBAL_MIN_YEAR) or pd.isna(GLOBAL_MAX_YEAR): GLOBAL_MIN_YEAR, GLOBAL_MAX_YEAR = 2025, 2150
GLOBAL_MIN_YEAR = int(GLOBAL_MIN_YEAR // 5 * 5); GLOBAL_MAX_YEAR = int(GLOBAL_MAX_YEAR // 5 * 5 + 5); STEP_Y = 5
SLIDER_MARKS_Y = {y: str(y) for y in range(GLOBAL_MIN_YEAR, GLOBAL_MAX_YEAR + 1, STEP_Y * 2)}

if 'PS max' in ASTEROID_DF.columns and not ASTEROID_DF['PS max'].empty:
    GLOBAL_MIN_PS = ASTEROID_DF['PS max'].min(); GLOBAL_MAX_PS = ASTEROID_DF['PS max'].max()
else: GLOBAL_MIN_PS, GLOBAL_MAX_PS = -10.0, 0.0
if pd.isna(GLOBAL_MIN_PS) or pd.isna(GLOBAL_MAX_PS): GLOBAL_MIN_PS, GLOBAL_MAX_PS = -10.0, 0.0
GLOBAL_MIN_PS = np.floor(GLOBAL_MIN_PS - 1); GLOBAL_MAX_PS = np.ceil(GLOBAL_MAX_PS + 1)
SLIDER_MARKS_PS = {p: f'{p:.1f}' for p in np.arange(GLOBAL_MIN_PS, GLOBAL_MAX_PS + 0.1, 1)}


app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'height': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    html.H1("Visualização 3D de Órbitas de NEOs", style={'textAlign': 'center', 'paddingTop': '20px', 'marginBottom': '10px'}),
    
    dcc.Graph(id='3d-orbit-graph', style={'height': '70vh'}, config={'displayModeBar': False}),

    html.Div(style={'padding': '10px 5%', 'display': 'flex', 'flex-direction': 'column', 'gap': '15px'}, children=[
        
        html.Div(style={'flexGrow': 1}, children=[html.Label('Diâmetro (m) Min / Max:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),dcc.RangeSlider(id='diameter-slider', min=MIN_DIAM, max=MAX_DIAM, value=[MIN_DIAM, MAX_DIAM],marks=SLIDER_MARKS_D, step=1, tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(style={'flexGrow': 1}, children=[html.Label('Ano de Queda (Intervalo de Risco):', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),dcc.RangeSlider(id='year-slider', min=GLOBAL_MIN_YEAR, max=GLOBAL_MAX_YEAR, value=[GLOBAL_MIN_YEAR, GLOBAL_MAX_YEAR],marks=SLIDER_MARKS_Y, step=STEP_Y, tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(style={'flexGrow': 1}, children=[html.Label('PS max (Escala de Palermo):', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),dcc.RangeSlider(id='ps-slider', min=GLOBAL_MIN_PS, max=GLOBAL_MAX_PS, value=[GLOBAL_MIN_PS, GLOBAL_MAX_PS],marks=SLIDER_MARKS_PS, step=0.1, tooltip={"placement": "bottom", "always_visible": True})]),
    ]),
    
    html.Div(id='asteroid-info-panel', style={'padding': '10px 5%', 'borderTop': '1px solid #444', 'marginTop': '10px'})
])


@callback(
    Output('3d-orbit-graph', 'figure'),
    [Input('diameter-slider', 'value'), Input('year-slider', 'value'), Input('ps-slider', 'value')]
)
def update_figure(selected_diameter_range, selected_year_range, selected_ps_range):
    # [Lógica de Filtro]
    min_diam, max_diam = selected_diameter_range
    min_year_sel, max_year_sel = selected_year_range
    min_ps, max_ps = selected_ps_range
    current_traces = []
    
    for trace in ASTEROID_TRACES:
        trace_dict = trace.to_plotly_json()
        
        try:
            if trace.customdata and trace.customdata[0][0] != 'reference': 
                diam = trace.customdata[0][0]
                min_year_ast = trace.customdata[0][4] 
                max_year_ast = trace.customdata[0][5]
                ps_value = trace.customdata[0][8]
                
                is_visible = False
                
                is_filtered_by_diameter = not pd.isna(diam) and min_diam <= diam <= max_diam
                is_filtered_by_year = (not pd.isna(min_year_ast) and not pd.isna(max_year_ast) and min_year_sel <= min_year_ast and max_year_ast <= max_year_sel)
                is_filtered_by_ps = (pd.isna(ps_value) and True) or (not pd.isna(ps_value) and min_ps <= ps_value <= max_ps)
                
                if is_filtered_by_diameter and is_filtered_by_year and is_filtered_by_ps:
                    is_visible = True
                
                trace_dict['visible'] = is_visible
            
            else:
                 trace_dict['visible'] = True 

        except Exception:
             trace_dict['visible'] = True 

        current_traces.append(trace_dict)
        
    scene_layout = dict(
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False, title='', range=[-AXIS_LIMIT, AXIS_LIMIT]),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False, title='', range=[-AXIS_LIMIT, AXIS_LIMIT]),
        zaxis=dict(showticklabels=False, zeroline=False, showgrid=False, title='', range=[-AXIS_LIMIT, AXIS_LIMIT]),
        aspectmode='data', bgcolor='rgba(0,0,0,0)'
    )
    
    fig = go.Figure(data=current_traces)
    
    fig.update_layout(
        scene=scene_layout, template='plotly_dark', margin=dict(l=0, r=0, b=0, t=30), showlegend=True,
        scene_camera=dict(center=dict(x=0, y=0, z=0), eye=dict(x=3, y=1, z=0.5))
    )

    return fig

if __name__ == '__main__':
    app.run()
