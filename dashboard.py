"""
Dashboard — Prédiction de Maladies Cardiaques
Master IA & Data Science — UMEF University
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
NUM_COLS     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CAT_COLS     = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
TARGET       = 'target'
RANDOM_STATE = 42

COLORS = {
    'primary'  : '#c0392b',
    'secondary': '#1a2f45',
    'success'  : '#1a7a4a',
    'warning'  : '#d68910',
    'info'     : '#1a6fa8',
    'light'    : '#eef2f7',
    'muted'    : '#5d7185',
    'card'     : '#ffffff',
    'border'   : '#d0dce8',
    'text'     : '#2d3748'
}

# ─── CHARGEMENT DES DONNÉES ───────────────────────────────────────────────────
df = pd.read_csv('data/heart.csv').drop_duplicates().reset_index(drop=True)
X  = df.drop(TARGET, axis=1)
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_COLS),
    ('cat', 'passthrough',   CAT_COLS)
])

# ─── CHARGEMENT DES MODÈLES ───────────────────────────────────────────────────
model_names = ['Logistic Regression', 'Random Forest', 'SVM']
model_files = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'Random Forest'      : 'models/random_forest.pkl',
    'SVM'                : 'models/svm.pkl'
}
model_configs = {
    'Logistic Regression': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(C=0.1, random_state=RANDOM_STATE))]),
    'Random Forest'      : Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))]),
    'SVM'                : Pipeline([('pre', preprocessor), ('clf', SVC(probability=True, C=0.1, kernel='linear', random_state=RANDOM_STATE))])
}

results = {}
for name in model_names:
    try:
        model = joblib.load(model_files[name])
        print(f"✅ {name} chargé depuis pkl")
    except:
        print(f"⚠️ {name} pkl non trouvé — réentraînement...")
        model = model_configs[name]
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'model'    : model,
        'y_pred'   : y_pred
    }

best_name   = max(results, key=lambda x: results[x]['accuracy'])
best_model  = results[best_name]['model']
y_pred_best = results[best_name]['y_pred']

try:
    best_model = joblib.load('models/best_model.pkl')
    print("✅ Meilleur modèle chargé depuis best_model.pkl")
except:
    pass

# ─── FONCTIONS GRAPHIQUES ─────────────────────────────────────────────────────

def create_kpi_card(title, value, color, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '24px', 'marginBottom': '6px'}),
        html.Div(value, style={'fontSize': '22px', 'fontWeight': '700', 'color': color}),
        html.Div(title, style={'fontSize': '11px', 'color': COLORS['muted'], 'marginTop': '4px'})
    ], style={
        'backgroundColor': COLORS['card'],
        'borderRadius'   : '14px',
        'padding'        : '20px 14px',
        'textAlign'      : 'center',
        'boxShadow'      : '0 4px 16px rgba(26,47,69,0.10)',
        'border'         : f'1px solid {COLORS["border"]}',
        'flex'           : '1',
        'margin'         : '6px'
    })


def fig_distribution():
    counts        = df[TARGET].value_counts().reset_index()
    counts.columns= ['Classe', 'Nombre']
    counts['Label']= counts['Classe'].map({1: 'Sain', 0: 'Malade'})
    fig = px.bar(counts, x='Label', y='Nombre',
                 color='Label',
                 color_discrete_map={'Sain': COLORS['success'], 'Malade': COLORS['primary']},
                 title='Distribution : Sains vs Malades',
                 text='Nombre')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, plot_bgcolor='white',
                      paper_bgcolor='white', margin=dict(t=50, b=20))
    return fig


def fig_age():
    fig = go.Figure()
    for val, color, label in [(0, COLORS['primary'], 'Malade'), (1, COLORS['success'], 'Sain')]:
        fig.add_trace(go.Histogram(
            x=df[df[TARGET]==val]['age'], name=label,
            marker_color=color, opacity=0.7, nbinsx=20))
    fig.update_layout(
        barmode='overlay', title="Distribution de l'âge par classe",
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98), margin=dict(t=50, b=20))
    return fig


def fig_correlation():
    corr = df.corr(numeric_only=True)
    fig  = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmid=0,
        text=corr.round(2).values,
        texttemplate='%{text}', textfont=dict(size=9)
    ))
    fig.update_layout(
        title='Matrice de Corrélation',
        paper_bgcolor='white', margin=dict(t=50, b=20), height=500)
    return fig


def fig_model_comparison():
    metrics  = ['accuracy', 'f1', 'precision', 'recall']
    labels   = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    colors_m = [COLORS['info'], COLORS['primary'], COLORS['success'], COLORS['warning']]

    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors_m):
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=[results[n][metric] for n in model_names],
            marker_color=color,
            text=[f"{results[n][metric]:.3f}" for n in model_names],
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group', title='Comparaison des modèles',
        yaxis=dict(range=[0.6, 1.05]),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.1),
        margin=dict(t=80, b=20))
    return fig


def fig_confusion():
    cm  = confusion_matrix(y_test, y_pred_best)
    fig = px.imshow(cm, text_auto=True,
                    x=['Malade', 'Sain'], y=['Malade', 'Sain'],
                    color_continuous_scale='Blues',
                    title=f'Matrice de Confusion — {best_name}',
                    labels=dict(x='Prédit', y='Réel'))
    fig.update_layout(paper_bgcolor='white', margin=dict(t=50, b=20))
    return fig


def fig_roc():
    fig    = go.Figure()
    colors = [COLORS['info'], COLORS['primary'], COLORS['success']]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['model'].predict_proba(X_test)[:, 1])
        roc_auc     = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'{name} (AUC={roc_auc:.3f})',
            line=dict(color=color, width=2)))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray'), name='Baseline'))

    fig.update_layout(
        title='Courbes ROC', xaxis_title='FPR', yaxis_title='TPR',
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.6, y=0.1), margin=dict(t=50, b=20))
    return fig


# ─── APP DASH ─────────────────────────────────────────────────────────────────
app       = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Heart Disease — Dashboard ML"

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # HEADER
    html.Div([
        html.Div([
            html.Div([
                html.Span('🫀', style={'fontSize': '44px', 'lineHeight': '1'}),
            ], style={'padding': '8px', 'backgroundColor': 'rgba(192,57,43,0.25)',
                      'borderRadius': '12px'}),
            html.Div([
                html.H1('Prédiction de Maladies Cardiaques',
                        style={'color': 'white', 'margin': '0',
                               'fontSize': '22px', 'fontWeight': '500',
                               'letterSpacing': '0.3px'}),
                html.Div([
                    html.Span('🩺', style={'fontSize': '11px', 'marginRight': '5px'}),
                    html.Span("Outil d'aide à la décision médicale par IA",
                              style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '12px'})
                ], style={'marginTop': '4px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Span('👤', style={'fontSize': '11px', 'marginRight': '5px'}),
                    html.Span('Mouhamadou Moustapha BA — Master IA & Data Science — UMEF University',
                              style={'color': 'rgba(255,255,255,0.45)', 'fontSize': '11px'})
                ], style={'marginTop': '3px', 'display': 'flex', 'alignItems': 'center'})
            ], style={'marginLeft': '16px'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'backgroundColor': '#1a2f45',
        'padding'        : '20px 28px',
        'borderBottom'   : '3px solid #c0392b'
    }),

    # TABS
    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='🏠 Vue d\'ensemble', value='overview'),
        dcc.Tab(label='📊 Modèles',          value='models'),
        dcc.Tab(label='🔮 Prévision',         value='prediction'),
        dcc.Tab(label='ℹ️ À propos',           value='about'),
    ], style={'fontSize': '14px'}),

    # CONTENU
    html.Div(id='tab-content', style={
        'backgroundColor': '#eef2f7',
        'minHeight'      : '80vh',
        'padding'        : '24px'
    })

], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'margin': '0', 'backgroundColor': '#eef2f7'})


# ─── CALLBACK PRINCIPAL ───────────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if tab == 'overview':
        return html.Div([

            # KPIs
            html.Div([
                create_kpi_card('Patients total',    f"{len(df)}",                           COLORS['secondary'], '👥'),
                create_kpi_card('Malades (0)',        f"{(df[TARGET]==0).sum()}",             COLORS['primary'],   '🔴'),
                create_kpi_card('Sains (1)',          f"{(df[TARGET]==1).sum()}",             COLORS['success'],   '✅'),
                create_kpi_card('Âge moyen',         f"{df['age'].mean():.0f} ans",          COLORS['info'],      '🎂'),
                create_kpi_card('Cholestérol moy.',  f"{df['chol'].mean():.0f} mg/dl",       COLORS['warning'],   '🩺'),
                create_kpi_card('Meilleur modèle',   best_name.split()[0],                   COLORS['secondary'], '🏆'),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '16px'}),

            # GRAPHIQUES LIGNE 1
            html.Div([
                html.Div([dcc.Graph(figure=fig_distribution())],
                         style={'width': '48%', 'backgroundColor': COLORS['card'],
                                'borderRadius': '12px', 'padding': '12px',
                                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'}),
                html.Div([dcc.Graph(figure=fig_age())],
                         style={'width': '48%', 'backgroundColor': COLORS['card'],
                                'borderRadius': '12px', 'padding': '12px',
                                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '16px'}),

            # HEATMAP
            html.Div([dcc.Graph(figure=fig_correlation())],
                     style={'backgroundColor': COLORS['card'], 'borderRadius': '12px',
                            'padding': '12px', 'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'})
        ])

    # ── MODÈLES ───────────────────────────────────────────────────────────────
    elif tab == 'models':
        return html.Div([

            # KPIs meilleur modèle
            html.Div([
                create_kpi_card('Meilleur modèle', best_name.split()[0],                         COLORS['secondary'], '👑'),
                create_kpi_card('Accuracy',         f"{results[best_name]['accuracy']:.3f}",      COLORS['info'],      '🎯'),
                create_kpi_card('F1-Score',         f"{results[best_name]['f1']:.3f}",            COLORS['primary'],   '📊'),
                create_kpi_card('Precision',        f"{results[best_name]['precision']:.3f}",     COLORS['success'],   '✅'),
                create_kpi_card('Recall',           f"{results[best_name]['recall']:.3f}",        COLORS['warning'],   '🔍'),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '16px'}),

            # COMPARAISON + CONFUSION
            html.Div([
                html.Div([dcc.Graph(figure=fig_model_comparison())],
                         style={'width': '56%', 'backgroundColor': COLORS['card'],
                                'borderRadius': '12px', 'padding': '12px',
                                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'}),
                html.Div([dcc.Graph(figure=fig_confusion())],
                         style={'width': '40%', 'backgroundColor': COLORS['card'],
                                'borderRadius': '12px', 'padding': '12px',
                                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '16px'}),

            # ROC
            html.Div([dcc.Graph(figure=fig_roc())],
                     style={'backgroundColor': COLORS['card'], 'borderRadius': '12px',
                            'padding': '12px', 'boxShadow': '0 2px 12px rgba(0,0,0,0.08)'})
        ])

    # ── PRÉVISION ─────────────────────────────────────────────────────────────
    elif tab == 'prediction':
        return html.Div([

            html.H3('🔮 Simulateur de Risque Cardiaque',
                    style={'color': COLORS['secondary'], 'marginBottom': '16px'}),

            html.Div([

                # FORMULAIRE
                html.Div([
                    html.H4('Données du patient',
                            style={'color': COLORS['secondary'], 'marginBottom': '14px', 'fontSize': '14px'}),

                    html.Label('Âge (ans)', style={'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Slider(id='age', min=29, max=77, step=1, value=55,
                               marks={29:'29', 45:'45', 55:'55', 65:'65', 77:'77'},
                               tooltip={'placement': 'bottom'}),

                    html.Label('Sexe', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.RadioItems(id='sex', options=[
                        {'label': ' Homme', 'value': 1},
                        {'label': ' Femme',  'value': 0}
                    ], value=1, inline=True, style={'fontSize': '13px'}),

                    html.Label('Pression artérielle (mm Hg)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Slider(id='trestbps', min=94, max=200, step=2, value=130,
                               marks={94:'94', 120:'120', 140:'140', 160:'160', 200:'200'},
                               tooltip={'placement': 'bottom'}),

                    html.Label('Cholestérol (mg/dl)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Slider(id='chol', min=126, max=564, step=5, value=246,
                               marks={126:'126', 200:'200', 300:'300', 400:'400', 564:'564'},
                               tooltip={'placement': 'bottom'}),

                    html.Label('Fréquence cardiaque max', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Slider(id='thalach', min=71, max=202, step=1, value=150,
                               marks={71:'71', 120:'120', 150:'150', 180:'180', 202:'202'},
                               tooltip={'placement': 'bottom'}),

                    html.Label('Dépression ST (oldpeak)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Slider(id='oldpeak', min=0.0, max=6.2, step=0.1, value=1.0,
                               marks={0:'0', 2:'2', 4:'4', 6:'6'},
                               tooltip={'placement': 'bottom'}),

                    html.Label('Type de douleur thoracique (cp)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Dropdown(id='cp', options=[
                        {'label': '0 — Asymptomatique (grave)', 'value': 0},
                        {'label': '1 — Angine typique',         'value': 1},
                        {'label': '2 — Angine atypique',        'value': 2},
                        {'label': '3 — Douleur non cardiaque',  'value': 3},
                    ], value=1, clearable=False, style={'fontSize': '12px'}),

                    html.Label('Vaisseaux bloqués (ca)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Dropdown(id='ca', options=[
                        {'label': '0 — Aucun vaisseau',    'value': 0},
                        {'label': '1 — Un vaisseau',       'value': 1},
                        {'label': '2 — Deux vaisseaux',    'value': 2},
                        {'label': '3 — Trois vaisseaux',   'value': 3},
                    ], value=0, clearable=False, style={'fontSize': '12px'}),

                    html.Label('Angine d\'effort (exang)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.RadioItems(id='exang', options=[
                        {'label': ' Non', 'value': 0},
                        {'label': ' Oui', 'value': 1}
                    ], value=0, inline=True, style={'fontSize': '13px'}),

                    html.Label('Glycémie à jeun > 120 mg/dl (fbs)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.RadioItems(id='fbs', options=[
                        {'label': ' Non', 'value': 0},
                        {'label': ' Oui', 'value': 1}
                    ], value=0, inline=True, style={'fontSize': '13px'}),

                    html.Label('Thalassémie (thal)', style={'marginTop': '12px', 'display': 'block', 'fontSize': '12px', 'color': COLORS['muted']}),
                    dcc.Dropdown(id='thal', options=[
                        {'label': '0 — Normal',            'value': 0},
                        {'label': '1 — Défaut fixe',       'value': 1},
                        {'label': '2 — Normal (courant)',   'value': 2},
                        {'label': '3 — Défaut réversible', 'value': 3},
                    ], value=2, clearable=False, style={'fontSize': '12px'}),

                ], style={
                    'width'          : '45%',
                    'backgroundColor': COLORS['card'],
                    'borderRadius'   : '12px',
                    'padding'        : '20px',
                    'boxShadow'      : '0 2px 12px rgba(0,0,0,0.08)'
                }),

                # RÉSULTAT
                html.Div([
                    html.H4('Résultat de la prévision',
                            style={'color': COLORS['secondary'], 'marginBottom': '16px', 'fontSize': '14px'}),

                    html.Div(id='prediction-result'),

                    html.H4('Profils Sénégalais — Tests rapides',
                            style={'color': COLORS['secondary'], 'margin': '20px 0 12px', 'fontSize': '13px'}),

                    html.Div([
                        html.Button('👴 Mamadou Diallo — 63 ans — Risque élevé',
                                    id='btn-mamadou', n_clicks=0,
                                    style={'width': '100%', 'marginBottom': '6px', 'padding': '9px',
                                           'borderRadius': '8px', 'border': f'1px solid {COLORS["primary"]}',
                                           'backgroundColor': 'white', 'cursor': 'pointer',
                                           'color': COLORS['primary'], 'fontWeight': '500', 'fontSize': '12px', 'textAlign': 'left'}),
                        html.Button('👩 Aïssatou Ndiaye — 45 ans — Profil sain',
                                    id='btn-aissatou', n_clicks=0,
                                    style={'width': '100%', 'marginBottom': '6px', 'padding': '9px',
                                           'borderRadius': '8px', 'border': f'1px solid {COLORS["success"]}',
                                           'backgroundColor': 'white', 'cursor': 'pointer',
                                           'color': COLORS['success'], 'fontWeight': '500', 'fontSize': '12px', 'textAlign': 'left'}),
                        html.Button('👩 Fatou Sow — 65 ans — Profil intermédiaire',
                                    id='btn-fatou', n_clicks=0,
                                    style={'width': '100%', 'marginBottom': '6px', 'padding': '9px',
                                           'borderRadius': '8px', 'border': f'1px solid {COLORS["warning"]}',
                                           'backgroundColor': 'white', 'cursor': 'pointer',
                                           'color': COLORS['warning'], 'fontWeight': '500', 'fontSize': '12px', 'textAlign': 'left'}),
                        html.Button('👦 Ibrahima Mbaye — 40 ans — Profil sain',
                                    id='btn-ibrahima', n_clicks=0,
                                    style={'width': '100%', 'marginBottom': '6px', 'padding': '9px',
                                           'borderRadius': '8px', 'border': f'1px solid {COLORS["info"]}',
                                           'backgroundColor': 'white', 'cursor': 'pointer',
                                           'color': COLORS['info'], 'fontWeight': '500', 'fontSize': '12px', 'textAlign': 'left'}),
                        html.Button('👧 Rokhaya Diop — 35 ans — Très saine',
                                    id='btn-rokhaya', n_clicks=0,
                                    style={'width': '100%', 'padding': '9px',
                                           'borderRadius': '8px', 'border': f'1px solid {COLORS["success"]}',
                                           'backgroundColor': 'white', 'cursor': 'pointer',
                                           'color': COLORS['success'], 'fontWeight': '500', 'fontSize': '12px', 'textAlign': 'left'}),
                    ])

                ], style={
                    'width'          : '50%',
                    'backgroundColor': COLORS['card'],
                    'borderRadius'   : '12px',
                    'padding'        : '20px',
                    'boxShadow'      : '0 2px 12px rgba(0,0,0,0.08)'
                })

            ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '16px'})
        ])

    # ── À PROPOS ──────────────────────────────────────────────────────────────
    elif tab == 'about':

        def info_card(icon, title, content, color):
            return html.Div([
                html.Div(f"{icon}  {title}",
                         style={'fontWeight': '500', 'fontSize': '14px',
                                'color': color, 'marginBottom': '10px'}),
                html.Div(content, style={'color': COLORS['text'], 'lineHeight': '1.8',
                                         'fontSize': '13px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'borderRadius'   : '12px',
                'padding'        : '18px',
                'boxShadow'      : '0 2px 12px rgba(0,0,0,0.08)',
                'borderLeft'     : f'4px solid {color}',
                'marginBottom'   : '12px',
                'borderTopLeftRadius' : '0',
                'borderBottomLeftRadius': '0',
            })

        best_acc = results[best_name]['accuracy']
        best_rec = results[best_name]['recall']

        return html.Div([
            html.H3('ℹ️ À Propos du Projet',
                    style={'color': COLORS['secondary'], 'marginBottom': '20px'}),

            info_card('👤', 'Auteur', html.Div([
                html.Div('Mouhamadou Moustapha BA', style={'fontWeight': '500', 'fontSize': '15px', 'color': COLORS['secondary']}),
                html.Div('Master IA & Data Science', style={'marginTop': '2px'}),
                html.Div('UMEF University — Genève, Suisse', style={'marginTop': '2px'}),
                html.Div('Cours : Généralités sur les IA', style={'marginTop': '2px', 'color': COLORS['muted']}),
            ]), COLORS['info']),

            info_card('📊', 'Dataset', html.Div([
                html.Div('Heart Disease UCI Dataset', style={'fontWeight': '500', 'color': COLORS['secondary']}),
                html.Div('Source : Kaggle / UCI Machine Learning Repository', style={'marginTop': '4px'}),
                html.Div(f'Patients : 302 (après suppression des doublons)', style={'marginTop': '4px'}),
                html.Div('Features : 13 variables cliniques + 1 cible', style={'marginTop': '4px'}),
                html.Div('Période : Données collectées en 1988', style={'marginTop': '4px'}),
                html.Div('Distribution : 164 sains (54.3%) — 138 malades (45.7%)', style={'marginTop': '4px'}),
            ]), COLORS['warning']),

            info_card('🤖', 'Modèles ML utilisés', html.Div([
                html.Div([
                    html.Span('👑 ', style={'fontSize': '13px'}),
                    html.Span(f'Logistic Regression — Meilleur modèle (Accuracy : {best_acc:.1%}, Recall : {best_rec:.1%})',
                              style={'fontWeight': '500', 'color': COLORS['secondary']})
                ], style={'marginBottom': '6px'}),
                html.Div('• Random Forest (n_estimators=100, GridSearch optimisé)', style={'marginTop': '4px'}),
                html.Div('• SVM — Support Vector Machine (kernel=linear)', style={'marginTop': '4px'}),
                html.Div('Optimisation : GridSearchCV avec validation croisée 5 folds', style={'marginTop': '8px', 'color': COLORS['muted']}),
            ]), COLORS['primary']),

            info_card('📈', 'Performances', html.Div([
                html.Div([
                    html.Div(f"Accuracy   : {results[best_name]['accuracy']:.3f}  (82.0%)",  style={'marginBottom': '4px'}),
                    html.Div(f"F1-Score   : {results[best_name]['f1']:.3f}",                 style={'marginBottom': '4px'}),
                    html.Div(f"Precision  : {results[best_name]['precision']:.3f}",           style={'marginBottom': '4px'}),
                    html.Div(f"Recall     : {results[best_name]['recall']:.3f}  (87.9%) ← Métrique prioritaire en médecine", style={'marginBottom': '4px'}),
                    html.Div(f"AUC-ROC    : 0.872",                                           style={'marginBottom': '4px'}),
                ])
            ]), COLORS['success']),

            info_card('🛠️', 'Technologies utilisées', html.Div([
                html.Div('• Python 3.12', style={'marginBottom': '4px'}),
                html.Div('• Scikit-learn — Modèles ML + Pipeline + GridSearch', style={'marginBottom': '4px'}),
                html.Div('• Pandas / NumPy — Manipulation des données', style={'marginBottom': '4px'}),
                html.Div('• Dash / Plotly — Dashboard interactif', style={'marginBottom': '4px'}),
                html.Div('• Joblib — Sauvegarde du modèle', style={'marginBottom': '4px'}),
                html.Div('• Google Colab — Développement du notebook', style={'marginBottom': '4px'}),
            ]), COLORS['secondary']),

        ], style={'maxWidth': '860px'})


# ─── CALLBACK PRÉVISION ───────────────────────────────────────────────────────
@app.callback(
    Output('prediction-result', 'children'),
    [Input('age', 'value'),      Input('sex', 'value'),
     Input('trestbps', 'value'), Input('chol', 'value'),
     Input('thalach', 'value'),  Input('oldpeak', 'value'),
     Input('cp', 'value'),       Input('ca', 'value'),
     Input('exang', 'value'),    Input('fbs', 'value'),
     Input('thal', 'value'),
     Input('btn-mamadou', 'n_clicks'),  Input('btn-aissatou', 'n_clicks'),
     Input('btn-fatou', 'n_clicks'),    Input('btn-ibrahima', 'n_clicks'),
     Input('btn-rokhaya', 'n_clicks')]
)
def update_prediction(age, sex, trestbps, chol, thalach, oldpeak,
                      cp, ca, exang, fbs, thal,
                      n1, n2, n3, n4, n5):

    profils_predefinis = {
        'btn-mamadou' : {'age':63,'sex':1,'cp':0,'trestbps':160,'chol':340,'fbs':1,'restecg':2,'thalach':110,'exang':1,'oldpeak':3.5,'slope':0,'ca':3,'thal':3},
        'btn-aissatou': {'age':45,'sex':0,'cp':2,'trestbps':110,'chol':180,'fbs':0,'restecg':0,'thalach':175,'exang':0,'oldpeak':0.2,'slope':2,'ca':0,'thal':2},
        'btn-fatou'   : {'age':65,'sex':0,'cp':1,'trestbps':145,'chol':260,'fbs':1,'restecg':1,'thalach':130,'exang':0,'oldpeak':1.8,'slope':1,'ca':1,'thal':2},
        'btn-ibrahima': {'age':40,'sex':1,'cp':2,'trestbps':115,'chol':195,'fbs':0,'restecg':0,'thalach':180,'exang':0,'oldpeak':0.1,'slope':2,'ca':0,'thal':2},
        'btn-rokhaya' : {'age':35,'sex':0,'cp':3,'trestbps':105,'chol':170,'fbs':0,'restecg':0,'thalach':190,'exang':0,'oldpeak':0.0,'slope':2,'ca':0,'thal':2},
    }

    triggered = callback_context.triggered[0]['prop_id'].split('.')[0] \
                if callback_context.triggered else None

    if triggered in profils_predefinis:
        profil = profils_predefinis[triggered]
    else:
        profil = {
            'age'     : age,      'sex'    : sex,
            'cp'      : cp,       'trestbps': trestbps,
            'chol'    : chol,     'fbs'    : fbs,
            'restecg' : 1,        'thalach': thalach,
            'exang'   : exang,    'oldpeak': oldpeak,
            'slope'   : 1,        'ca'     : ca,
            'thal'    : thal
        }

    patient = pd.DataFrame([profil])
    prob    = best_model.predict_proba(patient)[0][1]
    risque  = (1 - prob) * 100

    if risque >= 70:
        color, icon, statut = COLORS['primary'], '🔴', 'Risque élevé'
    elif risque >= 40:
        color, icon, statut = COLORS['warning'], '🟡', 'Risque modéré'
    else:
        color, icon, statut = COLORS['success'], '✅', 'Faible risque'

    confiance = max(prob, 1 - prob) * 100

    fig_gauge = go.Figure(go.Indicator(
        mode  = 'gauge+number',
        value = risque,
        title = {'text': 'Risque Cardiaque (%)'},
        gauge = {
            'axis'     : {'range': [0, 100]},
            'bar'      : {'color': color},
            'steps'    : [
                {'range': [0,  40], 'color': '#d5f5e3'},
                {'range': [40, 70], 'color': '#fef9e7'},
                {'range': [70, 100],'color': '#fadbd8'},
            ],
            'threshold': {'line': {'color': 'red', 'width': 4}, 'thickness': 0.75, 'value': 70}
        },
        number = {'suffix': '%', 'font': {'size': 36}}
    ))
    fig_gauge.update_layout(
        height=260, margin=dict(t=40, b=0, l=20, r=20), paper_bgcolor='white')

    sexe_label = 'Homme' if profil['sex'] == 1 else 'Femme'

    return html.Div([
        # Badge résultat
        html.Div([
            html.Span(icon, style={'fontSize': '32px'}),
            html.Div([
                html.Div(f"{risque:.1f}% de risque cardiaque",
                         style={'fontSize': '20px', 'fontWeight': '700', 'color': color}),
                html.Div(statut,
                         style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '2px'})
            ], style={'marginLeft': '12px'})
        ], style={'display': 'flex', 'alignItems': 'center',
                  'backgroundColor': COLORS['light'], 'padding': '14px',
                  'borderRadius': '10px', 'marginBottom': '12px',
                  'border': f'2px solid {color}'}),

        # Gauge
        dcc.Graph(figure=fig_gauge, config={'displayModeBar': False}),

        # Confiance
        html.Div(f"Confiance du modèle : {confiance:.1f}%",
                 style={'fontSize': '12px', 'color': COLORS['muted'],
                        'textAlign': 'center', 'marginTop': '4px'}),

        # Résumé patient
        html.Div([
            html.Div(f"👤 {sexe_label}, {profil['age']} ans",
                     style={'fontSize': '12px', 'color': COLORS['text']}),
            html.Div(f"💉 Pression : {profil['trestbps']} mm Hg",
                     style={'fontSize': '12px', 'color': COLORS['text']}),
            html.Div(f"🩸 Cholestérol : {profil['chol']} mg/dl",
                     style={'fontSize': '12px', 'color': COLORS['text']}),
            html.Div(f"❤️ Fréq. cardiaque max : {profil['thalach']} bpm",
                     style={'fontSize': '12px', 'color': COLORS['text']}),
        ], style={'marginTop': '10px', 'lineHeight': '1.9'}),

        # Avertissement
        html.P("⚠️ Estimation probabiliste — pas un diagnostic médical.",
               style={'fontSize': '10px', 'color': COLORS['muted'],
                      'marginTop': '12px', 'fontStyle': 'italic'})
    ])


# ─── LANCEMENT ────────────────────────────────────────────────────────────────
server = app.server

if __name__ == '__main__':
    app.run(debug=False, port=8050)