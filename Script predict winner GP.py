import os
import sys
from datetime import datetime

# ------------------ 1. Gestion des dépendances ------------------

required_packages = [
    'pandas', 'numpy', 'scikit-learn', 'fastf1', 'matplotlib'
]
SKIP_DEPENDENCY_CHECK = True


def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    import importlib.util
    missing = [pkg for pkg in required_packages if not importlib.util.find_spec(pkg)]
    if missing:
        print("Installation des dépendances manquantes :", ', '.join(missing))
        os.system(f"{sys.executable} -m pip install " + ' '.join(missing))
    else:
        print("Toutes les dépendances sont déjà installées.")


# ------------------ 2. Imports ------------------

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration du cache
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


# ------------------ 3. Fonctions utilitaires ------------------

def obtenir_prochain_gp():
    """Retourne le prochain Grand Prix au calendrier"""
    année_actuelle = datetime.now().year

    try:
        schedule = fastf1.get_event_schedule(année_actuelle)
        today = pd.Timestamp.now(tz='UTC')
        next_races = schedule[schedule['Session5Date'] > today]

        if next_races.empty:
            schedule = fastf1.get_event_schedule(année_actuelle + 1)
            next_races = schedule

        if next_races.empty:
            raise ValueError("Aucune course future trouvée")

        next_gp = next_races.iloc[0]
        return next_gp['EventName'], next_gp['Session5Date'].year

    except Exception as e:
        print(f"Erreur lors de la récupération du calendrier: {e}")
        return "Bahrain Grand Prix", année_actuelle + 1


def charger_données_weekend(année, course):
    """Charge les sessions d'un weekend en tenant compte du format sprint"""
    sessions = {}
    année_courante = datetime.now().year

    # Pour les courses futures
    if année > année_courante:
        print(f"GP {course} {année} est dans le futur, utilisation de données simulées")
        return créer_sessions_fictives(course), False

    # Normaliser le nom de la course
    course_name = course.split(' Grand Prix')[0]

    # Détecter si c'est un weekend sprint
    try:
        is_sprint = est_weekend_sprint(année, course_name)
        print(f"Weekend {'Sprint' if is_sprint else 'Standard'} détecté pour {course_name} {année}")

        if is_sprint:
            session_types = {'FP1': 'Practice 1', 'SQ': 'Sprint Qualifying',
                             'S': 'Sprint', 'Q': 'Qualifying', 'R': 'Race'}
        else:
            session_types = {'FP1': 'Practice 1', 'FP2': 'Practice 2',
                             'FP3': 'Practice 3', 'Q': 'Qualifying', 'R': 'Race'}
    except Exception as e:
        print(f"Erreur lors de la détection du format: {e}")
        session_types = {'FP1': 'Practice 1', 'FP2': 'Practice 2',
                         'FP3': 'Practice 3', 'Q': 'Qualifying', 'R': 'Race'}
        is_sprint = False

    # Chargement des sessions
    sessions_réussies = 0
    for abbr, name in session_types.items():
        try:
            session = fastf1.get_session(année, course_name, name)
            session.load(weather=True, laps=True, telemetry=False)
            sessions[abbr] = session
            print(f"Session {abbr} chargée pour {course_name} {année}")
            sessions_réussies += 1
        except Exception as e:
            print(f"Impossible de charger {abbr}: {e}")

    # Si aucune session n'a pu être chargée, utiliser des données fictives
    if sessions_réussies == 0:
        print(f"Aucune session disponible pour {course_name} {année}")
        return créer_sessions_fictives(course), False

    # Si les qualifications n'ont pas pu être chargées, créer une session fictive
    if 'Q' not in sessions:
        print("Session Q manquante, création d'une session fictive")
        fake_sessions = créer_sessions_fictives(course)
        sessions['Q'] = fake_sessions['Q']

    return sessions, is_sprint


def extraire_caractéristiques_qualification(session_q):
    """Extrait les données de qualification de façon robuste"""
    try:
        # Vérifier si les résultats sont disponibles
        if not hasattr(session_q, 'results') or session_q.results is None or len(session_q.results) == 0:
            print("Pas de résultats de qualification disponibles")
            return créer_données_fictives_qualification()

        # Utiliser directement les résultats officiels
        quali_data = session_q.results

        # Vérifier si c'est un DataFrame
        if not isinstance(quali_data, pd.DataFrame):
            print("Format de résultats incompatible")
            return créer_données_fictives_qualification()

        # Vérifier les colonnes minimales requises
        required_cols = ['Abbreviation', 'Position']
        if not all(col in quali_data.columns for col in required_cols):
            print("Format de résultats incomplet")
            return créer_données_fictives_qualification()

        # Créer le DataFrame de base avec les informations essentielles
        basic_data = pd.DataFrame()
        basic_data['Driver'] = quali_data['Abbreviation']
        basic_data['Position_Quali'] = pd.to_numeric(quali_data['Position'], errors='coerce')

        # Ajouter l'équipe si disponible
        if 'TeamName' in quali_data.columns:
            basic_data['Team'] = quali_data['TeamName']
        else:
            basic_data['Team'] = "Inconnue"

        # Essayer d'extraire les temps de qualification
        quali_time_found = False

        for q_session in ['Q3', 'Q2', 'Q1']:
            if q_session in quali_data.columns and not quali_data[q_session].isna().all():
                try:
                    # Convertir les temps de string à secondes
                    quali_times = []

                    for t in quali_data[q_session]:
                        if pd.isna(t):
                            quali_times.append(np.nan)
                        elif isinstance(t, (int, float)):
                            quali_times.append(float(t))
                        elif isinstance(t, str):
                            if ':' in t:  # Format MM:SS.mmm
                                parts = t.split(':')
                                minutes = float(parts[0])
                                seconds = float(parts[1])
                                quali_times.append(minutes * 60 + seconds)
                            else:  # Format SS.mmm
                                quali_times.append(float(t))
                        else:
                            quali_times.append(np.nan)

                    basic_data['QualiTime'] = quali_times
                    quali_time_found = True
                    break
                except Exception as e:
                    print(f"Erreur extraction temps de {q_session}: {e}")

        # Si aucun temps n'a été trouvé, utiliser les positions
        if not quali_time_found:
            basic_data['QualiTime'] = basic_data['Position_Quali'].apply(lambda p: 88.0 + (p - 1) * 0.3)
            print("Approximation des temps par position")

        # Calculer l'écart avec la pole
        valid_times = basic_data['QualiTime'].dropna()
        if not valid_times.empty:
            pole_time = valid_times.min()
            basic_data['GapToPole'] = basic_data['QualiTime'] - pole_time
        else:
            basic_data['GapToPole'] = basic_data['Position_Quali'] - 1

        return basic_data

    except Exception as e:
        print(f"Erreur générale qualification: {e}")
        return créer_données_fictives_qualification()


def get_pilotes_actuels(année=2025):
    """Récupère la liste des pilotes actuels via l'API FastF1"""

    try:

        # Récupérer une course récente

        schedule = fastf1.get_event_schedule(année)

        if schedule.empty and année == datetime.now().year:
            # Si aucune course pour l'année actuelle, essayer l'année précédente

            schedule = fastf1.get_event_schedule(année - 1)

        if schedule.empty:
            raise ValueError(f"Aucune course trouvée pour l'année {année}")

        # Parcourir les courses pour trouver des données valides

        for idx in range(len(schedule) - 1, -1, -1):

            try:

                course = schedule.iloc[idx]

                session = fastf1.get_session(année, course['EventName'], 'Q')

                session.load()

                if hasattr(session, 'results') and session.results is not None:

                    pilotes = []

                    for _, row in session.results.iterrows():

                        if 'Abbreviation' in row and 'TeamName' in row:
                            pilotes.append({

                                'Driver': row['Abbreviation'],

                                'Team': row['TeamName']

                            })

                    if pilotes:
                        return pilotes

            except Exception:

                continue

        raise ValueError("Aucune donnée de qualification valide trouvée")



    except Exception as e:

        print(f"Erreur lors de la récupération des pilotes : {str(e)}")

        # Retourner une liste par défaut des pilotes actuels

        return [

            {'Driver': 'VER', 'Team': 'Red Bull Racing'},

            {'Driver': 'LAW', 'Team': 'Red Bull Racing'},  # Liam Lawson

            {'Driver': 'HAM', 'Team': 'Ferrari'},

            {'Driver': 'LEC', 'Team': 'Ferrari'},

            {'Driver': 'RUS', 'Team': 'Mercedes'},

            {'Driver': 'ANT', 'Team': 'Mercedes'},  # Andrea Kimi Antonelli

            {'Driver': 'NOR', 'Team': 'McLaren'},

            {'Driver': 'PIA', 'Team': 'McLaren'},

            {'Driver': 'ALO', 'Team': 'Aston Martin'},

            {'Driver': 'STR', 'Team': 'Aston Martin'},

            {'Driver': 'GAS', 'Team': 'Alpine'},

            {'Driver': 'DOO', 'Team': 'Alpine'},  # Jack Doohan

            {'Driver': 'ALB', 'Team': 'Williams'},

            {'Driver': 'SAI', 'Team': 'Williams'},  # Carlos Sainz

            {'Driver': 'HUL', 'Team': 'Sauber'},

            {'Driver': 'BOR', 'Team': 'Sauber'},  # Gabriel Bortoleto

            {'Driver': 'TSU', 'Team': 'Racing Bulls'},  # Yuki Tsunoda

            {'Driver': 'HAD', 'Team': 'Racing Bulls'},  # Isack Hadjar

            {'Driver': 'MAG', 'Team': 'Haas'},

            {'Driver': 'OCO', 'Team': 'Haas'},  # Esteban Ocon

        ]


def créer_données_fictives_qualification(avec_temps=True):
    """Crée des données fictives basées sur les pilotes actuels"""
    pilotes = get_pilotes_actuels()

    # Créer le DataFrame de base
    df = pd.DataFrame(pilotes)

    # Ajouter positions de qualification
    for i in range(len(df)):
        df.at[i, 'Position_Quali'] = i + 1

    if avec_temps:
        # Temps de base plus réaliste (environ 1:28)
        base_time = 88.0
        # Écarts plus réalistes entre les pilotes
        df['QualiTime'] = df['Position_Quali'].apply(lambda p: base_time + (p - 1) * 0.3)
        df['GapToPole'] = df['QualiTime'] - df['QualiTime'].min()
    else:
        df['QualiTime'] = np.nan
        df['GapToPole'] = np.nan

    # Ajouter des facteurs historiques par équipe
    team_factors = {
        'Red Bull Racing': 1.1,
        'Mercedes': 1.05,
        'Ferrari': 1.03,
        'McLaren': 1.02,
        'Aston Martin': 1.0,
        'Alpine': 0.98,
        'Williams': 0.97,
        'Sauber': 0.96,
        'RB': 0.95,
        'Haas': 0.94
    }

    df['CircuitHistoricalFactor'] = df['Team'].map(team_factors).fillna(1.0)

    return df


def créer_sessions_fictives(course):
    """Crée des données fictives pour les courses futures"""
    sessions = {}
    pilotes = get_pilotes_actuels()

    # Créer une structure basique pour chaque session
    for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        # Pour les qualifications, créer un DataFrame avec les résultats
        if session_type == 'Q':
            quali_df = pd.DataFrame(pilotes)
            # Ajouter les positions
            for i in range(len(quali_df)):
                quali_df.at[i, 'Position'] = i + 1

            class FakeSession:
                def __init__(self, results_df):
                    self.results = results_df
                    self.weather_data = pd.DataFrame({
                        'AirTemp': [25.0], 'Humidity': [45.0],
                        'Rainfall': [False], 'TrackTemp': [35.0]
                    })
                    self.laps = pd.DataFrame()

            sessions[session_type] = FakeSession(quali_df)
        else:
            # Pour les autres sessions
            class FakeSession:
                def __init__(self):
                    self.results = None
                    self.weather_data = pd.DataFrame({
                        'AirTemp': [25.0], 'Humidity': [45.0],
                        'Rainfall': [False], 'TrackTemp': [35.0]
                    })
                    self.laps = pd.DataFrame()

            sessions[session_type] = FakeSession()

    return sessions

def analyser_performances_essais(sessions, is_sprint=False):
    """Analyse les performances en essais libres avec gestion sprint"""
    fp_data = pd.DataFrame()

    # Sessions à analyser selon le type de weekend
    if is_sprint:
        session_keys = ['FP1', 'SQ', 'S']
    else:
        session_keys = ['FP1', 'FP2', 'FP3']

    for session_key in session_keys:
        if session_key in sessions:
            try:
                session = sessions[session_key]

                # Vérifier si les laps sont bien chargés
                if not hasattr(session, 'laps') or session.laps is None:
                    print(f"Données de tours non disponibles pour {session_key}")
                    continue

                # Vérifier si les données sont vides
                if session.laps.empty:
                    print(f"Aucune donnée de tour pour {session_key}")
                    continue

                # Récupérer les meilleurs tours
                best_laps = session.laps.pick_fastest()

                if isinstance(best_laps, pd.DataFrame) and not best_laps.empty:
                    temp_df = pd.DataFrame()

                    if 'Driver' in best_laps.columns:
                        temp_df['Driver'] = best_laps['Driver'].values
                    else:
                        continue

                    # Convertir les temps avec notre fonction normalisée
                    if 'LapTime' in best_laps.columns:
                        temp_df[f'{session_key}Time'] = best_laps['LapTime'].apply(convertir_temps_en_secondes)

                    # Fusionner avec les données principales
                    if not temp_df.empty:
                        if fp_data.empty:
                            fp_data = temp_df.copy()
                        else:
                            fp_data = pd.merge(fp_data, temp_df, on='Driver', how='outer')
            except Exception as e:
                print(f"Erreur extraction {session_key}: {e}")

    # Calculer la moyenne des temps d'essais
    time_cols = [col for col in fp_data.columns if 'Time' in col]
    if time_cols and not fp_data.empty:
        fp_data['AvgPracticeTime'] = fp_data[time_cols].mean(axis=1, skipna=True)
    else:
        if not fp_data.empty:
            fp_data['AvgPracticeTime'] = np.nan
        else:
            fp_data = pd.DataFrame(columns=['Driver', 'AvgPracticeTime'])

    return fp_data

def analyser_historique_circuit(driver, circuit, année_courante):
    """Analyse les performances historiques du pilote sur ce circuit"""
    # Cette fonction pourrait être enrichie avec des données réelles
    facteurs_historiques = {
        'VER': {'Bahrain': 1.1, 'Monaco': 0.95, 'Saudi Arabia': 1.05},
        'HAM': {'Bahrain': 1.05, 'Monaco': 0.98, 'Great Britain': 1.2},
        'LEC': {'Monaco': 1.1, 'Italy': 1.05},
        # Vous pouvez enrichir cette liste
    }

    # Valeur par défaut si données non disponibles
    return facteurs_historiques.get(driver, {}).get(circuit, 1.0)

def convertir_temps_en_secondes(temps):
    """Convertit un temps de tour en secondes quelle que soit sa forme"""
    if pd.isna(temps):
        return np.nan

    # Si c'est déjà un nombre
    if isinstance(temps, (float, int)):
        return float(temps)

    # Si c'est un objet timedelta
    if hasattr(temps, 'total_seconds'):
        return temps.total_seconds()

    # Si c'est une chaîne de caractères
    if isinstance(temps, str):
        try:
            if ":" in temps:  # Format MM:SS.mmm
                minutes, seconds = temps.split(":")
                return float(minutes) * 60 + float(seconds)
            else:  # Format SS.mmm
                return float(temps)
        except (ValueError, TypeError):
            return np.nan

    return np.nan

def valider_données_course(données_course):
    """Valide et complète les données manquantes"""
    if données_course is None or données_course.empty:
        print("ATTENTION: Aucune donnée disponible pour la course")
        return créer_données_fictives_qualification()

    colonnes_requises = ['Driver', 'Team', 'Position_Quali', 'CircuitHistoricalFactor']
    colonnes_manquantes = [col for col in colonnes_requises if col not in données_course.columns]

    if colonnes_manquantes:
        print(f"ATTENTION: Colonnes manquantes: {', '.join(colonnes_manquantes)}")
        données_fictives = créer_données_fictives_qualification()

        # Fusionner avec les données existantes
        if not données_course.empty:
            données_course = pd.merge(
                données_course,
                données_fictives[colonnes_manquantes + ['Driver']],
                on='Driver',
                how='right'
            )
        else:
            données_course = données_fictives

    return données_course

def filtrer_pilotes_valides(data, colonnes_requises=None):
    """Filtre pour ne garder que les pilotes avec données valides"""
    if colonnes_requises is None:
        colonnes_requises = ['Position_Quali']

    mask = pd.Series(True, index=data.index)
    for col in colonnes_requises:
        if col in data.columns:
            mask = mask & data[col].notna()

    filtered_data = data[mask].copy()

    if len(filtered_data) < len(data):
        exclus = len(data) - len(filtered_data)
        print(f"{exclus} pilote(s) exclu(s) pour données manquantes")

    return filtered_data

def créer_dataset_course(année, course, historique=True, sessions=None):
    """Crée un ensemble de données complet pour une course donnée"""
    print(f"Création du dataset pour {course} {année}...")

    # Utiliser toutes les sessions si non spécifié
    if sessions is None:
        sessions = ['FP1', 'FP2', 'FP3', 'Q', 'R']

    # Normaliser le nom du circuit pour les facteurs historiques
    nom_circuit = course.split(' Grand Prix')[0]

    # Initialiser sessions_chargées avant tout bloc conditionnel
    sessions_chargées = {}

    # Vérifier si c'est un weekend sprint
    is_sprint = False
    try:
        is_sprint = est_weekend_sprint(année, nom_circuit)
        print(f"Format du weekend: {'Sprint' if is_sprint else 'Standard'}")

        # Adapter les sessions à charger selon le format
        if is_sprint and sessions == ['FP1', 'FP2', 'FP3', 'Q', 'R']:
            sessions = ['FP1', 'SQ', 'S', 'Q', 'R']
            print("Format sprint détecté: adaptation des sessions à charger")
    except Exception as e:
        print(f"Erreur lors de la vérification du format du weekend: {e}")

    # Vérifier si la course est dans le futur
    est_course_future = année > datetime.now().year
    if est_course_future:
        print(f"\nGP {course} {année} est une course future")
        print("Utilisation de données simulées basées sur les performances récentes")
        return créer_données_fictives_qualification()

    # Charger uniquement les sessions demandées (un seul bloc)
    for session_type in sessions:
        if session_type in ['FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q', 'R']:
            try:
                # Map des noms de sessions corrigé pour FastF1
                session_map = {
                    'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
                    'SQ': 'Sprint Qualifying', 'S': 'Sprint', 'Q': 'Qualifying', 'R': 'Race'
                }
                session = fastf1.get_session(année, nom_circuit, session_map[session_type])
                session.load(weather=True, laps=True, telemetry=False)
                sessions_chargées[session_type] = session
                print(f"Session {session_type} chargée pour {nom_circuit} {année}")
            except Exception as e:
                print(f"Impossible de charger {session_type}: {e}")

    # Utiliser toutes les sessions si non spécifié
    if sessions is None:
        sessions = ['FP1', 'FP2', 'FP3', 'Q', 'R']

    # Normaliser le nom du circuit pour les facteurs historiques
    nom_circuit = course.split(' Grand Prix')[0]

    # Initialiser sessions_chargées ici, avant les blocs try/except
    sessions_chargées = {}

    # Vérifier si c'est un weekend sprint
    is_sprint = False
    try:
        is_sprint = est_weekend_sprint(année, nom_circuit)
        print(f"Format du weekend: {'Sprint' if is_sprint else 'Standard'}")

        # Adapter les sessions à charger selon le format
        if is_sprint and sessions == ['FP1', 'FP2', 'FP3', 'Q', 'R']:
            sessions = ['FP1', 'SQ', 'S', 'Q', 'R']
            print("Format sprint détecté: adaptation des sessions à charger")
    except Exception as e:
        print(f"Erreur lors de la vérification du format du weekend: {e}")

    # Charger uniquement les sessions demandées
    for session_type in sessions:
        if session_type in ['FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q', 'R']:
            try:
                # Map des noms de sessions pour FastF1
                session_map = {
                    'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
                    'SQ': 'Sprint Qualifying', 'S': 'Sprint', 'Q': 'Qualifying', 'R': 'Race'
                }

                session = fastf1.get_session(année, nom_circuit, session_map[session_type])
                session.load(weather=True, laps=True, telemetry=False)
                sessions_chargées[session_type] = session
                print(f"Session {session_type} chargée pour {nom_circuit} {année}")
            except Exception as e:
                print(f"Impossible de charger {session_type}: {e}")

    # Charger uniquement les sessions demandées (bloc déplacé hors du except)
    sessions_chargées = {}
    for session_type in sessions:
        if session_type in ['FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q', 'R']:
            try:
                # Map des noms de sessions corrigé pour FastF1
                session_map = {
                    'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
                    'SQ': 'Sprint Qualifying', 'S': 'Sprint', 'Q': 'Qualifying', 'R': 'Race'
                }
                session = fastf1.get_session(année, nom_circuit, session_map[session_type])
                session.load(weather=True, laps=True, telemetry=False)
                sessions_chargées[session_type] = session
                print(f"Session {session_type} chargée pour {nom_circuit} {année}")
            except Exception as e:
                print(f"Impossible de charger {session_type}: {e}")


    # Vérifier si les qualifications sont disponibles (essentiel)
    if 'Q' not in sessions_chargées:
        print(f"Données de qualification non disponibles pour {nom_circuit} {année}")
        return None

    # Données de qualification
    quali_data = extraire_caractéristiques_qualification(sessions_chargées['Q'])
    if quali_data.empty or 'Driver' not in quali_data.columns:
        print("Pas de données de qualification valides")
        return None

    # Base de données pour la course
    course_data = quali_data.copy()

    # Ajouter les positions de départ si disponibles (peuvent être différentes des qualifications)
    if 'R' in sessions_chargées:
        try:
            race_results = sessions_chargées['R'].results
            if race_results is not None and 'GridPosition' in race_results.columns:
                grid_positions = race_results[['Abbreviation', 'GridPosition']]
                grid_positions = grid_positions.rename(columns={
                    'Abbreviation': 'Driver', 'GridPosition': 'StartingPosition'
                })
                course_data = pd.merge(course_data, grid_positions, on='Driver', how='left')
                course_data['GridDelta'] = course_data['Position_Quali'] - course_data['StartingPosition']
        except Exception as e:
            print(f"Erreur lors de l'extraction des positions de départ: {e}")

    # Ajouter données d'essais libres
    if is_sprint:
        fp_sessions = {k: v for k, v in sessions_chargées.items() if k in ['FP1', 'SQ', 'S']}
        sprint_data = None

        # Extraire données du sprint si disponible
        if 'S' in sessions_chargées:
            try:
                sprint_results = sessions_chargées['S'].results
                if sprint_results is not None:
                    sprint_positions = sprint_results[['Abbreviation', 'Position']]
                    sprint_positions = sprint_positions.rename(columns={
                        'Abbreviation': 'Driver', 'Position': 'SprintPosition'
                    })
                    course_data = pd.merge(course_data, sprint_positions, on='Driver', how='left')

                    # Calcul de performance dans le sprint
                    course_data['SprintPerformance'] = course_data['Position_Quali'] - course_data['SprintPosition']
            except Exception as e:
                print(f"Erreur lors de l'extraction des résultats du sprint: {e}")
    else:
        fp_sessions = {k: v for k, v in sessions_chargées.items() if k in ['FP1', 'FP2', 'FP3']}

    if fp_sessions:
        fp_data = analyser_performances_essais(fp_sessions, is_sprint)
        if not fp_data.empty and 'Driver' in fp_data.columns:
            course_data = pd.merge(course_data,
                                   fp_data[['Driver'] + [c for c in fp_data.columns if c != 'Driver']],
                                   on='Driver',
                                   how='left')
    else:
        course_data['AvgPracticeTime'] = np.nan

    # Ajouter facteurs historiques du circuit
    course_data['CircuitHistoricalFactor'] = course_data['Driver'].apply(
        lambda x: analyser_historique_circuit(x, nom_circuit, année)
    )

    # Ajouter données météo détaillées
    if 'Q' in sessions_chargées:
        try:
            quali_weather = sessions_chargées['Q'].weather_data
            if quali_weather is not None and not quali_weather.empty:
                course_data['QualiTrackTemp'] = quali_weather['TrackTemp'].median()
                course_data['QualiAirTemp'] = quali_weather['AirTemp'].median()
                course_data['QualiHumidity'] = quali_weather['Humidity'].median()
                course_data['QualiRainfall'] = 1 if quali_weather['Rainfall'].any() else 0
        except Exception as e:
            print(f"Erreur lors du traitement des données météo des qualifications: {e}")

    if 'R' in sessions_chargées:
        try:
            race_weather = sessions_chargées['R'].weather_data
            if race_weather is not None and not race_weather.empty:
                course_data['RaceTrackTemp'] = race_weather['TrackTemp'].median()
                course_data['RaceAirTemp'] = race_weather['AirTemp'].median()
                course_data['RaceHumidity'] = race_weather['Humidity'].median()
                course_data['RaceRainfall'] = 1 if race_weather['Rainfall'].any() else 0

                # Différences de température entre quali et course
                if 'QualiTrackTemp' in course_data.columns:
                    course_data['TempDelta'] = course_data['RaceTrackTemp'] - course_data['QualiTrackTemp']
        except Exception as e:
            print(f"Erreur lors du traitement des données météo de la course: {e}")

    # Ajouter résultats de course pour l'entraînement si demandé
    if 'R' in sessions_chargées and historique:
        try:
            results_df = pd.DataFrame(sessions_chargées['R'].results)
            if 'Abbreviation' in results_df.columns and 'Position' in results_df.columns:
                results = results_df[['Abbreviation', 'Position']]
                results = results.rename(columns={'Abbreviation': 'Driver'})
                results['Position_Course'] = pd.to_numeric(results['Position'], errors='coerce')

                # Calculer points gagnés/perdus par rapport à la qualification
                course_data = pd.merge(course_data,
                                       results[['Driver', 'Position_Course']],
                                       on='Driver',
                                       how='left')

                course_data['PositionDelta'] = course_data['Position_Quali'] - course_data['Position_Course']
                course_data['Winner'] = (course_data['Position_Course'] == 1).astype(int)

                # Ajouter une information sur le podium (top 3)
                course_data['Podium'] = (course_data['Position_Course'] <= 3).astype(int)

                # Ajouter les points marqués si disponibles
                if 'Points' in results_df.columns:
                    points_data = results_df[['Abbreviation', 'Points']]
                    points_data = points_data.rename(columns={'Abbreviation': 'Driver'})
                    course_data = pd.merge(course_data,
                                           points_data[['Driver', 'Points']],
                                           on='Driver',
                                           how='left')
            else:
                print("Format des résultats inattendu")
                course_data['Winner'] = 0
        except Exception as e:
            print(f"Erreur lors du traitement des résultats: {e}")
            course_data['Winner'] = 0

    # Filtrer les pilotes avec données valides
    course_data = filtrer_pilotes_valides(course_data)

    print(f"Dataset créé avec succès: {len(course_data)} pilotes et {len(course_data.columns)} caractéristiques")

    return course_data

def entraîner_modèle(données_historiques, n_splits=5):
    """Entraîne le modèle de prédiction du vainqueur de course"""
    print(f"Données disponibles: {données_historiques.shape[0]} entrées")

    # Vérifier les colonnes essentielles
    required_features = ['Position_Quali', 'Winner']
    missing_cols = [col for col in required_features if col not in données_historiques.columns]
    if missing_cols:
        print(f"Colonnes manquantes: {', '.join(missing_cols)}")
        raise ValueError("Données insuffisantes pour entraîner le modèle")

    # Sélection des caractéristiques
    base_features = ['Position_Quali', 'CircuitHistoricalFactor']
    optional_features = [
        'QualiTime', 'GapToPole', 'AvgPracticeTime',
        'TrackTemp', 'AirTemp', 'Humidity', 'Rainfall'
    ]

    # Utiliser uniquement les caractéristiques disponibles
    feature_cols = [f for f in base_features if f in données_historiques.columns]
    for f in optional_features:
        if f in données_historiques.columns and not données_historiques[f].isna().all():
            feature_cols.append(f)

    print(f"Caractéristiques utilisées: {', '.join(feature_cols)}")

    # Préparation des données
    X = données_historiques[feature_cols].fillna(-1)
    y = données_historiques['Winner']

    # Division entraînement/test pour évaluation (on ignore n_splits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Entraînement avec paramètres optimisés
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Évaluation rapide
    X_test_scaled = scaler.transform(X_test)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Score du modèle sur données de test: {test_score:.2f}")

    # Afficher l'importance des caractéristiques
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nImportance des caractéristiques:")
    for _, row in feature_importance.iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")

    return model, scaler, feature_cols

def est_weekend_sprint(année, course):
    """Détermine si un weekend est au format sprint"""
    try:
        # Vérifier si la session Sprint existe
        session = fastf1.get_session(année, course, 'S')
        session.load(weather=False, messages=False)
        return True
    except Exception:
        return False

def prédire_vainqueur(model, scaler, feature_cols, données_course):
    """Prédit le vainqueur potentiel d'une course"""
    # Valider et compléter les données
    données_course = valider_données_course(données_course)

    # Vérifier que toutes les colonnes nécessaires sont disponibles
    missing_cols = [col for col in feature_cols if col not in données_course.columns]
    if missing_cols:
        print(f"Attention: colonnes manquantes: {', '.join(missing_cols)}")
        for col in missing_cols:
            données_course[col] = -1

    # Préparation des données
    X_pred = données_course[feature_cols].fillna(-1)
    X_scaled = scaler.transform(X_pred)

    # Prédiction des probabilités
    probas = model.predict_proba(X_scaled)

    # Ajustements selon position de qualification et facteur historique
    position_factor = 1 / (données_course['Position_Quali'] ** 0.7)
    circuit_factor = données_course['CircuitHistoricalFactor']

    # Calculer les probabilités ajustées
    if probas.shape[1] > 1:
        adjusted_probas = probas[:, 1] * position_factor * circuit_factor
    else:
        adjusted_probas = probas[:, 0] * position_factor * circuit_factor

    # Normaliser pour obtenir des probabilités en pourcentage
    total_proba = adjusted_probas.sum()
    if total_proba > 0:
        adjusted_probas = adjusted_probas / total_proba * 100

    # Création du DataFrame final avec tous les pilotes
    résultats = pd.DataFrame({
        'Driver': données_course['Driver'],
        'Team': données_course['Team'],
        'WinProbability': adjusted_probas
    })

    # S'assurer d'avoir au moins 20 pilotes
    if len(résultats) < 20:
        données_manquantes = créer_données_fictives_qualification()
        pilotes_manquants = set(données_manquantes['Driver']) - set(résultats['Driver'])

        if pilotes_manquants:
            données_compl = données_manquantes[
                données_manquantes['Driver'].isin(pilotes_manquants)
            ].copy()

            # Attribuer des probabilités faibles aux pilotes manquants
            données_compl['WinProbability'] = 0.1
            résultats = pd.concat([résultats, données_compl[['Driver', 'Team', 'WinProbability']]])

            # Renormaliser les probabilités
            total = résultats['WinProbability'].sum()
            résultats['WinProbability'] = résultats['WinProbability'] / total * 100

    return résultats.sort_values('WinProbability', ascending=False)


# ------------------ 4. Interface utilisateur ------------------

def afficher_prédictions(prédictions, course, année):
    """Affiche les résultats de manière formatée"""
    print("\n=========================================")
    print(f"PRÉDICTIONS POUR LE GP DE {course.upper()} {année}:")
    print("=========================================")

    for i, (_, row) in enumerate(prédictions.iterrows(), 1):
        print(f"{i}. {row['Driver']} ({row['Team']}) - {row['WinProbability']:.1f}% de chances de gagner")

def sélectionner_courses_entraînement():
    """Interface pour sélectionner les courses et sessions d'entraînement"""
    print("\nSélection des courses d'entraînement")
    print("====================================")

    année_actuelle = datetime.now().year
    try:
        # Récupérer le calendrier actuel
        schedule = fastf1.get_event_schedule(année_actuelle)

        # Afficher toutes les courses disponibles
        print("\nCourses disponibles :")
        for idx, (_, race) in enumerate(schedule.iterrows(), 1):
            print(f"{idx}. {race['EventName']} ({race['EventDate'].strftime('%d/%m/%Y')})")

        print("\nEntrez les numéros des courses à utiliser (séparés par des espaces)")
        print("Par exemple: 1 2 3 pour les trois premières courses")
        print("Appuyez sur Entrée pour utiliser toutes les courses passées")

        choix = input("\nVotre choix : ").strip()

        if not choix:
            # Utiliser toutes les courses passées
            today = pd.Timestamp.now(tz='UTC')
            past_races = schedule[schedule['Session5Date'] < today]
            courses_selectionnées = past_races['EventName'].tolist()
        else:
            try:
                indices = [int(x) - 1 for x in choix.split()]
                courses_selectionnées = [
                    schedule.iloc[i]['EventName']
                    for i in indices
                    if 0 <= i < len(schedule)
                ]
            except ValueError:
                print("Sélection invalide. Utilisation des courses passées par défaut.")
                today = pd.Timestamp.now(tz='UTC')
                past_races = schedule[schedule['Session5Date'] < today]
                courses_selectionnées = past_races['EventName'].tolist()

        # Sélection des sessions à inclure
        print("\nQuelles sessions souhaitez-vous inclure ?")
        print("1. Toutes les sessions (FP1, FP2, FP3, Qualifications, Course)")
        print("2. Essais libres et Qualifications uniquement (FP1, FP2, FP3, Q)")
        print("3. Qualifications uniquement (Q)")
        print("4. Sélection personnalisée")

        choix_sessions = input("\nVotre choix (1-4) : ").strip()

        sessions_sélectionnées = []
        if choix_sessions == '1':
            sessions_sélectionnées = ['FP1', 'FP2', 'FP3', 'Q', 'R']
        elif choix_sessions == '2':
            sessions_sélectionnées = ['FP1', 'FP2', 'FP3', 'Q']
        elif choix_sessions == '3':
            sessions_sélectionnées = ['Q']
        elif choix_sessions == '4':
            print("\nSélectionnez les sessions (répondez O/N pour chaque) :")
            if input("FP1 (O/N) : ").upper().startswith('O'):
                sessions_sélectionnées.append('FP1')
            if input("FP2 (O/N) : ").upper().startswith('O'):
                sessions_sélectionnées.append('FP2')
            if input("FP3 (O/N) : ").upper().startswith('O'):
                sessions_sélectionnées.append('FP3')
            if input("Qualifications (O/N) : ").upper().startswith('O'):
                sessions_sélectionnées.append('Q')
            if input("Course (O/N) : ").upper().startswith('O'):
                sessions_sélectionnées.append('R')
        else:
            print("Choix invalide. Utilisation de toutes les sessions par défaut.")
            sessions_sélectionnées = ['FP1', 'FP2', 'FP3', 'Q', 'R']

        # Demander si l'année précédente doit être incluse
        inclure_année_précédente = input(
            "\nInclure également les données de l'année précédente ? (O/N) : ").upper().startswith('O')

        return courses_selectionnées, sessions_sélectionnées, inclure_année_précédente

    except Exception as e:
        print(f"Erreur lors de la récupération du calendrier: {e}")
        # Valeurs par défaut au cas où
        return ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix"], ['Q', 'R'], False

def main():
    # Obtenir le prochain GP par défaut
    default_gp, default_year = obtenir_prochain_gp()

    print(f"\nProchain GP: {default_gp} {default_year}")
    choix = input("Appuyez sur Entrée pour ce GP ou entrez un autre (ex: 'Bahrain 2024'): ")

    if choix.strip():
        try:
            course, année = choix.rsplit(' ', 1)
            année = int(année)
            if année < 2000 or année > datetime.now().year + 1:
                raise ValueError("Année non valide")
        except ValueError as e:
            print(f"Format invalide: {e}. Utilisation du GP par défaut.")
            course, année = default_gp, default_year
    else:
        course, année = default_gp, default_year

    # Sélection des courses et sessions d'entraînement
    courses_selectionnées, sessions_selectionnées, inclure_année_précédente = sélectionner_courses_entraînement()
    print(f"\nCourses sélectionnées: {', '.join(courses_selectionnées)}")
    print(f"Sessions incluses: {', '.join(sessions_selectionnées)}")

    # Créer le dataset d'entraînement
    print("\nChargement des données historiques...")
    données_historiques = pd.DataFrame()

    # Traiter l'année en cours
    for gp in courses_selectionnées:
        # Ne pas inclure le GP à prédire dans les données d'entraînement
        if gp == course:
            continue

        print(f"\nTraitement de {gp} {année}...")
        data = créer_dataset_course(année, gp, historique=True, sessions=sessions_selectionnées)
        if data is not None and not data.empty:
            données_historiques = pd.concat([données_historiques, data])
            print(f"Ajouté {len(data)} lignes de données pour {gp} {année}")

    # Si demandé, ajouter les données de l'année précédente
    if inclure_année_précédente:
        année_précédente = année - 1
        print(f"\nAjout des données de {année_précédente}...")

        try:
            schedule_précédent = fastf1.get_event_schedule(année_précédente)
            gps_précédents = schedule_précédent['EventName'].tolist()

            for gp in gps_précédents:
                print(f"\nTraitement de {gp} {année_précédente}...")
                data = créer_dataset_course(année_précédente, gp, historique=True, sessions=sessions_selectionnées)
                if data is not None and not data.empty:
                    données_historiques = pd.concat([données_historiques, data])
                    print(f"Ajouté {len(data)} lignes de données pour {gp} {année_précédente}")
        except Exception as e:
            print(f"Erreur lors de la récupération des données de {année_précédente}: {e}")

    if données_historiques.empty:
        print("Pas assez de données historiques pour l'entraînement.")
        return

    print(
        f"\nTotal des données d'entraînement: {données_historiques.shape[0]} lignes avec {données_historiques.shape[1]} caractéristiques")

    # Validation croisée temporelle
    print("\nConfiguration de la validation croisée temporelle...")
    n_splits = min(5, données_historiques['EventName'].nunique() if 'EventName' in données_historiques.columns else 3)
    print(f"Utilisation de {n_splits} partitions temporelles pour la validation")

    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    try:
        model, scaler, feature_cols = entraîner_modèle(données_historiques, n_splits=n_splits)

        # Prédire pour la course cible
        print(f"\nPrédiction pour {course} {année}...")

        # Pour la prédiction, utiliser les sessions disponibles uniquement (pas la course)
        sessions_prédiction = [s for s in sessions_selectionnées if s != 'R']
        données_course = créer_dataset_course(année, course, historique=False, sessions=sessions_prédiction)

        if données_course is None or données_course.empty:
            print("Impossible d'obtenir les données pour ce GP.")
            return

        prédictions = prédire_vainqueur(model, scaler, feature_cols, données_course)
        afficher_prédictions(prédictions, course.split(' Grand Prix')[0], année)

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not SKIP_DEPENDENCY_CHECK:
        check_dependencies()
    main()