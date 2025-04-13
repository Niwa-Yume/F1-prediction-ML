# Pour chaque Grand Prix à analyser:
sessions_à_charger = {
    'FP1': 'Tendances initiales et réglages',
    'FP2': 'Simulations de longues distances',
    'FP3': 'Réglages finaux avant qualifications',
    'Q': 'Performance pure sur un tour',
    'R': 'Données de course complètes'
}
import fastf1
from fastf1 import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuration
fastf1.Cache.enable_cache('cache')

def charger_données_weekend(année, course):
    """Charge toutes les sessions d'un weekend de course"""
    sessions = {}
    for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        try:
            session = fastf1.get_session(année, course, session_type)
            session.load()
            sessions[session_type] = session
            print(f"Session {session_type} chargée pour {course} {année}")
        except Exception as e:
            print(f"Impossible de charger {session_type}: {e}")

    return sessions

def extraire_caractéristiques_qualification(session_q):
    """Extraire les données de performance en qualification"""
    try:
        # Essayer d'abord avec les meilleurs tours de qualification
        best_laps = session_q.laps.pick_fastest()
        quali_data = best_laps[['Driver', 'Position', 'LapTime']]

        # Créer un dictionnaire pour associer pilotes et équipes
        driver_teams = {}
        for _, row in session_q.results.iterrows():
            if 'Abbreviation' in row and 'TeamName' in row:
                driver_teams[row['Abbreviation']] = row['TeamName']

        # Ajouter la colonne TeamName manuellement
        teamnames = []
        for driver in quali_data['Driver']:
            teamnames.append(driver_teams.get(driver, "Unknown"))
        quali_data['TeamName'] = teamnames

        # Renommer les colonnes manuellement
        quali_data_renamed = quali_data.copy()
        quali_data_renamed['GridPosition'] = quali_data['Position']
        quali_data_renamed['BestQualiTime'] = quali_data['LapTime']

        # Suppression des colonnes originales
        quali_data_renamed = quali_data_renamed.drop(columns=['Position', 'LapTime'])

        # Convertir LapTime en secondes en utilisant une boucle
        quali_seconds = []
        for t in quali_data_renamed['BestQualiTime']:
            if hasattr(t, 'total_seconds'):
                quali_seconds.append(t.total_seconds())
            else:
                quali_seconds.append(float(t))

        quali_data_renamed['BestQualiTimeSeconds'] = quali_seconds

        # Calculer l'écart avec la pole
        pole_time = quali_data_renamed['BestQualiTimeSeconds'].min()
        quali_data_renamed['GapToPoleSeconds'] = quali_data_renamed['BestQualiTimeSeconds'] - pole_time

        return quali_data_renamed.drop(columns=['BestQualiTime'], errors='ignore')
    except Exception as e:
        print(f"Utilisation d'un format alternatif pour les résultats officiels: {e}")

        # Créer un DataFrame directement
        data = []
        for _, row in session_q.results.iterrows():
            abbr = row.get('Abbreviation', row.get('DriverId', 'UNK'))
            team = row.get('TeamName', 'Unknown')
            pos = float(row.get('Position', 99))

            data.append({
                'Driver': abbr,
                'TeamName': team,
                'GridPosition': pos,
                'BestQualiTimeSeconds': pos,
                'GapToPoleSeconds': pos - 1
            })

        return pd.DataFrame(data)

def analyser_performances_essais(sessions):
    """Analyser les performances des pilotes pendant les essais libres"""
    fp_data = pd.DataFrame(columns=['Driver'])

    for session_key in ['FP1', 'FP2', 'FP3']:
        if session_key in sessions:
            try:
                # Récupérer les meilleurs tours
                best_laps = sessions[session_key].laps.pick_fastest()

                # Créer un dataframe temporaire
                temp_df = best_laps[['Driver', 'LapTime']]

                # Convertir les timedeltas correctement
                lap_times_seconds = []
                for t in temp_df['LapTime']:
                    if hasattr(t, 'total_seconds'):
                        lap_times_seconds.append(t.total_seconds())
                    else:
                        lap_times_seconds.append(float(t))

                temp_df[f'{session_key}TimeSeconds'] = lap_times_seconds

                # Fusionner avec les données principales
                if len(fp_data) == 0 or 'Driver' not in fp_data.columns:
                    fp_data = temp_df[['Driver', f'{session_key}TimeSeconds']]
                else:
                    fp_data = pd.merge(fp_data,
                                       temp_df[['Driver', f'{session_key}TimeSeconds']],
                                       on='Driver', how='outer')
            except Exception as e:
                print(f"Erreur lors de l'analyse de {session_key}: {e}")

    # Calculer la moyenne des performances en essais libres
    fp_columns = [col for col in fp_data.columns if 'TimeSeconds' in col]
    if fp_columns:
        fp_data['AvgFPPerformance'] = fp_data[fp_columns].mean(axis=1, skipna=True)

    return fp_data

def analyser_historique_circuit(driver, circuit, année_courante):
    """Analyser les performances historiques du pilote sur ce circuit"""
    # Cette fonction nécessiterait de charger les données des années précédentes
    # Implémentation simplifiée pour l'exemple
    historique = {
        'VER': {'Monaco': 0.95},  # Facteur de performance (>1 = bon, <1 = moins bon)
        'HAM': {'Monaco': 0.98},
        # Autres pilotes et circuits...
    }

    # Valeur par défaut si données non disponibles
    return historique.get(driver, {}).get(circuit, 1.0)

# Prédition avant une course du weekend
def créer_dataset_prédiction(année, course):
    """Version simplifiée pour prédire avant une course (sans résultat final)"""
    # 1. Charger seulement les sessions déjà complétées
    sessions = {}
    for session_type in ['FP1', 'FP2', 'FP3', 'Q']:
        try:
            session = fastf1.get_session(année, course, session_type)
            session.load()
            sessions[session_type] = session
            print(f"Session {session_type} chargée pour {course} {année}")
        except Exception as e:
            print(f"Impossible de charger {session_type}: {e}")

    if 'Q' not in sessions:
        print("Données de qualification manquantes")
        return None

    # 2. Extraire les données des qualifications
    quali_data = extraire_caractéristiques_qualification(sessions['Q'])

    # 3. Analyser les performances en essais libres - AJOUT DE CETTE LIGNE MANQUANTE
    fp_data = analyser_performances_essais(sessions)

    # 4. Fusionner les données
    if not fp_data.empty and 'Driver' in fp_data.columns:
        course_data = pd.merge(quali_data, fp_data, on='Driver', how='left')
    else:
        course_data = quali_data.copy()
        course_data['AvgFPPerformance'] = np.nan

    # 5. Ajouter les prévisions météo pour Bahreïn 2025
    weather_predictions = {
        'AirTemp': 25.0,  # Température en °C
        'Humidity': 45.0,  # Humidité en %
        'Rainfall': False,  # Est-ce qu'il va pleuvoir?
        'TrackTemp': 35.0,  # Température de piste
    }

    for key, value in weather_predictions.items():
        course_data[key] = value

    # 6. Ajouter les facteurs historiques du circuit
    course_data['CircuitHistoricalFactor'] = course_data['Driver'].apply(
        lambda x: analyser_historique_circuit(x, course, année)
    )

    return course_data

def créer_dataset_course(année, course, historique=False):
    """Créer un dataset complet pour la prédiction"""
    # 1. Charger toutes les sessions du weekend
    sessions = charger_données_weekend(année, course)

    if 'Q' not in sessions or 'R' not in sessions:
        print("Données de qualification ou de course manquantes")
        return None

    # 2. Extraire les données des qualifications
    quali_data = extraire_caractéristiques_qualification(sessions['Q'])

    # 3. Analyser les performances en essais libres
    fp_data = analyser_performances_essais(sessions)

    # 4. Fusionner quali_data et fp_data
    if not fp_data.empty and 'Driver' in fp_data.columns:
        course_data = pd.merge(quali_data, fp_data, on='Driver', how='left')
    else:
        course_data = quali_data.copy()
        course_data['AvgFPPerformance'] = np.nan

    # 5. Ajouter les données météo
    if 'R' in sessions:
        weather = sessions['R'].weather_data
        # Simplification: on prend la météo moyenne de la course
        avg_weather = {
            'AirTemp': weather['AirTemp'].mean(),
            'Humidity': weather['Humidity'].mean(),
            'Rainfall': weather['Rainfall'].sum() > 0,  # Booléen: a-t-il plu?
            'TrackTemp': weather['TrackTemp'].mean(),
        }

        # Ajouter à chaque ligne
        for key, value in avg_weather.items():
            course_data[key] = value

    # 6. Ajouter les résultats de course (notre variable cible)
    if 'R' in sessions:
        results = sessions['R'].results[['DriverId', 'Position']]
        results = results.rename(columns={'DriverId': 'Driver', 'Position': 'FinalPosition'})
        course_data = pd.merge(course_data, results, on='Driver', how='left')

        # Créer une variable cible binaire pour le vainqueur
        course_data['IsWinner'] = (course_data['FinalPosition'] == 1).astype(int)

    # 7. Ajouter des caractéristiques historiques si demandé
    if historique:
        course_data['CircuitHistoricalFactor'] = course_data['Driver'].apply(
            lambda x: analyser_historique_circuit(x, course, année)
        )

    return course_data

def entraîner_modèle_prédiction(données_historiques):
    """Entraîner un modèle pour prédire le vainqueur"""
    # Préparer les features et la cible
    X = données_historiques.drop(columns=['Driver', 'TeamName', 'FinalPosition', 'IsWinner'])
    y = données_historiques['IsWinner']

    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle de classification avec équilibrage des classes
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  # Compenser le déséquilibre des classes
        max_depth=5,              # Limiter la profondeur pour éviter le surapprentissage
        min_samples_leaf=5        # Exiger plus d'échantillons par feuille
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle: {accuracy:.2f}")

    return model

def prédire_vainqueur(model, données_course):
    """Prédire le vainqueur d'une course"""
    # Préparer les données pour la prédiction
    X_pred = données_course.drop(columns=['Driver', 'TeamName', 'FinalPosition', 'IsWinner'], errors='ignore')

    # Calculer les probabilités de victoire
    probas = model.predict_proba(X_pred)

    # Vérifier la taille de la sortie
    if probas.shape[1] > 1:
        win_probas = probas[:, 1]  # Classe 1 (vainqueur)
    else:
        win_probas = probas[:, 0]  # Une seule classe disponible

    # Ajouter les probabilités au DataFrame
    résultats = données_course[['Driver', 'TeamName']].copy()
    résultats['WinProbability'] = win_probas

    # Normaliser les probabilités pour qu'elles soient plus différenciées
    # Si toutes les probabilités sont identiques, générer des valeurs aléatoires pondérées
    if len(set(win_probas)) <= 1:
        print("Avertissement : Probabilités identiques détectées. Utilisation d'un modèle alternatif.")
        # Utiliser la position de qualification comme facteur de probabilité inverse
        grid_pos = données_course['GridPosition'].values
        # Convertir en probabilités (1/position)
        alt_probas = 1 / grid_pos
        # Normaliser pour avoir une somme = 1
        résultats['WinProbability'] = alt_probas / alt_probas.sum()
    else:
        # Normaliser pour avoir une somme = 1
        résultats['WinProbability'] = résultats['WinProbability'] / résultats['WinProbability'].sum()

    # Trier par probabilité de victoire
    résultats = résultats.sort_values('WinProbability', ascending=False)

    return résultats




# CODE PRINCIPAL - À exécuter pour prédire le GP de Bahreïn 2025
if __name__ == "__main__":
    print("Préparation de la prédiction pour le GP de Bahreïn 2025")

    # Étape 1: Collecter les données des courses précédentes
    courses_historiques = [
        {'année': 2024, 'course': 'Bahrain'},      # GP de Bahreïn 2024
        {'année': 2024, 'course': 'Saudi Arabia'},  # Et d'autres courses de 2024
        {'année': 2024, 'course': 'Australia'},
        {'année': 2024, 'course': 'Japan'},
        # Commence avec une seule course pour tester
    ]

    # Étape 2: Créer un tableau pour stocker les données historiques
    données_complètes = pd.DataFrame()

    # Étape 3: Remplir ce tableau avec les données de chaque course passée
    print("\nChargement des données historiques...")
    for course in courses_historiques:
        print(f"\nTraitement de {course['course']} {course['année']}:")
        données_course = créer_dataset_course(course['année'], course['course'], historique=True)
        if données_course is not None:
            données_complètes = pd.concat([données_complètes, données_course])
            print(f"Données ajoutées: {len(données_course)} lignes")

    print(f"\nTotal des données historiques: {len(données_complètes)} lignes")

    # Étape 4: Entraîner notre modèle
    print("\nEntraînement du modèle de prédiction...")
    modèle = entraîner_modèle_prédiction(données_complètes)

    # Étape 5: Préparer les données de Bahreïn 2025
    print("\nPréparation des données pour Bahreïn 2025...")
    données_bahreïn_2025 = créer_dataset_prédiction(2025, 'Bahrain')

    # Étape 6: Faire la prédiction
    if données_bahreïn_2025 is not None:
        prédictions = prédire_vainqueur(modèle, données_bahreïn_2025)

        # Étape 7: Afficher les résultats
        print("\n=========================================")
        print("PRÉDICTIONS POUR LE GP DE BAHREÏN 2025:")
        print("=========================================")
        for i, (index, row) in enumerate(prédictions.head(10).iterrows()):
            print(
                f"{i + 1}. {row['Driver']} ({row['TeamName']}) - {row['WinProbability'] * 100:.1f}% de chances de gagner")
    else:
        print("Impossible de faire une prédiction: données manquantes pour Bahreïn 2025")
