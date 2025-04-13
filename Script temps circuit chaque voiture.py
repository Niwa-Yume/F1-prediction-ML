# ========== 1. Imports & Configuration ==========
import os

import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Créer un dossier 'cache' dans le répertoire du projet
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)

# Activer le cache avec le bon chemin
fastf1.Cache.enable_cache(cache_dir)

# ========== 2. Charger une course ==========


# Charger une course (exemple: Grand Prix de Monaco 2023)
year = 2023
race = 'Monaco'
grand_prix = fastf1.get_session(year, race, 'R')
grand_prix.load()

print(f"Données chargées pour {race} {year}")

# ========== 3. Extraire des données utiles par pilote ==========


# Extraire les données de lap timing
laps = grand_prix.laps

# Ajouter des données de pilote
drivers = pd.unique(laps['Driver'])
print(f"Pilotes en course: {', '.join(drivers)}")

# Enrichir avec des données supplémentaires
laps = laps.loc[:, ['Driver', 'Team', 'LapTime', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus']]
laps = laps.dropna(subset=['LapTime'])  # Supprimer les tours sans temps

# ========== 4. Préparation des données pour le modèle ==========


# Convertir les temps au format numérique (secondes)
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Supprimer la colonne LapTime originale
laps = laps.drop(columns=['LapTime'])

# Encodage des variables catégorielles
data = pd.get_dummies(laps, columns=['Driver', 'Team', 'Compound', 'TrackStatus'])

# Définir X et y
features = [col for col in data.columns if col != 'LapTimeSeconds']
X = data[features]
y = data['LapTimeSeconds']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 5. Entraînement du modèle ==========


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erreur absolue moyenne: {mae:.3f} secondes")

# Importance des caractéristiques
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("Les 10 caractéristiques les plus importantes:")
print(feature_importance)
