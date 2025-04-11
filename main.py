import fastf1
fastf1.Cache.enable_cache('cache')  # pour éviter de re-télécharger à chaque fois

session = fastf1.get_session(2023, 'Monza', 'R')  # R = race
session.load()

laps = session.laps
print(laps.head())
