import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, simpledialog
import tkinter as tk
from scipy.signal import butter, lfilter

def apply_lowpass_filter(data, order, cutoff_freq, fs):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = pd.DataFrame(columns=data.columns)
    for col in data.columns:
        filtered_data[col] = lfilter(b, a, data[col])
    return filtered_data

# Créer une fenêtre Tkinter (invisible)
root = tk.Tk()
root.withdraw()

# Ouvrir une boîte de dialogue pour sélectionner le fichier Excel
excel_file = filedialog.askopenfilename(title="Sélectionnez le fichier Excel", filetypes=[("Fichiers Excel", "*.xlsx")])

# Demander à l'utilisateur les paramètres du filtre
order = simpledialog.askinteger("Ordre du filtre", "Entrez l'ordre du filtre (entier) :", initialvalue=5)
cutoff_freq = simpledialog.askfloat("Fréquence de coupure", "Entrez la fréquence de coupure (Hz) :", initialvalue=100)

try:
    # Lire le fichier Excel sans spécifier de nom de colonnes
    df = pd.read_excel(excel_file, header=None)

    # Extraire les données de capteur, en excluant la colonne de temps
    donnees = df.iloc[:, :-1]

    # Extraire la colonne du temps en millisecondes
    timer = df.iloc[:, -1]

    # Appliquer le filtre aux données de capteur
    fs = 500  # Fréquence d'échantillonnage en Hz
    filtered_data = apply_lowpass_filter(donnees, order, cutoff_freq, fs)

    # Adapter les données de pression entre -500 et 500 Pascal
    min_binary_value = -32768
    max_binary_value = 32768
    min_pressure_value = -500
    max_pressure_value = 500
    scaled_donnees = (donnees - min_binary_value) * (max_pressure_value - min_pressure_value) / (max_binary_value - min_binary_value) + min_pressure_value
    scaled_filtered_data = (filtered_data - min_binary_value) * (max_pressure_value - min_pressure_value) / (max_binary_value - min_binary_value) + min_pressure_value

    # Créer un graphique à partir des données avant et après le filtrage
    plt.figure(figsize=(10, 6))

    # Tracer les données brutes
    for i, col in enumerate(scaled_donnees.columns):
        plt.plot(timer, scaled_donnees[col], label=f'Données brutes - Capteur {i+1}')

    # Tracer les données filtrées
    for i, col in enumerate(scaled_filtered_data.columns):
        plt.plot(timer, scaled_filtered_data[col], label=f'Données filtrées - Capteur {i+1}')

    plt.xlabel('Temps (ms)')
    plt.ylabel('Valeurs (Pascal)')
    plt.title('Graphique des données avant et après filtrage')
    plt.legend()
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("Le fichier spécifié est introuvable.")
except Exception as e:
    print("Une erreur s'est produite :", e)
