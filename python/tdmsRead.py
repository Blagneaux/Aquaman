import matplotlib.pyplot as plt
from nptdms import TdmsFile
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askfloat
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
 
# Initialiser Tkinter et cacher la fenêtre principale
root = tk.Tk()
root.withdraw()

# Open the simulated pressure file
# digital_twin_pressure = pd.read_csv('C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_JCGE.csv', header=None)
digital_twin_pressure = pd.read_csv('C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_test.csv', header=None)

# Ouvrir la boîte de dialogue pour choisir un fichier et obtenir le chemin du fichier
chemin_fichier = askopenfilename(title="Sélectionnez un fichier TDMS", filetypes=[("Fichiers TDMS", "*.tdms")])
 
# Vérifier si un fichier a été sélectionné
if chemin_fichier:
    # Lire le fichier TDMS
    tdms_file = TdmsFile.read(chemin_fichier)
 
    # Demander à l'utilisateur la fréquence de coupure pour le filtre passe-bas via une boîte de dialogue
    fc = askfloat("Fréquence de coupure", "Veuillez entrer la fréquence de coupure pour le filtre passe-bas (en Hz):",
                  parent=root)
 
    if fc:  # Continuer seulement si une fréquence de coupure a été fournie
        # Supposons une fréquence d'échantillonnage fixe pour tous les canaux
        fe = 500  # Fréquence d'échantillonnage en Hz, ajustez selon vos données
 
        # Compter le nombre total de canaux pour dimensionner la grille de subplots
        nombre_canaux = sum(len(groupe.channels()) for groupe in tdms_file.groups())
        nrows = int(np.ceil(np.sqrt(nombre_canaux)))
        ncols = nrows
 
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
        axs = axs.flatten()  # Aplatir le tableau d'axes pour un accès plus facile
        canal_idx = 0
 
        # Itérer sur tous les groupes et canaux du fichier TDMS
        for groupe in tdms_file.groups():
            for canal in groupe.channels():
                if canal.name in ['S2', 'S4', 'S7']:
                    donnees = canal.data
                    if canal.name == 'S4':
                        dt_data = [digital_twin_pressure[i][210*128+39] for i in digital_twin_pressure.columns]
                    if canal.name == 'S2':
                        dt_data = [digital_twin_pressure[i][164*128+127-39] for i in digital_twin_pressure.columns]
                    if canal.name == 'S7':
                        dt_data = [-digital_twin_pressure[i][105*128+39] for i in digital_twin_pressure.columns]
    
                    # Générer une séquence de temps si possible
                    if hasattr(canal, 'time_track'):
                        temps = canal.time_track()
                    else:
                        temps = [i / fe for i in range(len(donnees))]
                    dt_time = np.linspace(19.5, 21.75, len(dt_data))


                    temps = temps[int(19.5*500):int(21.75*500)]
    
                    # Créer un filtre passe-bas
                    ordre = 2  # Ordre du filtre
                    nyquist = 0.5 * fe
                    frequence_normale = fc / nyquist
                    b, a = butter(ordre, frequence_normale, btype='low', analog=False)
    
                    # Appliquer le filtre
                    donnees_filtrees = filtfilt(b, a, donnees)
                    donnees_filtrees = donnees_filtrees[int(19.5*500):int(21.75*500)]
                    mean_donnees_filtrees = np.mean(donnees_filtrees)
    
                    # Configurer le graphique
                    axs[canal_idx].plot(temps, donnees_filtrees - mean_donnees_filtrees)
                    axs[canal_idx].plot(dt_time, [k * 1025 * 0.0275 * 0.0275 for k in dt_data])
                    axs[canal_idx].set_title(f'{groupe.name}\n{canal.name} (Filtré)')
                    axs[canal_idx].set_xlabel('Temps (s)')
                    axs[canal_idx].set_ylabel('Amplitude')
                    axs[canal_idx].grid(True)
                
                    canal_idx += 1
 
        # Cacher les axes non utilisés si le nombre de canaux n'est pas un carré parfait
        for idx in range(canal_idx, len(axs)):
            axs[idx].set_visible(False)
 
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune fréquence de coupure n'a été fournie.")
else:
    print("Aucun fichier sélectionné.")