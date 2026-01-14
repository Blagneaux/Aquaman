import matplotlib.pyplot as plt
from nptdms import TdmsFile
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askfloat
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import splev, splrep
import pandas as pd
import similaritymeasures

# --------------------------------------------------------

# From Corentin Porcon

# --------------------------------------------------------
 
# Initialiser Tkinter et cacher la fenêtre principale
root = tk.Tk()
root.withdraw()

# Open the simulated pressure file
# digital_twin_pressure = pd.read_csv('C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_JCGE.csv', header=None)
digital_twin_pressure = pd.read_csv('E:/data_HAACHAMA/pressure_map.csv', header=None)
digital_twin_pressure_REF = pd.read_csv('C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_test_REF.csv', header=None)

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
                        dt_data_REF = [digital_twin_pressure_REF[i][210*128+39] for i in digital_twin_pressure_REF.columns]
                    if canal.name == 'S2':
                        dt_data = [digital_twin_pressure[i][164*128+39+42] for i in digital_twin_pressure.columns]
                        dt_data_REF = [digital_twin_pressure_REF[i][164*128+39+42] for i in digital_twin_pressure_REF.columns]
                    if canal.name == 'S7':
                        dt_data = [-digital_twin_pressure[i][105*128+39] for i in digital_twin_pressure.columns]
                        dt_data_REF = [-digital_twin_pressure_REF[i][105*128+39] for i in digital_twin_pressure_REF.columns]
    
                    # Générer une séquence de temps si possible
                    if hasattr(canal, 'time_track'):
                        temps = canal.time_track()
                    else:
                        temps = [i / fe for i in range(len(donnees))]
                    dt_time = np.linspace(19.5, 21.75, len(dt_data))
                    dt_time_REF = np.linspace(19.5, 21.75, len(dt_data_REF))


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
                    axs[canal_idx].plot(temps, donnees_filtrees - mean_donnees_filtrees, label='Sensors')
                    # axs[canal_idx].plot(dt_time[2:], [k * 1025 * 0.0275 * 0.0275 for k in dt_data[2:]], label='Automated DT')
                    axs[canal_idx].plot(dt_time_REF[2:], [k * 1025 * 0.0275 * 0.0275 for k in dt_data_REF[2:]], label='Manual DT')
                    axs[canal_idx].set_title(f'{canal.name}')
                    axs[canal_idx].set_xlabel('Time (s)')
                    axs[canal_idx].set_ylabel('Pressure (Pa)')
                    axs[canal_idx].grid(True)
                    axs[canal_idx].legend()
                    
                    # Calcul la distance de Fréchet discrète
                    exp_data = np.zeros((len(temps),2))
                    exp_data[:,0] = donnees_filtrees - mean_donnees_filtrees
                    exp_data[:,1] = temps
                    num_data = np.zeros((len(dt_time),2))
                    num_data[:,0] = [k * 1025 * 0.0275 * 0.0275 for k in dt_data]
                    num_data[:,1] = dt_time
                    num_data_REF = np.zeros((len(dt_time_REF),2))
                    num_data_REF[:,0] = [k * 1025 * 0.0275 * 0.0275 for k in dt_data_REF]
                    num_data_REF[:,1] = dt_time_REF

                    dist = similaritymeasures.frechet_dist(exp_data, num_data[2:])
                    dist_REF = similaritymeasures.frechet_dist(exp_data, num_data_REF[2:])
                    print("Distance de Fréchet entre la courbe des capteurs et la courbe de HAACHAMA: ", dist)
                    print("Distance de Fréchet entre la courbe des capteurs et la courbe du DT parfait: ", dist_REF)

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