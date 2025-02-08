import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import wavio

# Paramètres de génération du bruit
duration = 3  # Durée en secondes
fs = 44100  # Fréquence d'échantillonnage (Hz)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Choisir le type de bruit ('white', 'pink', 'brown')
noise_type = 'white'

def generate_noise(noise_type, size):
    if noise_type == 'white':
        return np.random.normal(0, 1, size)
    elif noise_type == 'pink':
        return np.cumsum(np.random.normal(0, 1, size))
    elif noise_type == 'brown':
        return np.cumsum(np.random.normal(0, 0.01, size))
    else:
        raise ValueError("Type de bruit non supporté")

# Générer le bruit choisi
signal_noise = generate_noise(noise_type, len(t))

# Normalisation
signal_noise = signal_noise / np.max(np.abs(signal_noise))

# Sauvegarde du bruit dans un fichier WAV
wavio.write("bruit_genere.wav", signal_noise.astype(np.float32).reshape(-1, 1), fs, sampwidth=2)

# Affichage du signal temporel
plt.figure(figsize=(10, 4))
plt.plot(t, signal_noise, label=f'Bruit {noise_type}')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title(f'Signal du bruit {noise_type} en fonction du temps')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal sous forme d'image
plt.savefig('signal_bruit_genere.png')

# Calcul et affichage de la densité spectrale
frequencies, power_spectral_density = signal.welch(signal_noise, fs, nperseg=1024)

# Affichage de la densité spectrale
plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title(f'Densité spectrale du bruit {noise_type}')
plt.grid()

# Enregistrer le graphique de la densité spectrale sous forme d'image
plt.savefig('densite_spectrale_bruit.png')

# Afficher les graphiques
plt.show()
