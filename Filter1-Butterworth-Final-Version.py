import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import wavio

# Paramètres d'enregistrement
duration = 3  # Durée de l'enregistrement en secondes
fs = 44100  # Fréquence d'échantillonnage (Hz)

# Enregistrement de la voix
try:
    print("Parlez maintenant...")
    recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Enregistrement terminé.")
except Exception as e:
    print(f"Erreur lors de l'enregistrement : {e}")
    exit()

# Sauvegarde de l'audio dans un fichier WAV
wavio.write("enregistrement.wav", recorded_audio, fs, sampwidth=3)

# Affichage du signal temporel de la voix enregistrée
time = np.linspace(0, duration, num=len(recorded_audio))
plt.figure(figsize=(10, 4))
plt.plot(time, recorded_audio[:, 0], label='Voix enregistrée')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio enregistré (Voix)')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal de la voix sous forme d'image
plt.savefig('signal_voix.png')

# Calcul et affichage de la densité spectrale de la voix enregistrée
frequencies, power_spectral_density_voice = signal.welch(recorded_audio[:, 0], fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_voice)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal enregistré (Voix)')
plt.grid()

# Enregistrer le graphique de la densité spectrale de la voix sous forme d'image
plt.savefig('densite_spectrale_voix.png')

# Affichage des graphiques
plt.show()
plt.close()  # Ferme la figure après l'affichage

# Paramètres de génération du bruit
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

# Affichage du signal temporel du bruit généré
plt.figure(figsize=(10, 4))
plt.plot(t, signal_noise, label='Bruit généré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio généré (Bruit)')
plt.legend()
plt.grid()

# Enregistrer le graphique du bruit généré sous forme d'image
plt.savefig('signal_bruit.png')

# Calcul et affichage de la densité spectrale du bruit généré
frequencies, power_spectral_density_noise = signal.welch(signal_noise, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_noise)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du bruit généré')
plt.grid()

# Enregistrer le graphique de la densité spectrale du bruit sous forme d'image
plt.savefig('densite_spectrale_bruit.png')

# Affichage des graphiques
plt.show()
plt.close()  # Ferme la figure après l'affichage

# Ajuster le bruit généré à la voix en termes de volume
adjusted_noise = signal_noise[:len(recorded_audio)] * 0.1  # Facteur d'amplification pour le bruit

# Combinaison du bruit et de la voix
combined_audio = recorded_audio[:, 0] + adjusted_noise

# Sauvegarde du signal combiné
wavio.write("voix_et_bruit_combine.wav", combined_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Création de l'axe temporel pour le signal combiné
time = np.linspace(0, duration, num=len(combined_audio))

# Affichage du signal temporel combiné
plt.figure(figsize=(10, 4))
plt.plot(time, combined_audio, label='Voix + Bruit')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio combiné (Voix + Bruit)')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal sous forme d'image
plt.savefig('signal_combine.png')

# Calcul et affichage de la densité spectrale du signal combiné
frequencies, power_spectral_density = signal.welch(combined_audio, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal combiné (Voix + Bruit)')
plt.grid()

# Enregistrer le graphique de la densité spectrale sous forme d'image
plt.savefig('densite_spectrale.png')

# Affichage des graphiques
plt.show()
plt.close()  # Ferme la figure après l'affichage

# Application du filtre passe-bas pour filtrer les hautes fréquences
cutoff_frequency = 3000  # Fréquence de coupure en Hz (par exemple, 3000 Hz)
nyquist = 0.5 * fs
normal_cutoff = cutoff_frequency / nyquist

# Conception du filtre passe-bas
b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)  # Filtre de Butterworth d'ordre 4

# Filtrage du signal combiné
filtered_audio = signal.filtfilt(b, a, combined_audio)

# Vérification du signal filtré
print(f"Signal filtré - Min: {np.min(filtered_audio)}, Max: {np.max(filtered_audio)}")

# Sauvegarde du signal filtré
wavio.write("voix_et_bruit_filtre.wav", filtered_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Affichage du signal temporel filtré
plt.figure(figsize=(10, 4))
plt.plot(time, filtered_audio, label='Voix + Bruit Filtré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio filtré (Voix + Bruit)')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal filtré sous forme d'image
plt.savefig('signal_filtre.png')

# Calcul et affichage de la densité spectrale du signal filtré
frequencies, power_spectral_density_filtered = signal.welch(filtered_audio, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_filtered)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal filtré (Voix + Bruit)')
plt.grid()

# Enregistrer le graphique de la densité spectrale du signal filtré sous forme d'image
plt.savefig('densite_spectrale_filtre.png')

# Affichage des graphiques
plt.show()
plt.close()  # Ferme la figure après l'affichage

# Affichage des caractéristiques du filtre
print("Caractéristiques du filtre passe-bas utilisé :")
print(f"Type de filtre : Butterworth")
print(f"Ordre du filtre : 4")
print(f"Fréquence de coupure : {cutoff_frequency} Hz")
