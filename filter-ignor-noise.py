import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fft
import sounddevice as sd
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

# Ajuster le bruit généré à la voix en termes de volume
adjusted_noise = signal_noise[:len(recorded_audio)] * 0.1  # Facteur d'amplification pour le bruit

# Combinaison du bruit et de la voix
combined_audio = recorded_audio[:, 0] + adjusted_noise

# Sauvegarde du signal combiné
wavio.write("voix_et_bruit_combine.wav", combined_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Appliquer la soustraction spectrale
def spectral_subtraction(signal, noise):
    # FFT du signal et du bruit
    signal_freq = fft.fft(signal)
    noise_freq = fft.fft(noise)
    
    # Estimer la magnitude du bruit
    noise_magnitude = np.abs(noise_freq)
    
    # Soustraction spectrale : On soustrait la magnitude du bruit
    signal_mag = np.abs(signal_freq) - noise_magnitude
    signal_mag = np.maximum(signal_mag, 0)  # On s'assure que la magnitude est positive
    
    # Reconstruction du signal avec la phase du signal original
    restored_signal_freq = signal_mag * np.exp(1j * np.angle(signal_freq))
    restored_signal = np.real(fft.ifft(restored_signal_freq))
    
    return restored_signal

# Appliquer la soustraction spectrale pour restaurer le signal
restored_audio = spectral_subtraction(combined_audio, adjusted_noise)

# Sauvegarde du signal restauré
wavio.write("voix_et_bruit_restaure.wav", restored_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Affichage du signal temporel restauré
plt.figure(figsize=(10, 4))
plt.plot(t, restored_audio, label='Voix + Bruit Restauré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio restauré (Voix + Bruit)')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal restauré sous forme d'image
plt.savefig('signal_restaure.png')

# Calcul et affichage de la densité spectrale du signal restauré
frequencies, power_spectral_density_restored = signal.welch(restored_audio, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_restored)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal restauré (Voix + Bruit)')
plt.grid()

# Enregistrer le graphique de la densité spectrale du signal restauré sous forme d'image
plt.savefig('densite_spectrale_restaure.png')

# Affichage des graphiques
plt.show()
plt.close()  # Ferme la figure après l'affichage
