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
plt.savefig('signal_voix.png')

# Calcul et affichage de la densité spectrale de la voix enregistrée
frequencies, power_spectral_density_voice = signal.welch(recorded_audio[:, 0], fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_voice)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal enregistré (Voix)')
plt.grid()
plt.savefig('densite_spectrale_voix.png')
plt.show()
plt.close()

# Paramètres de génération du bruit
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
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

signal_noise = generate_noise(noise_type, len(t))
signal_noise = signal_noise / np.max(np.abs(signal_noise))
wavio.write("bruit_genere.wav", signal_noise.astype(np.float32).reshape(-1, 1), fs, sampwidth=2)

# Affichage du signal temporel du bruit généré
plt.figure(figsize=(10, 4))
plt.plot(t, signal_noise, label='Bruit généré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio généré (Bruit)')
plt.legend()
plt.grid()
plt.savefig('signal_bruit.png')

# Densité spectrale du bruit généré
frequencies, power_spectral_density_noise = signal.welch(signal_noise, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_noise)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du bruit généré')
plt.grid()
plt.savefig('densite_spectrale_bruit.png')
plt.show()
plt.close()

adjusted_noise = signal_noise[:len(recorded_audio)] * 0.1
combined_audio = recorded_audio[:, 0] + adjusted_noise
wavio.write("voix_et_bruit_combine.wav", combined_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Affichage du signal combiné
time = np.linspace(0, duration, num=len(combined_audio))
plt.figure(figsize=(10, 4))
plt.plot(time, combined_audio, label='Voix + Bruit')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio combiné (Voix + Bruit)')
plt.legend()
plt.grid()
plt.savefig('signal_combine.png')

# Densité spectrale du signal combiné
frequencies, power_spectral_density = signal.welch(combined_audio, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal combiné (Voix + Bruit)')
plt.grid()
plt.savefig('densite_spectrale.png')
plt.show()
plt.close()

# Application du filtre passe-bas de Tchebychev
epsilon = 0.5  # Ondulation en dB
cutoff_frequency = 3000
nyquist = 0.5 * fs
normal_cutoff = cutoff_frequency / nyquist

b, a = signal.cheby1(4, epsilon, normal_cutoff, btype='low', analog=False)
filtered_audio = signal.filtfilt(b, a, combined_audio)

print(f"Signal filtré - Min: {np.min(filtered_audio)}, Max: {np.max(filtered_audio)}")
wavio.write("voix_et_bruit_filtre.wav", filtered_audio.astype(np.float32).reshape(-1, 1), fs, sampwidth=3)

# Affichage du signal filtré
plt.figure(figsize=(10, 4))
plt.plot(time, filtered_audio, label='Voix + Bruit Filtré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio filtré (Voix + Bruit)')
plt.legend()
plt.grid()
plt.savefig('signal_filtre.png')

# Densité spectrale du signal filtré
frequencies, power_spectral_density_filtered = signal.welch(filtered_audio, fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density_filtered)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal filtré (Voix + Bruit)')
plt.grid()
plt.savefig('densite_spectrale_filtre.png')
plt.show()
plt.close()

print("Caractéristiques du filtre passe-bas utilisé :")
print(f"Type de filtre : Tchebychev")
print(f"Ordre du filtre : 4")
print(f"Fréquence de coupure : {cutoff_frequency} Hz")
print(f"Ondulation en bande passante : {epsilon} dB")
