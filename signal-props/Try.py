import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import wavio

# Paramètres d'enregistrement
duration = 3  # Durée de l'enregistrement en secondes
fs = 44100  # Fréquence d'échantillonnage (Hz)

print("Parlez maintenant...")
recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Enregistrement terminé.")

# Sauvegarde de l'audio dans un fichier WAV
wavio.write("enregistrement.wav", recorded_audio, fs, sampwidth=3)

# Création de l'axe temporel
time = np.linspace(0, duration, num=len(recorded_audio))

# Affichage du signal temporel
plt.figure(figsize=(10, 4))
plt.plot(time, recorded_audio, label='Signal enregistré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal audio en fonction du temps')
plt.legend()
plt.grid()

# Enregistrer le graphique du signal sous forme d'image
plt.savefig('signal_enregistre.png')

# Calcul et affichage de la densité spectrale
frequencies, power_spectral_density = signal.welch(recorded_audio[:, 0], fs, nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_spectral_density)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Densité spectrale de puissance')
plt.title('Densité spectrale du signal')
plt.grid()

# Enregistrer le graphique de la densité spectrale sous forme d'image
plt.savefig('densite_spectrale_signal.png')

# Afficher les graphiques
plt.show()
