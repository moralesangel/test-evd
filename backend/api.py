from keras.models import load_model
import os
import matplotlib.pyplot as plt
import librosa
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyaudio
import wave

models = ['cnn_chroma9395']
model_name = models[0]

#model = load_model(f'C:\\Users\\angel\\OneDrive - alum.uca.es\\UNI\\TFG\\Código\\TFG\\saved_nets\\{model_name}.keras')
model = load_model(f'{model_name}.keras')

CHUNK = 1024  # Tamaño de cada fragmento de audio
FORMAT = pyaudio.paInt16  # Formato de audio
CHANNELS = 1  # Número de canales de audio (mono)
RATE = 44100  # Tasa de muestreo (número de muestras por segundo)
RECORD_SECONDS = 5  # Duración de cada segmento de audio
OUTPUT_FOLDER = "audio_segments"  # Carpeta para guardar los archivos de audio
SEGMENT_COUNT = 1
RESULT = None

#$env:FLASK_APP = "api.py"

def calculate_rms(data):
    """ Calcula el valor RMS (Root Mean Square) del audio """
    return np.sqrt(np.mean(np.square(data)))

def calculate_decibels(rms):
    """ Convierte el valor RMS a decibelios (dB) """
    if rms > 0:
        db = 20 * np.log10(rms)
    else:
        db = -np.inf
    return db

def dB():
    # Inicializar PyAudio
    p = pyaudio.PyAudio()

    # Abrir el stream de audio
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Escuchando...")

    try:
        while True:
            # Leer datos del buffer
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            # Calcular RMS
            rms = calculate_rms(data)
            # Calcular dB
            db = calculate_decibels(rms)
            print(f"Decibelios: {db:.2f} dB")
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
    finally:
        # Cerrar el stream
        stream.stop_stream()
        stream.close()
        p.terminate()

def chroma(file, save=False, show=False):
    audio, sample_rate = librosa.load(file)
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_features = resize(chroma_features, (12, 431), mode='constant', anti_aliasing=True)

    if save:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma_features, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.savefig(f'chroma{SEGMENT_COUNT}.png')
    
    if save and show:
        plt.show()

    x = []
    x.append(chroma_features)
    x = np.array(x)

    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

    return x, chroma_features

def record_audio(filename):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    decibel_levels = []

    print("Grabando...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

        # Convertir los datos leídos en un array de numpy
        numpy_data = np.frombuffer(data, dtype=np.int16)

        # Calcular RMS y decibelios
        rms = calculate_rms(numpy_data)
        db = calculate_decibels(rms)
        decibel_levels.append(db)

    print("Terminado de grabar.")
    print(f"Decibelios medios: {np.mean(decibel_levels):.2f} dB")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Calcular y devolver la media de decibelios
    avg_db = np.mean([x**2 for x in decibel_levels])
    return avg_db

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/evd', methods=['GET'])
def start_detection():
    global RESULT
    global SEGMENT_COUNT

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    file = os.path.join(OUTPUT_FOLDER, f"segment_{SEGMENT_COUNT}.wav")
    filename = f'segment_{SEGMENT_COUNT}'
    filepath = f'{OUTPUT_FOLDER}/{filename}.wav'

    dB = record_audio(file)

    if dB < 500:
        print("Audio demasiado silencioso.")
        RESULT = ['❌', 0, int(dB)]
        return jsonify(RESULT)
    
    x, chroma_features = chroma(filepath)
    p1 = model.predict(x, verbose=0)[0][1]

    prediction = p1

    print(f'Probabilidad de vehiculo de emergencia: {prediction*100:.4f}%')
    if prediction > 0.95:
        output = '🚑🚒🚓'
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma_features, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.savefig(f'chroma{SEGMENT_COUNT}.png')
        """
    else:
        output = '❌'
    print(f'Prediccion: {output}')

    os.remove(filepath)
    SEGMENT_COUNT += 1

    RESULT = [output, int(prediction*100), int(dB)]
    return jsonify(RESULT)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Desactivar reloader para evitar reinicios múltiples
