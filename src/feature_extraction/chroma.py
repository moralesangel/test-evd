import librosa
import librosa.display
import numpy as np
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from rich.progress import track
import pickle

def chroma(dataset_path):
    x = []
    y = []
    files = os.listdir(dataset_path)

    input_shape = (12, 431)
    for file in track(files, description="Extracting Chroma Features:"):
        audio, sample_rate = librosa.load(dataset_path+file)
        chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        # Resize features to a fixed size
        chroma_features = resize(chroma_features, input_shape, mode='constant', anti_aliasing=True)
        x.append(chroma_features)
        y.append(int(file[0]))
        
    x = np.array(x)
    y = np.array(y)
    y = to_categorical(y, 2) # One hot encoding with 2 classes

    # TODO Normalize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

    # Save the features
    with open('saved_features/chroma.pkl', 'wb') as f:
        pickle.dump((x, y, input_shape), f)

    return x, y, input_shape

# Mostrar el diagrama de características cromáticas
def show_chroma(data):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    files = os.listdir('dataset/audios/')
    a = '/m/02mk9,/m/07pb8fc,/m/07yv9,/t/dd00066'
    traffic_label = '/m/0btp2'
    file1 = '1_ZWmXgdnE3ig'
    file2 = '0_-1d1HDhZpVM'
    file3 = '0_--0w1YA1Hm4'
    file4 = '0_--aVt0_KbIs'
    file5 = '0_--0bntG9i7E'
    file = file1
    audio, sample_rate = librosa.load('dataset/audios/'+file+'.wav')
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_features = resize(chroma_features, (12, 431), mode='constant', anti_aliasing=True)
    print(file)
    show_chroma(chroma_features)

