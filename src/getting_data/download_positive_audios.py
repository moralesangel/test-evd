import pandas as pd
from pytube import YouTube
from pydub import AudioSegment
import concurrent.futures
import threading

lock = threading.Lock()

CSV_FILE = 'dataset/large.csv'
OUTPUT_PATH = 'dataset/large'
ROWS = None

num = 0

def download_and_process(row):
    global num
    print(f'{num/4697}')
    try:
        # Check if the output is 0
        if row['output'] == 0:
            
            audio_id = row['YTID']
            start = row['start_seconds']
            end = row['end_seconds']

            # Construct the audio URL
            audio_url = f'https://www.youtube.com/watch?v={audio_id}'
            
            output_filename = f'0_{audio_id}.wav'

            # Download the audio
            yt = YouTube(audio_url)
            yt.streams.filter(only_audio=True).first().download(output_path=OUTPUT_PATH, filename=output_filename)
            print(f"{output_filename} audio downloaded successfully.")

            # Trim the audio
            audio = AudioSegment.from_file(f'{OUTPUT_PATH}/{output_filename}')
            clip = audio[start*1000:end*1000]
            clip.export(f'{OUTPUT_PATH}/{output_filename}', format='wav')
            print(f"{output_filename} audio trimmed successfully.")

            lock.acquire()
            num += 1
            lock.release()
    except Exception as e:
        print(f"{output_filename} could not be downloaded or trimmed. Error: {str(e)}")

# Read the CSV file
CSV = pd.read_csv(CSV_FILE, sep=',', engine='python', comment='#', nrows=ROWS)

# Create ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Execute downloads and trimming in parallel
    futures = [executor.submit(download_and_process, row) for index, row in CSV.iterrows()]
    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
