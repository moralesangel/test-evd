import pandas as pd
from pytube import YouTube
from pydub import AudioSegment
import concurrent.futures
import threading

CSV_FILE = 'dataset/tst.csv'
OUTPUT_PATH = 'dataset/tst'
ROWS = None
NUM_POSITIVE_AUDIOS = 118

class Car:
    def __init__(self, row):
        self.download = False
        self.row = row

class Bridge:
    def __init__(self):
        self.correct_exit = 0
        self.cars_bridge = 0
        self.cars_waiting = 0
        self.max_cars_bridge = NUM_POSITIVE_AUDIOS
        self.cars = []
        self.write = threading.Lock()
        self.lock = threading.Lock()
        self.condition = threading.Condition()

    def car_arrive(self, car):
        with self.write:
            self.cars.append(car)
            self.cars_waiting += 1
        self.try_enter(car)

    def try_enter(self, car: Car):

        self.lock.acquire()
        try:
            while self.correct_exit + self.cars_bridge >= self.max_cars_bridge:
                self.condition.wait()

            self.cars_bridge += 1
            self.cars_waiting -= 1
            
            download_and_process(car)

            self.cars_bridge -= 1
            
            if car.download:
                self.correct_exit += 1
                self.condition.notify_all()

        finally:
            self.lock.release()

def download_and_process(car: Car):

    row = car.row
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
            car.download = True

    except Exception as e:
        print(f"{output_filename} could not be downloaded or trimmed. Error: {str(e)}")

# Read the CSV file
CSV = pd.read_csv(CSV_FILE, sep=',', engine='python', comment='#', nrows=ROWS)

bridge = Bridge()

def process_row(row, bridge):
    car = Car(row)
    bridge.car_arrive(car)

# Create ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Execute downloads and trimming in parallel
    futures = [executor.submit(process_row, row, bridge) for index, row in CSV.iterrows()]    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("All rows done.")