# IMPORTING LIBRARIES
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set.")
warnings.filterwarnings("ignore", message="n_fft=2048 is too large for input signal of length=0")
import pickle
from crossval import crossval
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
 
# IMPORTING FEATURE EXTRACTION METHODS
from feature_extraction.chroma import chroma

# IMPORTING MODELS
from models.cnn import cnn

# SETTING UP THE CONFIGURATIONS
# ----------------------------------------------
FEATURE_EXTRACTION = chroma
MODEL = cnn
DATASET_PATH = 'dataset/audios/'

EPOCHS = 10
BATCH_SIZE = 256

CHOOSEN_NET = 'cnn_chroma9395' # If None, a new net will be trained. If not, the net will be loaded from saved_nets
# ----------------------------------------------


# FEATURE EXTRACTION (only first time with each feature extraction method)
# --------------------------------------------------
features = os.listdir('saved_features/')
features_loaded = False
if len(features) > 0:
    for f in features:
        if f == f'{FEATURE_EXTRACTION.__name__}.pkl':
            with open(f'saved_features/{f}', 'rb') as file:
                x, y, input_shape = pickle.load(file)
            print(f'Loaded {FEATURE_EXTRACTION.__name__} features')
            features_loaded = True
if not features_loaded:
    x, y, input_shape = FEATURE_EXTRACTION(DATASET_PATH)
# --------------------------------------------------



# SPLITTING THE DATASET INTO TRAINING AND TESTING SETS
#xtrn,xtst,ytrn,ytst=train_test_split(x,y,test_size=0.2,random_state=RANDOM_STATE, shuffle=True)

# CROSS VALIDATION
K = 5
n = len(y)
size = n/K

A = []
i = 1
while i <= K:
    xtrn, xtst, ytrn, ytst = crossval(x, y, K, i)

    # INITIALIZING THE MODEL
    model = None
    if CHOOSEN_NET is not None: # Load the model
        models = os.listdir('saved_nets/')
        for m in models:
            if m == f'{CHOOSEN_NET}.keras':
                model = load_model(f'saved_nets/{m}')
                print(f'Loaded {CHOOSEN_NET} model')
    else: # Train the model
        model = MODEL(input_shape)
        print(f'Created {MODEL.__name__} model')
        print(f'Training {MODEL.__name__} net...')
        model.fit(xtrn, ytrn, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(xtst, ytst))

        # GRID SEARCH
        """
        model = KerasClassifier(build_fn=model, epochs=10, batch_size=256, verbose=0)
        param_grid = {
            'num_filters': [16, 32, 64],
            'kernel_size': [3, 5],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'sgd']
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(x, y)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        model = grid_result.best_estimator_.model
        model.save('saved_nets/best_model.h5')
        print('Best model saved to "saved_nets/best_model.h5"')
        """


    # TESTING NET
    print(f'Testing net...')
    results = model.evaluate(xtst, ytst)
    if type(results) != float:
        loss = results[0]
        accuracy = results[1]
        print(f'Loss: {loss}')
    else:
        accuracy = results

    print(f'Accuracy: {accuracy*100}%')

    A.append(accuracy)
    i += 1

# AVERAGE ACCURACY OF THE NET
print(A)
average = sum(A)/len(A)*100
print(f'Average accuracy: {average}%')

# SAVING THE NET
if CHOOSEN_NET is None:
    model.save(f'saved_nets/{MODEL.__name__}_{FEATURE_EXTRACTION.__name__}{int(average*100)}.keras')
    print(f'Net saved: saved_nets/{MODEL.__name__}_{FEATURE_EXTRACTION.__name__}{int(average*100)}.keras')

