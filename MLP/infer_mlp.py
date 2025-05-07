import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ==== FILE PATHS ====
model_dir = 'Models'
mlp_dir = f'{model_dir}/MLP'
encoder_path = f'{model_dir}/label_encoders.pkl'
scaler_path = f'{model_dir}/scaler.pkl'

# ==== LOAD ENCODERS AND SCALER ====
with open(encoder_path, 'rb') as f:
    label_encoders = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# ==== INPUT COLUMNS ====
input_cols = ['inningNumber', 'oversActual', 'totalRuns', 'totalWickets',
              'time_of_day', 'Ground Name', 'Batsman_Name', 'Batsman_Role',
              'Batsman_Batting_Style', 'Batsman_Playing_Role']

# ==== DEFINE INPUTS ====
inputs = [
    [1, 5.2, 45, 1, 'Day', 'Eden Gardens', 'Virat Kohli', 'Top Order', 'Right-hand bat', 'Batsman'],
    [2, 15.4, 120, 4, 'Night', 'Wankhede Stadium', 'Jos Buttler', 'Middle Order', 'Right-hand bat', 'Wicketkeeper'],
    [1, 8.0, 60, 2, 'Day', 'MCG', 'David Warner', 'Opener', 'Left-hand bat', 'Batsman']
]

df_inputs = pd.DataFrame(inputs, columns=input_cols)

# ==== ENCODE CATEGORICAL ====
for col in input_cols:
    if col in label_encoders:
        df_inputs[col] = label_encoders[col].transform(df_inputs[col])

# ==== SCALE ====
scaled_inputs = scaler.transform(df_inputs)

# ==== LOAD MODELS ====
models = {
    'bowler_style': load_model(f'{mlp_dir}/Bowler_Bowling_Style_model_safe.h5'),
    'pitch_length': load_model(f'{mlp_dir}/pitchLength_model_safe.h5'),
    'pitch_line': load_model(f'{mlp_dir}/pitchLine_model_safe.h5'),
    'shot_type': load_model(f'{mlp_dir}/shotType_model.h5')
}

# ==== PREDICT ====
for task, model in models.items():
    preds = model.predict(scaled_inputs)
    preds = np.argmax(preds, axis=1)  # assuming output is softmax
    
    target_col = {
        'bowler_style': 'Bowler_Bowling_Style',
        'pitch_length': 'pitchLength',
        'pitch_line': 'pitchLine',
        'shot_type': 'shotType'
    }[task]
    
    if target_col in label_encoders:
        preds = label_encoders[target_col].inverse_transform(preds.astype(int))
    
    for i, pred in enumerate(preds):
        print(f"Input {i+1} - {task}: {pred}")
