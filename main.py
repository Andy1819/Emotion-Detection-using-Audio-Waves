from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import librosa
from keras.models import model_from_json
import joblib
from keras.models import load_model

app=FastAPI(title = "Emotion Detection usong Audio waves")

loaded_model = load_model('./emotion_model.h5')
# Load scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

# Define classes
EMOTIONS = ['female_happy', 'female_angry', 'female_fear', 'female_disgust','female_sad','female_neutral','male_neutral','male_fear','male_angry','male_happy','male_disgust','male_sad','female_surprise','male_surprise']


def extract_features(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    features = []

    # Zero Crossing Rate (ZCR)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    features.extend(zcr)

    # Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    features.extend(chroma)

    # Mel-Frequency Cepstral Coefficients (MFCCs)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr).T, axis=0)
    features.extend(mfcc)

    # Root Mean Square Energy (RMSE)
    rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    features.extend(rmse)

    # Mel spectrogram
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    features.extend(mel_spectrogram)

    # Assuming you need to scale your features
    features_scaled = scaler.transform([features])
    
    return features_scaled

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return open("index.html", "r").read()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type == "audio/wav":
        file_path = "./" + file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        features = extract_features(file_path)
        prediction = loaded_model.predict(features)
        result = EMOTIONS[np.argmax(prediction)]
        return {"result": result}
    else:
        return {"message": "Invalid file type. Please upload a .wav file."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1",reload=True)