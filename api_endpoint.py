import io 
import uuid
from datetime import datetime, timezone
from typing import List
import librosa
import torch
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
from models.m2d.M2D_wrapper import M2DWrapper


from data_util import audioset_classes
from helpers.decode_old import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.prediction_wrapper import PredictionsWrapper

CLASSIFIER_ID="sed_events_v2"
MODEL_VERSION="2026-03-10"


TARGET_CLASSES = [
    'Babbling',
    'Baby cry, infant cry',
    'Breaking',
    'Child speech, kid speaking',
    'Children shouting',
    'Crying, sobbing',
    'Explosion',
    'Female speech, woman speaking',
    'Glass shatter',
    'Gunshot, gunfire',
    'Male speech, man speaking',
    'Screaming',
    'Shout',
    'Smash, crash',
    'Wail, moan',
    'Yell',
]

EVENT_LABEL_MAP = {
    'Babbling': "babbling",
    'Baby cry, infant cry': "baby_cry",
    'Breaking': "breaking",
    'Child speech, kid speaking': "child_speech",
    'Children shouting': "children_shouting",
    'Crying, sobbing': "crying_sobbing",
    'Explosion': "explosion",
    'Female speech, woman speaking': "female_speech",
    'Glass shatter': "glass_shatter",
    'Gunshot, gunfire': "gunshot",
    'Male speech, man speaking': "male_speech",
    'Screaming': "screaming",
    'Shout': "shout",
    'Smash, crash': "smash_crash",
    'Wail, moan': "wail_moan",
    'Yell': "yell",
}

BROAD_CATEGORY_MAP={
    'Babbling': "danger",
    'Baby cry, infant cry': "danger",
    'Breaking': "danger",
    'Child speech, kid speaking': "speech",
    'Children shouting': "danger",
    'Crying, sobbing': "danger",
    'Explosion': "danger",
    'Female speech, woman speaking': "speech",
    'Glass shatter': "danger",
    'Gunshot, gunfire': "danger",
    'Male speech, man speaking': "speech",
    'Screaming': "danger",
    'Shout': "danger",
    'Smash, crash': "danger",
    'Wail, moan': "danger",
    'Yell': "danger",
}


class LabelOut(BaseModel):
    event_label: str 
    broad_category: str 
    overlap: bool 
    start_time: float
    end_time: float
    scores: float

class ResultOut(BaseModel):
    audio_file_id: str
    classifier_id: str
    model_version:str 
    run_id: str
    event_type: str 
    labels: List[LabelOut]
    created_at: str 


def _read_uploadfile(file):
    data=file.file.read()
    if not data:
        raise ValueError(f"Empty file {file.filename}")
    bio = io.BytesIO(data)
    return bio



def _mark_overlaps(label_items):
    for i,li in enumerate(label_items):
        li.overlap= False
        for j,lj in enumerate(label_items):
            if i==j:
                continue
            if (li.start_time<lj.end_time) and (lj.start_time<li.end_time):
                li.overlap=True
                break
    return label_items


def sound_event_detection(files,threshold=0.5):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    m2d = M2DWrapper()
    model = PredictionsWrapper(m2d, checkpoint="M2D_strong_1", embed_dim=m2d.m2d.cfg.feature_d)


    model.eval()
    model.to(device)

    sample_rate = 16_000  # all our models are trained on 16 kHz audio
    segment_duration = 10  # all models are trained on 10-second pieces
    segment_samples = segment_duration * sample_rate
    final_preds=[]

    for f in files:
        file_data=_read_uploadfile(f)
        (waveform, _) = librosa.load(file_data, sr=sample_rate, mono=True)
        waveform = torch.from_numpy(waveform[None, :]).to(device)
        waveform_len = waveform.shape[1]

        audio_len = waveform_len / sample_rate  # in seconds
        

        # encoder manages decoding of model predictions into dataframes
        # containing event labels, onsets and offsets
        encoder = ManyHotEncoder(audioset_classes.as_danger_classes, audio_len=audio_len)

        num_chunks = waveform_len // segment_samples + (waveform_len % segment_samples != 0)
        all_predictions = []


        # Process each 10-second chunk
        for i in range(num_chunks):
            start_idx = i * segment_samples
            end_idx = min((i + 1) * segment_samples, waveform_len)
            waveform_chunk = waveform[:, start_idx:end_idx]

            # Pad the last chunk if it's shorter than 10 seconds
            if waveform_chunk.shape[1] < segment_samples:
                pad_size = segment_samples - waveform_chunk.shape[1]
                waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

            # Run inference for each chunk
            with torch.no_grad():
                mel = model.mel_forward(waveform_chunk)
                y_strong, _ = model(mel)

            # Collect predictions
            all_predictions.append(y_strong)

        # Concatenate all predictions along the time axis
        y_strong = torch.cat(all_predictions, dim=2)
        # convert into probabilities
        y_strong = torch.sigmoid(y_strong)
        y_strong=y_strong.float()
        final_preds.append(y_strong)
    
    # final_preds=torch.cat(final_preds,dim=0)
   
    decoded_predictions = batched_decode_preds( 
        final_preds,
        files,
        encoder,
        median_filter=9,
        thresholds=[threshold],
    )

    return decoded_predictions
    


app= FastAPI(title="Sound Event Detection API", version="1.0.0")



@app.post("/detect", response_model=List[ResultOut])
@torch.inference_mode()
def detect(
    files:List[UploadFile] =File(...,description="wav audio files"), 
    threshold:float =Query(0.5,ge=0.0,le=1.0,description="per-class framewise threshold"), 
    classifier_id=Query(CLASSIFIER_ID),
    model_version=Query(MODEL_VERSION)
):

    

    output=sound_event_detection(files,threshold)


    run_id=f"{classifier_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:3]}"
    created_at= datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

    results=[]
    output=output[threshold]
    output=output.groupby("filename")
    print(len(output))
    for filename, group in output:
        label_items=[]
        for _, row in group.iterrows():
            class_name=row["event_label"]
            start_t=row["onset"]
            end_t=row["offset"]
            avg=row["score"]
            label_items.append(
                    LabelOut(
                        event_label=EVENT_LABEL_MAP.get(class_name),
                        broad_category=BROAD_CATEGORY_MAP.get(class_name),
                        overlap=False,
                        start_time=float(round(start_t,3)),
                        end_time=float(round(end_t,3)),
                        scores=float(round(avg,4)),
                    )
                )
        label_items=_mark_overlaps(label_items)
        results.append(
            ResultOut(
                audio_file_id=filename,
                classifier_id=classifier_id,
                model_version=model_version,
                run_id=run_id,
                event_type="sound event",
                labels=label_items,
                created_at=created_at,
            )
        )
    return results