import os
import mne
from mne.parallel import parallel_func
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import study_path, event_id_mapping, meg_dir
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_sub_epochs(subject_id: int, freq_band: str = None) -> mne.Epochs:
    """
    Read MNE epochs object for a specific subject and frequency band.

    Args:
        subject_id (int): The ID of the subject.
        freq_band (str): The frequency band name (optional).

    Returns:
        mne.Epochs: The MNE epochs object.
    """
    subject = f"sub-{subject_id:02d}"
    if freq_band is None:
        epochs = mne.read_epochs(
            os.path.join(meg_dir, subject, f"{subject}-epo.fif"), verbose=False
        )
    else:
        epochs = mne.read_epochs(
            os.path.join(meg_dir, subject, f"{subject}-{freq_band}-epo.fif"), verbose=False
        )
    return epochs

def run_spatiotemp_decoder_single_sub(subject_id:int):
    """"
    Run a spatiotemporal decoding analysis on MEG data for a specific subject.
    Four different modalities are being tested:
        - Face vs Scrambled
        - Famous vs Unfamiliar
        - Famous vs Scrambled
        - Unfamiliar vs Scrambled
    Many different filterings are being tested:
        - Raw
        - Delta (1-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-15 Hz)
        - Beta (15-30 Hz)
        - Low gamma (30-60 Hz)
        - High gamma (60-90 Hz)
    """""
    subject = f"sub-{subject_id:02d}"
    print(f"Processing subject: {subject}")

    scores_all = []    
    for modality in ["face_vs_scrambled", "famous_vs_unfamiliar", "famous_vs_scrambled", "unfamiliar_vs_scrambled"]:
        for freq_band in ["raw", "delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]:    
            if freq_band == "raw":
                epochs = read_sub_epochs(subject_id)
            else:
                epochs = read_sub_epochs(subject_id, freq_band)
            logger.info(f"  Modality: {modality}, Frequency band: {freq_band}")
            

            if modality == "face_vs_scrambled":
                events = mne.merge_events(epochs.events, [1,2], 1, replace_events=True) # Merge Famous and Unfamiliar
            elif modality == "famous_vs_unfamiliar":
                event_id_to_drop = 3
                epochs = epochs[epochs.events[:,2] != event_id_to_drop]
                events = epochs.events
            elif modality == "famous_vs_scrambled":
                event_id_to_drop = 2
                epochs = epochs[epochs.events[:,2] != event_id_to_drop]
                events = epochs.events
            else:
                event_id_to_drop = 1 #famous
                epochs = epochs[epochs.events[:,2] != event_id_to_drop]
                events = epochs.events
            
            
            epochs.pick(picks = "meg", exclude = "bads")

            X = epochs.get_data(copy = False)   # MEG signals: n_epochs, n_meg_channels, n_times
            y = events[:, 2]  # target

            def clv_spatio_temporal(X, y):
                clf = make_pipeline(
                Scaler(epochs.info),
                Vectorizer(),
                SVC(kernel='rbf', C=1, max_iter=1000)
                #LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto') 
                )

                scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None)

                # Mean scores across cross-validation splits
                score = np.mean(scores, axis=0)
                return score
            logger.info(f"  Running spatio-temporal decoding analysis for subject {subject}, modality {modality} and frequency band {freq_band}...") 
            score = clv_spatio_temporal(X, y)
            scores_all.append([subject, modality, freq_band, score])
    scores_all_df = pd.DataFrame(scores_all, columns=["Subject", "Modality", "Frequency Band", "Score"])
    logger.info('Saving all scores to csv file...')
    scores_all_df.to_csv(f"scores_decoding_svm_rbf_spatiotemporal_sub-{subject_id:02d}.csv", index=False)

# Leave one subject out decoding analysis
def run_spatiotemp_decoder_LOSOCV(subject_ids:list[int]): 
    scores_all_fvs = []
    scores_all_fvu = []

    # Iterate over all subjects considering them as testing split
    for test_subject_id in subject_ids:
        logger.info(f"Starting cross-validation with testing subject {test_subject_id:02d}")
          
        test_sub = f"sub-{test_subject_id:02d}"
        
        train_subjects = subject_ids.copy()
        train_subjects.remove(test_subject_id) # No need for the for loop
        
        # Iterate over both modalities 
        for modality in ["face_vs_scrambled", "famous_vs_unfamiliar"]:
            
            # Define the model each time we change modality
            model = SVC(kernel='linear', C=1, max_iter=1000)
            
            logger.info(f"Modality: {modality}")
            X, y = [], []
            # Iterate over all subjects except the testing subject to train the model
            for train_sub_id in train_subjects:
                epochs = mne.read_epochs(os.path.join(meg_dir, f'sub-{train_sub_id:02d}', f'sub-{train_sub_id:02d}-epo.fif'), verbose=False)
                
                if modality == "face_vs_scrambled":
                    events = mne.merge_events(epochs.events, [1,2], 1, replace_events=True)
                else:
                    event_id_to_drop = 3
                    epochs = epochs[epochs.events[:,2] != event_id_to_drop]
                    events = epochs.events
                epochs = epochs.pick(picks = "meg", exclude = "bads")
                X_subj = epochs.get_data(copy = False)
                y_subj = events[:, 2]
                X.append(X_subj)
                y.append(y_subj)
                print(len(X), len(y))
            # Flatten the list of arrays
            X = [epoch for subject in X for epoch in subject]
            y = [epoch for subject in y for epoch in subject]
            X = np.array(X)
            y = np.array(y)
            clf = make_pipeline(
                Scaler(epochs.info), # I don't know why the input is epochs.info but this might need to be changed
                Vectorizer(),
                model
            )
            clf.fit(X, y) # You fit the model on 15 subject simultaniously and test on 1 subject 
            # print(f"Fitted model for modality {modality} and subject {train_sub_id}")
            logger.info(f"Finished training for modality {modality}, now testing on subject {test_sub}")
            epochs = mne.read_epochs(os.path.join(meg_dir, test_sub, f'{test_sub}-epo.fif'), verbose=False)
            if modality == 'face_vs_scrambled':
                events = mne.merge_events(epochs.events, [1,2], 1, replace_events=True)
            else:
                event_id_to_drop = 3
                epochs = epochs[epochs.events[:,2] != event_id_to_drop]
                events = epochs.events
            epochs = epochs.pick(picks = "meg", exclude = "bads")
            X_test = epochs.get_data(copy = False)
            y_test = events[:, 2]
            score = clf.score(X_test, y_test)
            if modality == "face_vs_scrambled":
                scores_all_fvs.append([score])
                
            else:
                scores_all_fvu.append([score])
        # Mean of scores across cross-validation splits
    score_fvs = np.mean(scores_all_fvs, axis=0)
    score_fvu = np.mean(scores_all_fvu, axis=0)
    scores_all = [
    {"Modality": "Scores Face vs Scrambled", "Score": score_fvs},
    {"Modality": "Scores Famous vs Unfamiliar", "Score": score_fvu}
    ]
    scores_all_df = pd.DataFrame(scores_all)
    logger.info('Saving all scores to csv file...')
    scores_all_df.to_csv(f"scores_decoding_svm_rbf_spatiotemporal_LOSOCV.csv", index=False)




#parallel, run_func, _ = parallel_func(run_spatiotemp_decoder_single_sub, n_jobs=-1)
#parallel(run_func(subject_id) for subject_id in range(1, 2))

run_spatiotemp_decoder_LOSOCV(list(range(1, 17)))
