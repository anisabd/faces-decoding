import os.path as op
from pandas import read_csv
import mne
from mne.parallel import parallel_func
import numpy as np
from autoreject import get_rejection_threshold

from config import study_path, meg_dir, N_JOBS, map_subjects, reject_tmax, event_id_mapping


def run_events(subject_id):
    subject = "sub-%02d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(meg_dir, subject)
    out_path = op.join(meg_dir, subject)
    for run in (1,3,4,5,6):

        run_fname = op.join(in_path, 'run_%02d_filt_raw.fif' % (run,))
        raw = mne.io.read_raw_fif(run_fname)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)
        # Read CSV for events
        csv_path = op.join(study_path, 'ds000117', subject, 'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_events.tsv')
        event_csv = read_csv(csv_path, sep='\t')
        events = [event for event in events if event[-1] in event_csv['trigger'].values]
        print(type(events))
        print(len(events))
        for event in events:
            event[-1] = event_id_mapping[event[-1]]
        print(type(events))
 
        print("  S %s - R %s" % (subject, run))
        events = np.array(events)
        fname_events = op.join(out_path, 'run_%02d-eve.fif' % run)
        mne.write_events(fname_events, events, overwrite=True)

def run_epochs(subject_id: int, fmin: float = 0.5, fmax: float = 4, frequency_band: str = None) -> None:
    """
    Process and epoch raw MEG data for a specific subject.

    Args:
        subject_id (int): The ID of the subject.
        fmin (float): The lower frequency bound for filtering.
        fmax (float): The upper frequency bound for filtering.
        frequency_band (str): The frequency band name (optional).

    Returns:
        None
    """
    subject = f"sub-{subject_id:02d}"
    print(f"Processing subject: {subject}")

    data_path = op.join(meg_dir, subject)

    mapping = map_subjects[subject_id]

    raw_list = []
    events_list = []

    print("Loading raw data")
    for run in (1,3,4,5,6):
        bads = []

        bad_name = op.join('bads', mapping, f'run_{run:02d}_raw_tr.fif_bad')
        if op.exists(bad_name):
            with open(bad_name) as f:
                bads = [line.strip() for line in f]

        run_fname = op.join(data_path, f'run_{run:02d}_filt_raw.fif')
        raw = mne.io.read_raw_fif(run_fname, preload=True, verbose=False)

        # A fixed 34 ms delay between trigger and stimulus
        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(op.join(data_path, f'run_{run:02d}-eve.fif'))
        
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        
        raw.info['bads'] = bads
        raw.interpolate_bads()

        if frequency_band is not None:
            raw.filter(fmin, fmax, n_jobs=-1, l_trans_bandwidth=1, h_trans_bandwidth=1)

        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    raw.set_eeg_reference(projection=True)
    del raw_list



    if frequency_band is not None:
        print('Applying hilbert transform')
        raw.apply_hilbert(envelope=True)

    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True, exclude=())

    print('Epoching')
    events_id = [event[2] for event in events]
    epochs = mne.Epochs(raw, events, events_id, -0.2, 0.8, proj=True,
                        picks=picks, baseline=(-0.2, 0.0), preload=True,
                        reject=None, reject_tmax=reject_tmax, on_missing='warn')

    print('ICA')
    ica_name = op.join(meg_dir, subject, 'run_concat-ica.fif')
    ica_out_name = op.join(meg_dir, subject, 'run_concat-ica-epo.fif')
    ica = mne.preprocessing.read_ica(ica_name)
    ica.exclude = []
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=-.3, tmax=.3, preload=False)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)
    del raw

    n_max_ecg = 3  # use max 3 components
    ecg_epochs.decimate(5)
    ecg_epochs.load_data()
    ecg_epochs.apply_baseline((None, None))
    ecg_inds, scores_ecg = ica.find_bads_ecg(ecg_epochs, method='ctps',
                                             threshold=0.8)

    print(f"Found {len(ecg_inds)} ECG indices")
    ica.exclude.extend(ecg_inds[:n_max_ecg])
    ecg_epochs.average().save(op.join(meg_dir, subject, 'ecg-ave.fif'), overwrite=True)
    np.save(op.join(meg_dir, subject, 'ecg-scores.npy'), scores_ecg)
    del ecg_epochs

    n_max_eog = 3  # use max 3 components
    eog_epochs.decimate(5)
    eog_epochs.load_data()
    eog_epochs.apply_baseline((None, None))
    eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)


    print(f"Found {len(eog_inds)} EOG indices")
    ica.exclude.extend(eog_inds[:n_max_eog])
    eog_epochs.average().save(op.join(data_path, 'eog-ave.fif'), overwrite=True)
    np.save(op.join(data_path, 'eog-scores.npy'), scores_eog)
    del eog_epochs    

    ica.save(ica_out_name, overwrite=True)
    epochs.load_data()
    ica.apply(epochs)

    print('Rejecting bad epochs')
    reject = get_rejection_threshold(epochs.copy().crop(None, reject_tmax))
    epochs.drop_bad(reject=reject)
    print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))

    print('Writing to disk')
    if frequency_band is not None:
        epochs.save(op.join(data_path, f'{subject}-{frequency_band}-epo.fif'), overwrite=True)
    else:
        epochs.save(op.join(data_path, f'{subject}-epo.fif'), overwrite=True)

    print("Epoching completed.")
    del epochs




parallel, run_func, _ = parallel_func(run_events, n_jobs=-1)
parallel(run_func(subject_id) for subject_id in range(3, 4))


parallel, run_func, _ = parallel_func(run_epochs, n_jobs=4)
parallel(run_func(subject_id) for subject_id in range(3, 4))
#FREQ_BANDS = {
#             "delta": [1, 4.5],
#             "delta_theta": [1, 8.5],
#             "theta": [4.5, 8.5]
#             "theta_alpha": [4.5, 15.5]
#             "theta_beta": [4.5, 30],
#             "alpha_beta": [8.5, 30],       
#             "alpha": [8.5, 15.5],
#             "beta": [15.5, 30],
#             "Gamma1": [30, 60],
#             "Gamma2": [60, 90]
#}

#subjects_ids=[i for i in range(1,17)]
# Create epochs for power bands
#for  frequency_band, f in FREQ_BANDS.items():
#    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
#    parallel(run_func(subject_id, f[0], f[1], frequency_band) for subject_id in range(2, 3))