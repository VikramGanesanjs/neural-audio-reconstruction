import scipy.io
import scipy.signal
import librosa
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np



NUM_PATIENTS = 29
patient_data = []


spectrogram_mat = scipy.io.loadmat("../pink-floyd-data/thewall1_stim128.mat")
spectrogram_128 = spectrogram_mat['stim128']

class HFA_MAT:
    def __init__(self, mat, coords_mat):
        self.artifacts = mat['artifacts']
        self.electrode_locations = coords_mat['elec_mni_frvr']['elecpos'][0][0]
        self.ecog = mat['ecog']
        self.dataInfo = mat['dataInfo']
        self.shape = self.ecog.shape
        self.spectrogram = spectrogram_128

    def remove_artifact_electrodes(self, ecog, artifacts):
        indices = []
        if(np.sum(artifacts) == 0):
            return ecog
        for i in range(artifacts.shape[1]):
            for j in range(artifacts.shape[0]):
                if(artifacts[j][i] == 1):
                    indices.append(i)

        return np.delete(ecog, indices, 1)

for i in range(NUM_PATIENTS):
    mat = scipy.io.loadmat(f"../pink-floyd-data/P{i+1}_HFA_data.mat")
    coords_mat = scipy.io.loadmat(f"../pink-floyd-data/P{i+1}_MNI_electrode_coordinates.mat")
    hfa = HFA_MAT(mat, coords_mat)
    patient_data.append(hfa)




fs = 100
num_rows_total = patient_data[0].shape[0]


electrode_positions = {
    'superior_temporal_gyrus': [-59, -24, 1],
    'middle_temporal_gyrus': [-59, -29, -3],
    'inferior_frontal_gyrus':[45, 48, -2],
}

class PFDataset(Dataset):
    def __init__(self, full_patient_data,segment_length_seconds, channels_crop=40, spectrogram_transform=True):
        self.ecog_segments = []
        self.stim_labels = []
        self.segment_idxs = []
        self.ecog_spectrogram_segments = []
        self.spectrogram_transform = spectrogram_transform
        self.full_patient_data = full_patient_data
        self.channels_crop = channels_crop
        self.patient_ids = []
        self.electrode_locations = []
        self.location_categories = []
        self.segment_length_seconds = segment_length_seconds
        if(self.spectrogram_transform):
            self.create_spectrogram_data()
        else:
            self.create_data()
    def __len__(self):
        return len(self.stim_labels)

    def create_data(self):
        for pat_id, data in enumerate(self.full_patient_data):
            for i in range(int(np.floor(num_rows_total / (self.segment_length_seconds * fs)))):
                random_channels_start = int((data.shape[1] - self.channels_crop) * random.random())
                start_idx = int(i * self.segment_length_seconds * fs)
                end_idx = start_idx + int(self.segment_length_seconds * fs)
                ecog_segment = data.ecog[start_idx:end_idx, random_channels_start:random_channels_start + self.channels_crop]
                self.electrode_locations.append(data.electrode_locations[random_channels_start:random_channels_start + self.channels_crop, :])
                stim_segment = data.spectrogram[start_idx:end_idx, :]
                self.ecog_segments.append(ecog_segment)
                self.patient_ids.append(pat_id)
                self.stim_labels.append(stim_segment)
                self.segment_idxs.append([start_idx, end_idx])

    def create_spectrogram_data(self):
        for pat_id, data in enumerate(self.full_patient_data):
            full_spectrograms, location_categories = self.input_spectrogram_transform(data.ecog, data.electrode_locations)
            if(len(location_categories) > 0):
                for i in range(num_rows_total // int(self.segment_length_seconds * fs)):
                    start_idx = int(i * self.segment_length_seconds * fs)
                    end_idx = start_idx + int(self.segment_length_seconds * fs)
                    ecog_segment = data.ecog[start_idx:end_idx, :]
                    ecog_spectrogram_segment = full_spectrograms[:, start_idx:end_idx, 0]
                    stim_segment = data.spectrogram[start_idx:end_idx, :]
                    if(ecog_spectrogram_segment.shape[0] == 200 and ecog_spectrogram_segment.shape[1] == 150):
                        print(ecog_spectrogram_segment.shape)
                        self.patient_ids.append(pat_id)
                        self.ecog_segments.append(ecog_segment)
                        self.stim_labels.append(stim_segment)
                        self.location_categories.append(location_categories)
                        self.segment_idxs.append([start_idx, end_idx])
                        self.ecog_spectrogram_segments.append(ecog_spectrogram_segment)


    def get_data_shape(self):
        return self.__getitem__(0)[0][0].shape

    def get_label_shape(self):
        return self.stim_labels[0].shape

    def input_spectrogram_transform(self, ecog_data, electrode_locations):
        num_bins = 200
        num_timesteps = ecog_data.shape[0]
        f_min = 0
        f_max = 50

        geographical_electrodes, location_categories = self.get_geographical_electrodes(electrode_locations)
        transformed_ecog = np.zeros([num_bins, num_timesteps, len(geographical_electrodes.keys())])
        
        for _, electrode_idxs in geographical_electrodes.items():
            k, i = 0, 0
            electrodes_composed = np.zeros([ecog_data.shape[0], len(electrode_idxs)])
            for idx in electrode_idxs:
                electrodes_composed[:, i] += ecog_data[:, idx]
                i += 1
            spectrogram_per_localization = np.zeros([num_bins, num_timesteps])
            for j in range(electrodes_composed.shape[1]):
                mel_spec = librosa.feature.melspectrogram(y=electrodes_composed[:, j], sr=fs, hop_length=(electrodes_composed.shape[0]//num_timesteps),
                  n_fft=250, n_mels=num_bins, fmin=f_min, fmax=f_max, win_length=num_bins)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:num_bins,:num_timesteps]
                spectrogram_per_localization += mel_spec_db
                spectrogram_per_localization /= electrodes_composed.shape[1]
                
            
            transformed_ecog[:, :, k] += spectrogram_per_localization
            k += 1
        return transformed_ecog, location_categories


    def get_geographical_electrodes(self, electrode_locations):
        res = {
            'superior_temporal_gyrus': [],
            'middle_temporal_gyrus': [],
            'inferior_frontal_gyrus': [],
        }
        for i in range(electrode_locations.shape[0]):
            for electrode_name, electrode_position in electrode_positions.items():
                dist = np.linalg.norm(np.array(electrode_locations[i]) - np.array(electrode_position))
                if (dist < 30):
                    res[electrode_name].append(i)
        

        arr = []
        for electrode_name, electrode_idxs in res.items():
            if(len(electrode_idxs)>0):
                arr.append(electrode_name)
        
        return res, arr

    def __getitem__(self, index):
        ecog = None
        patient_id = self.patient_ids[index]
        location_categories = None
        if(self.spectrogram_transform):
            ecog = self.ecog_spectrogram_segments[index]
            location_categories = self.location_categories[index]
        else:
            ecog = self.ecog_segments[index]
        stim = self.stim_labels[index]
        segment_idx = self.segment_idxs[index]
        return (ecog, location_categories, segment_idx, patient_id), stim
