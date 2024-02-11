import matplotlib.pyplot as plt
import librosa
from torch.utils.data import DataLoader

from pf_dataset import PFDataset, patient_data

SEGMENT_LENGTH = 1.5 #(seconds)

pf_dataset = PFDataset(patient_data, SEGMENT_LENGTH)

(sample_datapoint, location_categories, segment_idx, patient_id), sample_stim = next(iter(DataLoader(pf_dataset, shuffle=True)))
print(location_categories)
sample_datapoint = sample_datapoint.view(200,150)
sample_stim = sample_stim.view(150, 128)
fig, ax = plt.subplots()
img = librosa.display.specshow(np.array(sample_datapoint[:, :]), sr=100, x_axis="s", y_axis="linear", hop_length=1, fmin=0, fmax=50)
ax.set(title=f'{location_categories[0][0]} for patient {patient_id.item() + 1} from {(segment_idx[0].item() // 100) // 60}:{((segment_idx[0].item() // 100) % 60)}-{(segment_idx[1].item() // 100) // 60}:{((segment_idx[1].item() // 100) % 60)}')
plt.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(f'../figures/P{patient_id.item() + 1}_{(segment_idx[0].item() // 100) // 60}:{((segment_idx[0].item() // 100) % 60)}-{(segment_idx[1].item() // 100) // 60}:{((segment_idx[1].item() // 100) % 60)}_{location_categories[0][0]}.png')
