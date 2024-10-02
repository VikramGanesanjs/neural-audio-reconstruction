# Finding Useful Latent Representations for Auditory Cortex Signals

In this project, I explored the effectiveness of Variational Autoencoders (VAE) to create latent representations of auditory cortex signals. The accurate interpretation of auditory cortex signals is very important with regard to Brain Computer Interface (BCI)  based speech synthesis for patients with mobility disorders, but it is difficult due to the inherent heterogeneity of auditory cortex signals. VAEs could solve this problem, by creating latent-representations of the data which can generalize between patients and tasks. 


For this experiment, I used datasets from stimulus reconstruction and speech synthesis tasks. I developed a model architecture which combines a traditional VAE with a regression to decode the spectrogram stimulus, and trained the models simultaneously. Due to memory and computing constraints, I had to downsample the data significantly before training. 


The model was able to accurately reconstruct spectrograms from the speech synthesis dataset, however it failed to encode any temporal information for the stimulus reconstruction dataset. The latent representations that it created generally have the same signal power across all frequency bands; this could indicate that a representation where all frequency bands are equal in power is a more efficient representation of auditory cortex signals. The model was computationally efficient after training, showing that it would fit into a real-time speech synthesis BCI. 


The results I obtained in this experiment were heavily limited by my computing constraints. If granted time on a high performance computing cluster, I could train my model on full datasets and discover its true performance; I would also experiment with the effectiveness of other deep learning model architectures, like transformer based models. I hope to continue investigating auditory cortex signal processing projects in the future because of their huge importance in the creation of effective, real-time, speech synthesis devices for those who cannot communicate. 


Link to full Extended Essay: https://docs.google.com/document/d/1NWbJizTeQLKHJwxB3YByFHYNLqs-358Ze06R_cJDzTw/edit?usp=sharing
