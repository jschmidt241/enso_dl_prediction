# Applying deep learning techniques to predict ENSO phases

This work is adapted from Environmental ML coursework, updated and improved with 2026 data, more sophisticated tuning, and better hardware. Most recent studies point to an upcoming 'super' El Niño. Can a convolutional neural net that's only seen two 'super' events (1997 and 2015, using data from 1993-2026), trained on monthly sea surface temperature anomaly (SSTa) data, and targeting sea surface height (SSH) anomaly values get close to much more sophisticated methods?

Data used here is pulled from NOAA's SSTERv6 and NASA's SSH_ENSO_INDICATOR web storage, accessed 6/20/2026 at: 
* https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/
* https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/NASA_SSH_ENSO_INDICATOR/NASA_SSH_ENSO_INDICATOR_20260615.txt

<img width="1322" height="259" alt="image" src="https://github.com/user-attachments/assets/9cdf22f3-bc78-4879-a00d-a058edf75c4b" />
<img width="1389" height="390" alt="image" src="https://github.com/user-attachments/assets/87734247-c3a6-4932-9fec-42808a616aca" />

## Methods
The overall methodology followed here is:
* First align the SSH Index and normalized SSTa data, calculating median SSH for every month present in the SSTa dataset
* Only include SSTa values within the ENSO Standard Domain (120°E to 90°W, 20°S to 20°N):
* Create defined 'sequences' of consecutive SSTa data to train and predict with
* Split up training, test, and validation sets
* Settle on a rough CNN-LSTM model structure after some manual experimentation, looking for closely tied training and validation loss, high accuracy, precision, and recall on phase classes, and R^2 values.
* Setup a few hyperparameter tuning spaces and iterate
* Repeat with sequences that include lead-time buffers and predict multiple upcoming values
* Visualize results from the best-performing models

## Results
Here are a couple near-term predictions from the best-performing models:

<img width="1167" height="528" alt="image" src="https://github.com/user-attachments/assets/21ddddd3-8e25-4508-8f4f-94d21d5b7efa" />
Prediction from best-perfoming 3-month lead time model after hyperparameter tuning. This model used 6-month sequences, 32 2-dimensional kernels, a learning rate of 0.001, an LSTM unit value of 48, two dropout layers with rates of 0.3 and 0.2, and ReLU as an activation function.
<br><br>

<img width="1167" height="528" alt="image" src="https://github.com/user-attachments/assets/ae7a03c3-a3cb-4f39-806c-d35683d0a2f5" />
Prediction from the second-best 3-month lead time model after hyperparameter tuning. This model used 6-month sequences, 32 2-dimensional kernels, a learning rate of 0.0005, an LSTM unit value of 48, two dropout layers with rates of 0.3 and 0.2, and ReLU as an activation function.
<br><br>

<img width="1167" height="528" alt="image" src="https://github.com/user-attachments/assets/a0fa543e-b363-4c8e-8e2c-8a6181b858c7" />
Prediction from the best-performing 6-month lead time model after hyperparameter tuning. This model used 12-month sequences, a learning rate of 0.0005, 64 2-dimensional kernels, an LSTM unit value of 64, two dropout layers with rates of 0.2, and ReLU as an activation function.

## Discussion

Almost of the models that perform well on historic data also predict an El Niño event (as of 6/20, we are already in an El Niño phase, SSH_Index: 0.839265, but SSTa data has not yet been released so the model must predict based on May 2026 data). A 1-month lead time performed the best, reaching 85-87% accuracy accross the board and accurately predicting 5 out of 5 El Niño events in the test set. A 3-month lead time still performed well, reaching 65-85% accuracy across the board. The 3-month lead time only sometimes accurately predicted El Niño events, in some cases only predicting La Niña or Neutral phases. Interestingly, those models usually still predicted an El Niño sometime in June-August 2026. Finally, the 6-month lead time could not reach reasonable levels of performance. Phase prediction accuracy never exceeded 60% and could not even predict the much higher number of La Niña events in the test set.
<br><br>
So far, none of these models have predicted a 'super' El Niño event (SSH Index > 2) in the coming months. These preliminary results are still promising especially considering the limited input SSTa data, and future work could likely close that gap.

## Results (contd.)

More details from best performing models:


<img width="668" height="590" alt="image" src="https://github.com/user-attachments/assets/b7238965-ac8b-46d1-906f-f6e5c481dbe5" />
This was the best-performing model overall, reaching 80-87% accuracy depending on the random seed. This model was trained as a 1-month lead time phase classification task, not a phase intensity regression task.
<br><br>

<img width="668" height="590" alt="image" src="https://github.com/user-attachments/assets/f021ecd1-0d05-4cd9-9bdf-1a4c5c6c2b8c" />
This was the best-performing 3-month model, reaching 70-85% accuracy depending on the random seed. This model was trained as an SSH-anomaly regression task in order to plot its predicted upcoming phase.
<br><br>

<img width="668" height="590" alt="image" src="https://github.com/user-attachments/assets/a295749e-231c-46a2-9d78-7ed35073eb91" />
This was the best-performing 6-month model, varying from 32-60% accuracy depending on random seed. High variance and lower accuracy suggests it was unable to capture real ocean dynamics.
<br><br>

## Future Work
Next steps:
* Train a phase classification task predicting 'super' events (SSH Index > 2) with lead times > 1 month
* Continued work adjusting the CNN model capacity
* Try Huber loss instead of MSE to filter out some of the aggressive noise present in SST data.
* Find datasets with SSTa that go back before 1993
