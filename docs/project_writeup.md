# Pilot Cognitive Workload Monitor | Project Writeup

### Motivation and Purpose
Neuroergonomics is the study of the human brain in relation to performance at work and in everyday settings. [ScienceDirect](https://www.sciencedirect.com/topics/engineering/neuroergonomics)

Pilot cognitive workload is a prime example of one of its central applications. Aviation safety research has long since looked for ways to determine a pilot's workload during a task, especially since both states degrade performance and are hard to self-report reliably. 

This project builds a working prototype of that idea: given EEG recorded during a cognitively demanding task, can a system classify the workload level (low, medium, high) to a degree of usefulness, via the usage of signal processing and neuroscience rather than the usage of devices like a black box? It's framed around aviation specifically in fact. The target application is a pilot workload monitor but the underlying task data comes from a controlled non-flight cognitive load (see below), with a flight-simulator dataset held as the next validation step. (see Future Work)

Personally, I love traveling and wanted to do something aviation-related as a result but still make it an neuroengineering related project. 

### Data
**Source:** Hernández-Sabaté, A., Yauri, J., Folch, P., Álvarez, D., & Gil, D. (2024). <em>EEG Dataset Collection for Mental Workload Predictions in Flight-Deck Environment.</em> Sensors, 24(4), 1174. https://doi.org/10.3390/s24041174

The full collection of data includes three experiment types: N-back memory test, a "Heat-the-Chair" game, and an Airbus A320 flight-simulator scenario. This project in particular used the N-back test subset: 16 subjects, 3 workload conditions (1 = low/position task, 2 = medium/arithmetic task, 3 = high/dual task), each with baseline, task, and recovery phases.

**Hardware:** A Emotiv Epoc X EEG headset (channel names being AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4). This is a professional grade EEG headset that was used for each of the 3 experiments during the data collection process. 

### Methods

#### Preprocessing
The headset performs a documented onboard signal chain before the data is written out because of its own internal processes: 2048 Hz internal sampling, dual notch filtering at 50 Hz and 60 Hz, and an onboard low-pass filer around 64 Hz. This meant that there was not a notch filter applied and the sampling rate was determined empirically rather than calculated on its own.

Additionally, a 4-40 Hz band-pass filter and segmentation into fixed 2-second epochs (because the data had no stimulus markers to align to) and amplitude-based epoch rejection with a threshold derived from the data's peak-to-peak amplitude distribution were additional preprocessing steps taken. The band-pass filer purposely excludes delta waves because the dataset page remarked that the sensors only provide the power bands for theta, beta, and gamma. Also delta was excluded to better remove slow drift.

#### Feature extraction
Power spectral density (psd) was computed per epoch (Welch's method) and reduced to the average power per channel in five frequency bands: theta (4-8 Hz), alpha (8-12 Hz), beta-low (12-18 Hz), beta-high (18-25 Hz), and gamma (25-40 Hz). The power bands listed reflect the dataset paper's own analysis and the structure of the eeg headset (Section 2.1). 40 Hz was chosen as the upper limit for gamma because of the project's own band-pass filter. An earlier version of the pipeline included delta (following general convention rather the eeg device's own structure) and it featured as one of the most important features in that version. However it was eventually removed because it was not mentioned as part of the headset's band structure. The removal of it in this finalized pipeline is a methodology trade-off, and its effect on the accuracy is discussed in the Results section. 

Band power was converted to dB scale before usage as a model feature. One engineered feature was added in case of the presence of AF3/AF4 channels: frontal theta/beta ratio. Task phase 2 epochs were used for classification. Task 1 (baseline) and task 3 (recovery) were not included because they don't really represent a workload. 

**Notice:** The feature-extraction logic is implemented twice- in 04_feature_extraction.ipynb and gui/app.py rather than being located in spot. Both locations of logic were kept in sync manually during the development, however there is not currently code to enforce synchronization (note of the bug occurring is the Appendix). 

#### MATLAB Validation 
A randomly picked epoch's theta band power was computed independently with Python (Welch's method) and MATLAB (bandpower()) as a cross-check that the feature extraction pipeline created is computing something real (double checking my work essentially) and also to try out MATLAB's signal processing features. 
<ul>
<li>Python: 1.9655 x 10<sup>-12</sup></li>
<li>MATLAB: 2.7501 x 10<sup>-12</sup></li>
</ul>

The two methods differ by 39.92% which I hypothesize is because of having different spectral estimation approaches. A one-way ANOVA test on whether theta power differs significantly across the three workload levels was also run in MATLAB in compute_stats.m; ANOVA p-value = 4.756163 x 10 <sup>-75</sup>

#### Reasoning for MATLAB Comparison
Technically scipy could have reproduced the cross-check and the ANOVA via Python alone. However, MATLAB was included in order to showcase the ability to bridge two enviornments working together in one pipeline. Also because a significant amount of current neuroengineering involves MATLAB. However, MATLAB was not ultimately used for the feature extraction process because the MATLAB Engine API is not able to create a general M-by-N cell array which was significant enough of a constraint for me to not utilize it. 

##### SVC Attempt
A comparison against a Support Vector Classifier was also attempted but the accuracy of the model did not change significantly enough to convince changing the model from RandomForestClasifier to SVC.

#### Model
This project initially utilized cross-subject validation. One RandomForestClassifier trained on all 16 subjects pooled together, evalued with a GroupKFold (5 folds grouped by subject). A StandardScaler was fit independently on each subject's own epochs in order to normalize each subject and thereby reduce the influence of person-specific baseline EEG amplitude. This form of validation was useful for the case of: if the monitor is handed to a pilot it has never seen before, with zero prior data, how well would it work immediately? Eventually, it was concluded that the lack of accuracy (only about 37% between various iterations) required a change in validation in order for the monitor to be better applicable for the project's purposes.
**Note:** The .joblib file/model for cross-subject validation was removed in the final iteration of the project but the folder is being left there for future work which may need the cross-subject model and/or additional models saved in that folder.
**Note 2:** The cross-subject validation notebook is gitignored.

Within-subject validation was the final form of validation used in the project. 16 separate personalized models, on per subject, each trained and tested on that person's own data alone. A StratifiedKFold with 5 folds within each subject was utilized. This version of validation is useful for the case of: if there is already data from a certain pilot, how well can a model built just for them classify their future workload? 

The deployment implications for each validation type is discussed in the Future Work section. 

### Results

**Cross-subject: 37.9% accuracy, 37.7% macro F1**

Across 16 subjects and 20,769 task-phase epochs — meaningfully above the ~33.3% chance baseline for three balanced classes, and evaluated in a way that specifically guards against subject-identity leakage.

Per-class breakdown:

| Class | Precision | Recall | F1 | Support|
| ----- | --------- | ------ | -- | ------ |
|Low | 0.439 | 0.422 | 0.430 | 7157 |
| Medium | 0.358 | 0.297 | 0.325 |6694|
| High | 0.344 | 0.415 | 0.376 | 6918 |

Medium is consistently the model's weak point. Low recall (0.297) means most true "medium" epochs are predicted as something else, while high shows the opposite pattern (recall being 0.415, precision only being 0.344 so the model over-predicts "high"). This suggests the model leans toward calling ambiguous epochs "high" rather than "medium," plausibly because the medium task (arithmetic 1-back) and high task (dual 2-back) may share more EEG-level similarity with each other than either does with the simpler low task (position 1-back). This exact pattern- medium weakest, high showing recall well above precision were reproduced consistently across four separate training runs spanning different preprocessing pipelines during development, which is strong evidence for it being a genuine property of the data than any single confusion matrix would be on its own.

Feature importance: importance is distributed across many features rather than concentrated in one (top value 0.029, versus ~0.014 expected if all 71 features contributed equally). EEG.F7_theta, EEG.AF3_theta, and EEG.AF4_theta all rank highly, a direct, repeatable connection to frontal theta's established role in the workload. Gamma-band features across several scalp regions also rank prominently, while beta-low/beta-high features are notably absent from the top 10 despite the single merged beta band ranking highly in an earlier version of this pipeline. 

EEG.AF4_gamma appears among the top features, as it did in earlier pipeline versions. AF4 is a frontal electrode positioned near the eyes and was independently flagged during preprocessing for blink-artifact susceptibility. Some of what the model is using from that channel may be residual ocular artifact rather than pure cortical signal, noted directly rather than presenting feature importance as unambiguous evidence of a clean neural signal.

**Two documented methodological findings, not just a single result:**

First- an overfitting failure and its fix. An early version of this pipeline showed one engineered feature (the raw, unclamped frontal theta/beta ratio) receiving 100% of the trained model's feature importance while every other feature received 0%, alongside cross-validated accuracy of only 35.9% (barely above chance). That combination of the total importance concentration paired with near-chance held-out performance is the signature of an unregularized model memorizing an unstable, high-variance feature rather than learning anything generalizable: the ratio's denominator (frontal beta power) occasionally landed near zero, producing extreme outlier values the model could exploit almost like a unique row identifier. The fix was a data-scaling change of converting band power to dB scale and clamping the ratio feature which raised accuracy to 40.2% and created a more sensible, distributed feature importance pattern, without changing the model at all.

Second- a rigor-versus-accuracy trade-off, made deliberately. Every change made after that 40.2% run such as widening the high-pass filter edge for better drift removal (at the cost of losing delta entirely), deriving the rejection threshold per subject, adding per-subject feature normalization, and finally re-aligning the band scheme with this specific hardware's own structure instead of general convention were intentional decisions. Accuracy nonetheless settled at 37.9%, below that earlier peak. The most likely single cause is delta's removal: EEG.F7_delta ranked third in the 40.2% run's feature importance, and its absence here is a direct, known consequence of the 4 Hz high-pass filter rather than something offset by the other changes. 

**Within-subject: 94.9% accuracy**

Computed from the aggregated confusion matrix across all 16 subjects' personalized models (20,769 total epoch-condition predictions, same feature set and dataset as the cross-subject result above):

**Per-class breakdown (derived from the confusion matrix):**

|Class | Precision | Recall | F1 | Support|
|----- | --------- | ------ | -- | -------|
|Low | 0.959 | 0.945 | 0.952 | 7157
|Medium | 0.952 | 0.960 | 0.956 | 6694|
|High |0.936 | 0.941|0.939 | 6918 |

Unlike the cross-subject result, no class stands out as a particular weak point — all three sit in a tight 0.94–0.96 band. This itself is informative: the medium-vs-high confusion that dominates the cross-subject error pattern largely disappears once the model only ever has to distinguish one specific person's own states from each other, rather than generalize across 16 different people's individually varying signatures.

This aproximately 57-point gap between 37.9% and 94.9% is not me saying that "the second model is better". Rather it's the quantified cost of the cross-subject generalization problem discussed throughout this project.

### Limitations
The Emotiv Epoc X headset does not have the same accuracy as higher grade EEG headsets.

Small subject count. 16 subjects is workable for a subject-grouped 5-fold evaluation but is a real constraint on how confidently the cross-subject results generalize to a broader population.

Truncated gamma band. The 30–40 Hz gamma feature reflects this project's own preprocessing filter, not this device's actual onboard gamma range (left unbounded above 25 Hz in the source paper).

Delta excluded, with a measurable cost. Delta's absence follows directly from the 4 Hz high-pass filter and this device's own band structure, and is a large contributor to the accuracy gap between this pipeline's 37.9% cross-subject result and an earlier version's 40.2% (see Results) but nonetheless a deliberate trade-off.

Possible artifact contamination in top features. EEG.AF4_gamma consistently ranks among the most important features across pipeline versions; AF4 is a frontal electrode positioned near the eyes and was personally independently flagged during preprocessing for blink-artifact susceptibility.

Feature extraction logic is duplicated, not shared, between 04_feature_extraction.ipynb and gui/app.py. Kept manually in sync throughout development; a real drift risk for any future change (see Methods and the Appendix).

The within-subject result, while methodologically sound in its held-out evaluation, still requires substantial future work before it represents a real deployment scenario as the labels it uses come from a controlled experiment design, not from anything a real pilot's calibration session would produce on its own (see Future Work).

Validated on a controlled cognitive task, not real flight. The N-back paradigm induces workload in a controlled, repeatable way, but hasn't yet been validated against the flight-simulator portion of the same dataset collection which was the more directly aviation-relevant test (see Future Work). However that portion was not utilized in this project because it was a very small amount of data.

### Future Work

#### Remarks upon goals for the project in the future. 

Cross-subject >> zero-setup and fleet-wide deployment. Install it in every cockpit, any pilot sits down, it works immediately. The ~38% accuracy is the real, quantified cost of that ambition. With more data that also contains more applicable data (more flight simulator type data), the accuracy should be expected to increase.

Within-subject >> put the EEG on one pilot, personalize a model to just them. The GUI's "Calibrate Subject" feature is a small working prototype of exactly this, not just a proposal and a genuine held-out split (the first 20% of each workload condition trains the model, live predictions are only ever shown for the remaining, unseen 80%) makes a small-scale demonstration. It was this concept that ended up appealing more to me over the cross-subject idea. 

A genuine calibration protocol, not borrowed labels from one source paper. This project's calibration works because it already has workload labels from the N-back experiment design. A real pilot doesn't arrive with pre-labeled workload data and therefore a deployable version needs the pilot to complete something like a short, controlled pre-flight task (essentially recreating the N-back protocol itself) before a personalized model can exist at all.

The cold-start trade-off, named rather than resolved. Personalization buys accuracy at the cost of calibration time before every first use; generalization costs accuracy but requires no time to calibrate. Both are legitimate versions and hopefully can be implemented together in the future. 

A hybrid, adaptive model as the natural next step beyond a binary choice. Start every new pilot on the cross-subject model, then progressively fine-tune toward a personalized one as that pilot accumulates their own labeled flight history over time (for example, new pilots would use the cross-subject model more often but pilots with lots of experience would have a personalized model). 

#### Other directions:

More subjects. The most significant contribution to the cross-subject model's actual bottleneck is the lack of individual EEG variability, not artifact contamination or model choice (see Appendix for why ICA was considered and set aside). More subjects gives the model a broader sample of how workload manifests across different brains.

Higher-quality (research-grade) recording hardware, as a separate lever from subject count would better help with filtering processes.

Consolidate the duplicated feature-extraction logic between 04_feature_extraction.ipynb and gui/app.py into one shared module (see Appendix).

Test on the flight-simulator subset of the same dataset collection. Training here (N-back) and testing there (A320 simulator) mirrors the transfer-learning approach the dataset's own authors used, and is the closest thing to an actual aviation-relevant result available without new data collection.

Temporal modeling. The current models classify each 2-second epoch independently but real workload doesn't change instantaneously, so a model incorporating a rolling window of recent epochs is a more realistic next iteration.

Live hardware / real-time demo. The GUI currently replays pre-recorded, pre-processed epochs; a genuine real-time streaming version would require live acquisition and processing. However, this would be a very interesting upgrade and likely very necessary for actual real-world application. 

More Libraries. Incorporate libraries such as TensorFlow or PyTorch. Includes incorporating more of MATLAB.

**NOTE:** I did end up using Python for the majority of this project. I used Jupyter Notebooks instead of MATLAB because I found the segmentation of Jupyter much easier for me to manage but I do hope to be able to use MATLAB more handily in further projects. 

### Appendix: Development Log

Kept for reference: a chronological record/journal of the actual debugging journey, not just the polished result above. Future-me will thank present-me for writing all this down.

**Phase 1–2: Getting oriented, and the sampling-rate.** The raw dataset turned out to already be in pandas-friendly .parquet format, with real column names (EEG.AF3 etc.) confirmed by actually printing them. Computing the true sampling rate from the data took a total three failed attempts before landing on the right approach: .mean() of timestamp diffs was distorted by multi-second gaps between recording blocks while .median() returned zeroes because the timestamp column's rounding meant the majority of consecutive diffs were exactly 0.0. Filtering diffs to a "plausible" small range before taking the median still failed, because the real jumps in this data were multi-second block boundaries, not small per-sample steps and there was no clean small-vs-large split to filter on. The actual fix abandoned inferring the rate from timestamps entirely: Emotiv hardware outputs a fixed rate (128 or 256 Hz), so the right move was arithmetic proof (a known-duration segment's sample count divided by 128 matched its real-world duration; divided by 256 didn't) which matched the dataset paper's own stated 128 Hz figure. **Lesson:** Learning how to calculate sampling rate for future cases of not being able to find or being given a sampling rate. However, a given constant is obviously much more helpful and quickens the coding process. 

**Phase 2: the notch filter that didn't need to exist.** The dataset paper's device description revealed the headset already applies a dual 50/60 Hz notch onboard, before data ever reaches this project. Re-applying one in preprocessing would have been redundant, not harmful — but including it anyway would have suggested a misunderstanding of the data pipeline to anyone reviewing the methods closely. **Lesson:** Despite previous beliefs that I still needed to refilter the dual notch, I learned that I did not need to do that again and instead just use my own separate filtering techniques that complement any prior ones. 

**Phase 4: the overfitting bug, and what actually fixed it for me.** An early feature set gave one engineered feature (an unclamped frontal theta/beta ratio) 100% of a trained Random Forest's importance, with every other feature at 0%, alongside near-chance accuracy (35.9%). The ratio's denominator occasionally landed near zero, producing extreme outlier values the model could exploit almost like a unique row ID. The fix was dB-scaling the band power and clamping the ratio which raised accuracy to 40.2% and fixed the importance distribution, with zero change to model regularization. **Lesson:** a single feature dominating importance while accuracy stays near chance is a specific, recognizable signature of memorization on unstable feature scale, not "that feature is just really good." It is better for the feature importance to be quite distributed.

**Phase 3: MATLAB Engine API's cell-array limit.** Calling anova1 live via the Engine API with nargout=3 (requesting p, tbl, stats) crashed with ValueError: cell arrays returned from MATLAB must be 1-by-N or M-by-1. tbl is a multi-row, multi-column cell array and unfortunately a shape the Engine API cannot convert well back to Python at all from a structural standpoint. The fix was simply requesting nargout=1 (only p, which was the only value actually used downstream). **Lesson:** understanding a significant and specific constraint of cross-language via the usage of MATLAAB Engine API.

**Phase 4, band scheme: three separate rounds of revision.** Started with a generic-literature band scheme (merged beta, delta included) that produced the 40.2% peak result above. However this differed from the dataset paper's own device description (Section 2.1) which described that the headset's actual onboard band structure splits beta into two sub-bands and doesn't report delta at all which eventually convinced me to do a deliberate switch to match the hardware/paper exactly, despite initially worrying about differing from my conception of convention. Also, I found that "match the actual source being used" was a more defensible choice than either option being objectively "correct." This, combined with independently re-deriving the rejection threshold per subject and adding per-subject feature normalization, changed several things simultaneously which resulted in a couple of big changes. **Lesson:** comparing 40.2% to the final 37.9% conflated at least four independent changes, and isolating delta's removal as the most likely single cause required going back and checking which specific feature had previously ranked highly as a sanity check.

**On the "should accuracy be 70-90%?" question.** External personal research suggested the ~38% cross-subject accuracy was too low and a different model (SVC) would fix it. Checking actual published literature instead of guessing: properly-evaluated (leave-one-subject-out or subject-grouped) cross-subject EEG workload classification consistently shows large accuracy drops compared to within-subject or leakage-prone evaluation schemes. One directly comparable study reported subject-specific accuracy of 96.7% against a true leave-one-out result of 67% on a similar 3-class situation while another found switching from randomized to proper leave-one-subject-out cross-validation cost over 20 accuracy percent points; a third found cross-subject models on a different task reaching literal chance level. This project's own within-subject (94.9%) vs. cross-subject (37.9%) gap directly reproduced that same pattern. **Lesson:** a modest number under rigorous evaluation is not automatically a sign of doing something wrong or a bug and that I should check what comparable rigorous evaluations actually report before assuming the number itself is the problem.

**Phase 5/6: the GUI calibration leakage bug, and its fix.** An early version of the GUI's "Calibrate Subject" feature fit a personalized model on 100% of a selected subject's data, then displayed live "predictions" on that exact same data during playback which is really just pure memorization made to look like a real-time demo, unrelated to the legitimate 94.9% within-subject evaluation result computed properly in the notebook. The fix (src/ml_engine.py) introduced a held-out split but the first version of that fix wasn't stratified per workload condition, and since a subject's epochs are naturally loaded in blocks (all-low, then all-medium, then all-high), it occurred to me that a single front/back percentage cutoff risked leaving one or more classes entirely absent from either the training portion or the displayed portion. The corrected version splits within each condition separately, and was also disconnected entirely from full_features.csv and instead training directly on the exact same live-extracted features already being displayed. A final, small typo (is_train_row vs. is_train_now — two different attribute names, easy to type differently without noticing) caused a few more TypeErrors before the fix was fully working. **Lesson:** Always read through your code once or twice or 15 times because as any programmer can expect, there's always a bug somewhere. 

#### Final Learning Notes:
I learned about pull requests to GitHub, tagging versions, scipy, scikit-learn, Jupyter Notebook, beginner EEG processing techniques, MATLAB's Signal Processing Toolbox, and beginning machine learning techniques. 

### Additional Helpful Sources 
[Frontal Theta/Beta Ratio Understanding](https://doi.org/10.1016/j.biopsycho.2018.03.002)

[Superscript in Markdown](https://meta.stackexchange.com/questions/226869/how-can-i-add-the-mathematical-symbol-for-power-like-x-2-to-a-question)
[MNE-Python Documentation](https://mne.tools/stable/documentation/index.html)
[MATLAB Documentation](https://www.mathworks.com/help/matlab/index.html)

[MikeXCohen](https://www.youtube.com/@mikexcohen1)
[StatQuest](https://www.youtube.com/@statquest)
[3Blue1Brown](https://www.youtube.com/@3blue1brown)

Various google searches of "beginners guide to machine learning" and "beginners guide to processing EEG data".

**AI Usage Note:** AI (specifically Claude) was used in this project for figuring out errors that I could not figure out on my own or via familiar sources like StackOverflow, further explaining parts that my own self-learning was not able to fully breakdown, and double-checking parts of my code at certain points for streamlining purposes and hopefully preventing some bugs. 