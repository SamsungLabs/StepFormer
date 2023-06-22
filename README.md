# unsup-step-pred
Transformer-based unspervised insturction step prediction model, called StepFormer. It is built as a transformer decoder with learnable queries; the queries attend to the video and producde step slots that are likely to represent instruction steps.
StepFormer is trained on a large (unlabeled) instructionla video dataset, HowTo100M, and uses narration text as a supervisory signal, to extract procedure steps from the video.

1. Setup: make sure that all the paths in `paths.py` point to the appropriate folders. Also, make sure to install all the necessary packages, such as pytorch, pytorch-lightning, wandb (optional, can be switched from flags) and webdataset.
2. Training: run `python train.py` to train the best performing model from the submission
3. Evaluation: run `python test.py` to get unsupervised segmentation and zero-shot step localization performance, as reported in Tables 1 and 2 of our sumbission.
4. Demo: open `demo.ipynp`
