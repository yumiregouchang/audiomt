"""Names shared by the command line and the modules.

Kept free of heavy imports, so `audiomt` can load torch before scikit-learn, librosa or sqlite
load their DLLs (see cli._preload_torch).
"""
SUBSETS = ("esc50", "esc10")
MODEL_NAMES = ("mfcc_svm", "ast_logreg")
