# Coarse-to-Fine-Grained Image Classification

**COMP30027 Machine Learning, Project 2 (2026)**

---

## Abstract

This report investigates how feature representation and classifier choice interact with class granularity in image classification. On a coarse-grained 10-animal task and a fine-grained 10-bird-species task, hand-crafted features (colour histogram, HOG, LBP, HSV) achieve ~52 % accuracy on the coarse task but collapse to ~24 % on the fine-grained one, while frozen ImageNet ResNet-18 embeddings hold at ~88 % across both. The best classifier flips between tasks — RBF SVM on coarse, Random Forest on fine-grained — and remaining errors concentrate almost entirely on visually similar species pairs (warblers, gulls) that the dataset designers flagged as the genuinely hard distinctions.

## 1. Introduction

A central difficulty in visual recognition is that classification becomes dramatically harder as classes become visually similar to one another. This report explores that progression directly through two sequential tasks. Task 1 classifies 64×64 images into ten visually distinct animal categories (bird, butterfly, cat, deer, dog, elephant, frog, horse, sheep, spider), with roughly 3,750 training images. Task 2 classifies 128×128 images into ten bird species drawn from CUB-200-2011 (cardinal, blue jay, American goldfinch, red-winged blackbird, house sparrow, song sparrow, herring gull, ring-billed gull, yellow warbler, Wilson warbler), with only ~420 training images. The bird species share body plan and differ only in subtle markings — bill shape, head colouring, breast streaks — and were chosen so that some pairs (the two gulls; the two yellow warblers) are intentionally close to indistinguishable.

I treat the two tasks identically — same evaluation harness, same model families, same hyperparameter grids — so that any difference between them is attributable to either the data (granularity, size) or to the features themselves. This setup lets me ask three questions concretely:

1. How much do increasingly powerful feature representations contribute at each granularity?
2. Does the best classifier choice transfer from coarse to fine-grained tasks?
3. Where do the residual errors live, and what does their structure say about the limits of current feature representations?

The short answers, developed below: hand-crafted features lose more than half their accuracy under the granularity shift, ImageNet-pretrained CNN features barely lose any; the best classifier *does not* transfer (SVM-RBF wins Task 1, Random Forest wins Task 2); and almost all of the remaining Task 2 errors fall on the two species pairs the dataset designers explicitly flagged as similar.

## 2. Methodology

**Feature representations.** I evaluate three groups of features:

*Provided features* — colour histograms (RGB bins), HOG features extracted from grayscale and reduced via PCA, and additional features (edge density, texture variance, channel means). These were supplied with the dataset.

*Engineered features* — Local Binary Patterns histograms on grayscale (P=8, R=1, uniform method, 10 bins) capture local texture (feather patterning, fur, skin) distinct from HOG's gradient-orientation signal. HSV per-channel mean and standard deviation (6 dimensions) decouple brightness from hue, which the RGB-binned provided histogram tangles together. A saturation-weighted hue histogram (24 bins for Task 1, 36 bins for Task 2) gives a more sensitive colour-distribution signal; weighting by saturation suppresses contribution from near-grey pixels where hue is poorly defined. The finer Task 2 binning reflects that bird species are heavily colour-defined (cardinal red, jay blue, two yellow warblers, etc.).

*CNN features* — 512-dimensional embeddings from a ResNet-18 [1] pretrained on ImageNet [2]. The classifier head is removed and the post-average-pooling vector is used as a fixed feature. Images are normalised with ImageNet statistics and upsampled (64→224 for Task 1; 128→232→centre-crop-224 for Task 2). The model was trained only on ImageNet, satisfying the spec's prohibition on extractors pretrained on CIFAR-10 or CUB-200-2011. No fine-tuning is performed; the CNN is used purely as a frozen feature extractor.

**Models.** Three algorithms from distinct families are tuned and compared: Logistic Regression (linear, log-loss), Support Vector Machine with RBF kernel (kernel-based, non-linear), and Random Forest (bagged decision-tree ensemble). A DummyClassifier predicting the majority class is included as a sanity baseline. Each model sits in a Pipeline behind StandardScaler so that feature standardisation is fit inside each CV fold.

**Evaluation.** Five-fold stratified cross-validation drives both hyperparameter selection and final reporting. For every (feature subset × model) combination, GridSearchCV with multi-metric scoring returns mean accuracy and mean macro-F1 from a single CV pass. Grids are deliberately compact (LogReg: C ∈ {0.01, 0.1, 1, 10}; SVM: C ∈ {0.1, 1, 10} × γ ∈ {scale, 0.001, 0.01}; RF: n_estimators ∈ {300, 500} × max_depth ∈ {None, 20} × max_features ∈ {sqrt, log2}) — these regions cover where each algorithm peaks on this data without inflating compute on extremes that never win. Out-of-fold predictions, used for confusion matrices and per-class reports, are computed only for the (feature, model) combinations that appear in the figures, not for every grid cell. The final Kaggle submission refits the overall best (feature, model, HP) configuration on the entire training set.

## 3. Results

Table 1 summarises 5-fold CV accuracy across feature subsets and learners for both tasks; macro-F1 tracks accuracy closely on both (Δ < 0.005) so it is omitted here for compactness.

**Table 1.** Five-fold CV accuracy by feature subset and model. Bold = best for that feature subset.

| Feature subset       | LogReg | SVM-RBF | RF        | Dummy |
| -------------------- | ------ | ------- | --------- | ----- |
| **Task 1 (coarse-grained, n=3,750)** | | | | |
| provided             | **0.532** | 0.525   | 0.524     | 0.100 |
| provided + extra     | **0.591** | 0.583   | 0.560     | 0.100 |
| cnn_only             | 0.878  | **0.880** | 0.833    | 0.100 |
| provided + cnn       | 0.874  | **0.879** | 0.836    | 0.100 |
| all                  | 0.878  | **0.883** | 0.842    | 0.100 |
| **Task 2 (fine-grained, n=417)** | | | | |
| provided             | 0.240  | 0.235   | **0.321** | 0.096 |
| provided + extra     | 0.343  | 0.333   | **0.405** | 0.096 |
| cnn_only             | **0.866** | 0.863   | **0.866** | 0.096 |
| provided + cnn       | 0.854  | 0.849   | **0.858** | 0.096 |
| all                  | 0.842  | 0.835   | **0.878** | 0.096 |

The overall best configurations are: Task 1 **all + SVM-RBF, accuracy 0.883, macro-F1 0.883** (C=10, γ=0.001); Task 2 **all + Random Forest, accuracy 0.878, macro-F1 0.876** (n_estimators=500, max_depth=None, max_features=sqrt). Both clear the 50 % Kaggle thresholds comfortably.

The per-class breakdown on Task 1 (Figure 1) shows easy classes — elephant (F1 0.97), sheep (0.97), butterfly (0.95), spider (0.95) — and a cluster of harder ones centred on the mammalian quadrupeds: cat (0.78), dog (0.81), bird (0.80), deer (0.83). The single largest off-diagonal mass is dog→cat (14 %) with cat↔dog symmetric.

Task 2 shows much sharper bimodality (Figure 2). Four classes are near-perfect (Blue_Jay 0.99, Cardinal 0.99, American_Goldfinch 0.98, Red_winged_Blackbird 0.98) and four are visibly weaker (Wilson_Warbler 0.74, Yellow_Warbler 0.74, Ring_billed_Gull 0.77, Herring_Gull 0.78). Table 2 ranks the top off-diagonal confusion cells.

**Table 2.** Top 5 confused pairs on Task 2 best configuration. `recall_lost` = fraction of true class instances misclassified into the predicted class.

| True                  | Predicted             | Count | Recall lost |
| --------------------- | --------------------- | ----- | ----------- |
| Yellow_Warbler        | Wilson_Warbler        | 12    | 0.29        |
| Ring_billed_Gull      | Herring_Gull          | 10    | 0.24        |
| Wilson_Warbler        | Yellow_Warbler        | 9     | 0.21        |
| Herring_Gull          | Ring_billed_Gull      | 8     | 0.19        |
| House_Sparrow         | Song_Sparrow          | 4     | 0.10        |

The two warbler↔warbler and two gull↔gull cells alone account for 39 of the 51 total errors (76 %).

## 4. Discussion and Critical Analysis

**Feature representations are the dominant factor — and they degrade unevenly with granularity.** The provided hand-crafted features achieve 0.53 on Task 1 (more than 5× the random baseline) but collapse to 0.24 on Task 2, only 2.5× over a 0.10 baseline. Adding the engineered LBP + HSV features lifts both tasks comparably in absolute terms (+0.06 on Task 1, +0.08 on Task 2), but the deficit relative to CNN features is enormous: CNN-only features achieve 0.88 on Task 1 and 0.87 on Task 2, an essentially flat curve across granularity. The conventional reading — that fine-grained classification is hard because of subtle inter-class differences — turns out to be conditional on the feature representation. Subtle differences are only subtle in the representation space we choose. ImageNet's hundreds of bird species in its 1000-class label set push ResNet's early-to-mid convolutional features to encode plumage texture, beak silhouette, and head colouring as discriminable directions; once we live in that feature space, the fine-grained distinctions are no longer hidden. Hand-crafted features, by contrast, are tuned to *low-level* statistics — overall colour distribution, gradient orientation histograms, simple texture descriptors — which capture useful global signals when classes differ globally (a butterfly vs a horse) but cannot resolve features that depend on local high-level structure (a black cap, a bill shape).

**The best classifier flips between tasks.** SVM-RBF wins decisively on Task 1 (0.883 vs RF's 0.842), but Random Forest wins on Task 2 (0.878 vs SVM-RBF's 0.835 with the same all-features set, 0.863 with cnn_only). Two pieces of theory help explain this. First, the bias–variance trade-off shifts with sample size: with 3,750 training images, SVM-RBF can locate a smooth, low-bias decision boundary in the 569-dimensional combined feature space and benefits from RBF kernels' flexibility; with only ~330 training images per fold on Task 2, the same kernel produces a high-variance estimate sensitive to which support vectors end up in any given fold. Random Forest mitigates this through bagging plus the random feature subset at each split — built-in regularisation that is particularly valuable when data is scarce. Second, RF is invariant to feature scaling and feature heterogeneity: its splits are univariate, so combining a 512-d CNN block, a 50-d hand-crafted block, and an 18-d provided block does not require those blocks to share a common scale. SVM-RBF's Gaussian kernel takes a Euclidean distance over the joint feature vector, so a noisy hand-crafted block can dilute the contribution of more informative CNN dimensions — which is exactly what the Task 2 table shows for SVM-RBF (cnn_only 0.863, all 0.835: adding "more features" hurts the kernel model).

**Errors cluster on dataset-designer-flagged pairs.** Table 2 is the single most informative artefact of this experiment. Of 51 total Task 2 errors on the best configuration, 39 (76 %) come from just four cells: yellow↔Wilson warbler and herring↔ring-billed gull. The dataset documentation explicitly described Ring-Billed Gull as "very similar to Herring Gull, subtle bill differences" and the two yellow warbler species are visually distinguishable mainly by the Wilson Warbler's black cap. The model is therefore failing in *exactly* the places that human experts also find difficult, and succeeding everywhere else; the residual error is not noise but signal. Two structural observations follow. First, the classes that fail are precisely those whose discriminative feature is *local and small* (a cap, a bill detail) relative to the bird's overall body — a global average-pooled CNN representation pools away this signal. A fine-grained-specific architecture (part-based attention, fine-scale pooling, or simply higher input resolution) is the natural next step. Second, the warbler-pair and gull-pair errors are symmetric and roughly equal in both directions, which is consistent with the model treating each pair as a single class with two labels — i.e. the embeddings are nearly co-located in feature space, not biased to one side.

**Task 1 errors are qualitatively different.** The largest off-diagonal on Task 1 — dog→cat at 14 % — is also visually motivated (small dogs and cats at 64×64 share silhouette and pose) but the confusion is much more diffuse, spanning multiple class pairs at modest magnitudes (cat→dog 0.09, deer→horse 0.05, bird→deer 0.07). Task 2 errors, by contrast, are concentrated in two symmetric pairs that *exhaust the space of plausible confusions*. The structural difference is informative: a coarse-grained model fails on the long tail of the visual-similarity distribution, where many class pairs contribute a few errors each; a fine-grained model fails on a small, dataset-specific set of label pairs whose feature-space images overlap. This shape difference also constrains the remedy: simple "more data" or "more capacity" fixes — which help diffuse-error problems — would do less for Task 2 than a representation that makes those specific pairs separable in the first place.

**Connection to theoretical concepts from the subject.** The result that LBP + HSV adds large absolute lift on top of provided features but near-zero lift on top of CNN features is a clean illustration of feature redundancy and the curse of correlated features: once a strong, semantically rich representation is present, hand-crafted descriptors mostly span subspaces it already covers. The bigger lesson — and the one I would emphasise if writing this up for a non-CS audience — is that the "10 % rule" intuition for multi-class accuracy is misleading. A Task 2 result of 0.88 accuracy looks like routine multi-class success, but the per-class breakdown reveals that the model has effectively solved 6 classes and is guessing nearly at random within two species pairs. Aggregate accuracy obscures structural failure modes; macro-F1 partially exposes them; the confusion matrix exposes them fully.

## 5. Conclusion

Coarse-to-fine progression makes hand-crafted features collapse but barely scratches ImageNet-pretrained CNN embeddings: the provided features fall from 0.53 to 0.24, while CNN features hold at ~0.88 across both tasks. The best classifier does not transfer — SVM-RBF wins on the larger, well-separated Task 1; Random Forest wins on the smaller, fine-grained Task 2, because bagging plus feature-subsetting regularises better under small-sample, heterogeneous-feature conditions. Almost all of the remaining fine-grained error concentrates on the two species pairs the dataset designers warned would be difficult, suggesting that further accuracy on this task depends less on better classifiers than on representations that preserve the small, locally-anchored visual features (bill, cap) that distinguish those pairs — directions a globally-pooled CNN embedding largely discards.

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.

[2] Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.

[3] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, Caltech.

[4] Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto.

[5] Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.

[6] Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*.

[7] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
