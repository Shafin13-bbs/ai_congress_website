# Skin model notes for DiagNova

**Repository:** `RevelaCap/clinical-skin-condition-v1`  
**Model card:** https://huggingface.co/RevelaCap/clinical-skin-condition-v1  
**Architecture:** EfficientNet-B0 (PyTorch / torchvision)  
**Task:** five-class educational classification from clinical/macroscopic skin photos  
**Input:** RGB image resized to 224 × 224  
**Classes documented by the model card:**
1. Eczema / dermatitis
2. Urticaria / allergic reaction
3. Folliculitis / acne-like
4. Psoriasis / papulosquamous
5. Lesion — dermoscopic review recommended (routing label, not a malignancy diagnosis)

The model card reports test macro-F1 0.6527 and balanced accuracy 0.6700 on a held-out set of 1,515 images from public research sources. It reports lower macro-F1 (0.43) on the smartphone-photo SCIN subset, uneven skin-tone representation, and no pediatric evaluation.

**Critical caveat:** The model card says the model is not validated for clinical use and is not suitable for telemedicine, clinical decision support, patient triage, diagnosis, or treatment decisions without independent validation. DiagNova exposes it only as an optional educational inference demo. Do not present its scores as calibrated clinical probabilities or claim it improves health outcomes.
