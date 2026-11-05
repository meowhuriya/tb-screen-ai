# TB-Screen AI: Official Hackathon Submission Summary
**Track:** HI-01 — Medical Image Diagnostic Assistant | **Target Word Count:** <= 150 Words (Actual: 140 Words)

---

Tuberculosis kills 930 people daily in India, largely due to 3-week radiologist delays in rural Primary Health Centers. TB-Screen AI (Track HI-01) is an edge-optimized diagnostic assistant delivering sub-10ms CPU inference with zero cloud dependency.

Trained with lung region-of-interest cropping and CLAHE contrast enhancement, our calibrated EfficientNet-B0 achieves a 0.9780 ROC-AUC (95% CI: 0.964–0.990) and 95.10% sensitivity, fulfilling the WHO screening benchmark. In the spirit of scientific integrity, external validation on the NIH Montgomery cohort is transparently reported, explaining physical domain shift from analog film digitizers.

TB-Screen AI features deterministic Out-of-Distribution rejection, Grad-CAM pulmonary explainability with interactive opacity blending, high-volume batch triage sorted by highest risk first, trilingual support (English, Hindi, Telugu), and 1-click clinical referral PDF generation. Reviewed by Medical Advisor Dr. Bepar Mohammad Tayyib, MBBS, TB-Screen AI compresses rural diagnostic turnaround from 21 days to under 2 hours.
