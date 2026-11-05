# TB-Screen AI: 2-Minute Demonstration Video Script
**Track:** HI-01 — Medical Image Diagnostic Assistant | **Target Length:** 120 Seconds (2:00)

---

### [0:00 - 0:15] SECTION 1: THE PROBLEM (15 Seconds)
**[Visual]:** Opening shot of a rural Primary Health Center (PHC). Split-screen showing an analog X-ray machine with a stack of unread film envelopes, next to a map of rural India.
**[Spoken Audio]:**
> "Tuberculosis kills over 900 people every day in India. While TB is curable with standard regimens, rural clinics face a 3-week radiologist reporting delay. During those 21 days, an untreated patient continues spreading airborne bacteria in their household. Frontline clinics desperately need an assistive tool that helps find TB cases sooner by prioritising patients for confirmatory tests, running right on a standard laptop without internet."

---

### [0:15 - 1:15] SECTION 2: LIVE DEMO WALKTHROUGH (60 Seconds)
**[Visual]:** Screen recording of TB-Screen AI Streamlit interface. Clean, uncluttered UI.
**[Spoken Audio]:**
> "Meet TB-Screen AI. It runs 100% locally in RAM on a standard dual-core laptop. Let's test with a pre-loaded sample radiograph."

**[Visual]:** Clicks `Sample 1: Confirmed Normal`.
**[Spoken Audio]:**
> "In just 7 milliseconds on CPU, the model evaluates the radiograph. It classifies the patient as: 'Low risk. No TB pattern detected by the model.' The clinical action plan advises routine primary care."

**[Visual]:** Clicks `Sample 2: Active Advanced Pulmonary TB`.
**[Spoken Audio]:**
> "Now, an active high-risk TB case. Instantly, the interface flags: 'High risk. TB suspected; confirm with a sputum test (GeneXpert or smear).' Beside it, Grad-CAM highlights the pulmonary attention patterns. Using the opacity slider, the clinician can seamlessly blend between raw lung tissue and AI saliency."

**[Visual]:** Drags the opacity slider back and forth from 0.2 to 0.8. Then attempts to upload a color photograph of a person/flower.
**[Spoken Audio]:**
> "For clinical safety, our deterministic Out-of-Distribution filter immediately rejects non-radiographs or severe blur, preventing misleading guesses."

**[Visual]:** Switches to `📂 High-Volume Batch Triage` tab. Drops 10 X-rays. Progress bar runs in under 1 second.
**[Spoken Audio]:**
> "In mobile screening camps, our Batch Mode triages dozens of X-rays at once, automatically sorting patients by highest risk first so the sickest patients receive immediate sputum testing."

**[Visual]:** Clicks `📥 Download Official Referral PDF`. PDF opens showing patient ID, side-by-side images, triage status, and doctor sign-off.
**[Spoken Audio]:**
> "With one click, a standardized single-page referral report is generated, complete with a medical officer sign-off box and full translations in Hindi and Telugu."

---

### [1:15 - 1:40] SECTION 3: ACCURACY & HONEST EXTERNAL VALIDATION (25 Seconds)
**[Visual]:** Transitions to model evaluation graphics: ROC curve, calibration reliability plot, and Montgomery County drop chart.
**[Spoken Audio]:**
> "Under the hood, we trained EfficientNet-B0 with CLAHE pulmonary contrast enhancement and lung ROI cropping. On our untouched test set of 627 radiographs, the model achieves a **0.9780 ROC-AUC** and **95.1% sensitivity** with 1,000 bootstrap confidence intervals.
>
> But we went further. We externally validated the model on the US NIH Montgomery County dataset—and we report the performance honestly: AUC dropped to 0.5873. Why? Because analog flatbed digitizers capture completely different film-grain physics compared to modern digital sensors. True medical AI requires honest reporting, not cherry-picked numbers."

---

### [1:40 - 2:00] SECTION 4: REAL-WORLD IMPACT, SAFETY & ADVISORY (20 Seconds)
**[Visual]:** Closes on the "Public Health Impact" tab and the "Medical Advisor" credit.
**[Spoken Audio]:**
> "Our clinical workflow and triage referral protocols were reviewed by Medical Advisor Dr. Bepar Mohammad Tayyib, MBBS.
>
> TB-Screen AI does not replace doctors—it is an assistive screening aid that cuts diagnostic turnaround from 21 days to under 2 hours, preserving ₹2,000 GeneXpert cartridges for the patients who need them most. Thank you."
