# Pitch Script: Return-Risk Scorer (5 Minutes)

## 0:00 - 1:00 | The Problem (Problem Taste)
"E-commerce return fraud costs merchants billions. But the standard industry response—automatically denying suspicious returns—is a massive mistake. False positives alienate your best customers. Furthermore, building offensive AI models (like deepfake detectors) to spot fake images just trains fraudsters to become better. Our approach is entirely defense-only. We don't fight the fraudster; we structurally secure the return process so fraud becomes economically unviable."

## 1:00 - 2:00 | The Solution (Architecture & Build Quality)
"We built the Return-Risk Scorer. It uses a Late-Fusion architecture. 
First, we analyze behavioral metadata (tabular data) like account age and return velocity. 
Second, we use a Vision Pipeline powered by DINOv2 to check for substitution fraud, and heuristics to check for empty boxes.
Most importantly: we built this for the real world. If the imageCDN goes down, or the vision model crashes, our Circuit Breaker catches the exception and gracefully degrades to tabular-only scoring. The system never crashes."

## 2:00 - 3:00 | The AI Judgment (Late-Fusion & Cost Awareness)
"We don't output binary 'Yes/No' decisions. Our meta-learner fuses the modalities and outputs a probabilistic Trust Score. 
We then route that score through a cost-sensitive decision router:
- High trust gets AUTO_APPROVE.
- Medium trust gets a generic NUDGE (e.g., 'Please upload a photo').
- Low trust forces MANUAL_REVIEW.
Because a False Negative (refunding a fraudster) costs ₹2,300, but a False Positive (denying a legit customer) costs ₹1,200 in churn, our thresholds mathematically optimize for the merchant's bottom line, not just raw accuracy."

## 3:00 - 4:00 | Security & Failure Recovery (Demo)
*(Show the UI here)*
"Let's look at the demo. We upload a legitimate return. The system approves it.
Now we upload a fraudulent substitution. The vision pipeline catches the mismatch, the trust score plummets, and it routes to Manual Review.
What if a fraudster tries to spam the API to guess our thresholds? Our server-side database tracks re-photo attempts. On the third attempt, it hard-locks the return to Manual Review. No client-side bypasses."

## 4:00 - 5:00 | Conclusion & Validity
"We evaluated this on a 2,000-row synthetic test set. By fusing vision with tabular data, our PR-AUC jumped from 0.447 to 0.858. 
This is a production-grade architecture that is ready to be plugged into a real merchant's data warehouse. It's safe, defense-only, and failure-tolerant."
