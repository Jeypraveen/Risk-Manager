# 🎤 Pitch Script — Risk Manager (5 Minutes)

> **Track 2: AI Risk Manager**
> **Style:** Natural, conversational English. Speak like you're explaining to a smart friend, not reading a paper.

---

## ⏱️ TIMING OVERVIEW

| Section | Duration | Cumulative |
|---|---|---|
| 1. Hook + Problem | 0:45 | 0:45 |
| 2. Why Current Solutions Fail | 0:45 | 1:30 |
| 3. My Solution (Architecture) | 1:00 | 2:30 |
| 4. Live Demo | 1:30 | 4:00 |
| 5. Honest Metrics + Close | 1:00 | 5:00 |

---

## SECTION 1 — THE HOOK + PROBLEM (0:00 – 0:45)

> 📺 **SCREEN:** Show your README page on GitHub (the repo landing page)

**Say:**

> "Hi, I'm Jeypraveen. I built **Risk Manager** — an AI system that stops return fraud for e-commerce merchants.
>
> Let me start with a number. Last year, return fraud cost online sellers **over 100 billion dollars** globally. In India alone, platforms like Flipkart and Amazon see return rates of 25 to 35 percent in categories like fashion and electronics. And roughly **8 to 10 percent** of those returns are fraudulent.
>
> What does return fraud look like? Three main types:
>
> **Empty box** — customer ships back an empty box and claims a refund.
> **Substitution** — customer orders an iPhone, returns a cheap phone case inside the iPhone box.
> **Wardrobing** — customer buys a dress, wears it to a party, returns it the next day saying 'didn't fit.'
>
> Now here's the real question — **who pays for this?** The merchant does. And that's where Razorpay comes in, because if the merchant loses money, everyone in the payment chain feels it."

---

## SECTION 2 — WHY CURRENT SOLUTIONS FAIL (0:45 – 1:30)

> 📺 **SCREEN:** Keep README visible, or switch to a simple slide/image showing the refund flow

> 🖼️ **VISUAL TIP:** If you have a whiteboard or paper, draw this quick flow:
> `Customer returns → Warehouse receives → Merchant inspects → Refund issued`
> Point at it while you speak.

**Say:**

> "Now you might ask — 'the merchant gets the product back, they can check it, so what's the problem?'
>
> Great question. But here's what's actually happening in real life.
>
> **Amazon** has a program called **Returnless Refunds**. If the item costs less than the return shipping, Amazon tells the customer — 'just keep it, here's your refund.' The merchant **never sees the product**. No inspection happens.
>
> **Flipkart** does the same for low-value items and certain categories. The refund goes out **before** the product reaches the warehouse.
>
> Even **FedEx and UPS** — they handle the logistics, not the inspection. The package goes from customer to warehouse. By the time someone opens it and finds an empty box... the refund is already processed. The money is gone.
>
> And here's the scale problem. **Amazon India** processes **millions of returns every month**. You cannot have a human being open every box, photograph every item, and compare it with the original. It's physically impossible at that scale.
>
> So the real question becomes — **how do you make a smart decision about whether to approve a refund instantly, BEFORE anyone opens the box?** That's exactly what my system does."

---

## SECTION 3 — MY SOLUTION (1:30 – 2:30)

> 📺 **SCREEN:** Show `docs/architecture.md` or the architecture section of your PDF report

> 🖼️ **VISUAL TIP:** If you have a printed diagram of the architecture (tabular → vision → meta-learner → decision), hold it up briefly.

**Say:**

> "Risk Manager uses a **Late-Fusion Meta-Learner** architecture. Let me break that down simply.
>
> **Step 1 — Tabular Scoring.** The moment a return request comes in, the system looks at behavioral signals. How old is this account? How many returns has this person filed in the last 7 days? Is this a high-value COD order? Was the return filed suspiciously close to the policy deadline? A LightGBM model scores all of this and gives a base risk score.
>
> **Step 2 — Vision Pipeline.** If the customer uploaded a return photo, the system uses **DINOv2** — a vision transformer from Meta — to compare the return photo against the original catalog photo. If someone ordered a smartphone but returns a phone case, the cosine similarity drops dramatically. The system also runs an empty-box detector using edge density and color entropy.
>
> **Step 3 — Fusion.** A meta-learner combines both signals. The key design choice is **late fusion** — if the photo is missing or the vision model crashes, the system doesn't break. It falls back to tabular-only scoring. It **degrades gracefully, never dies**.
>
> **Step 4 — Three-Way Decision.** The system never auto-rejects. It routes to one of three outcomes:
> - **Auto-Approve** — high trust, instant refund
> - **Nudge** — medium trust, asks for a photo or offers store credit
> - **Manual Review** — low trust, sends to a human
>
> This is **strictly defense-only**. The system protects the merchant without ever denying a customer outright."

---

## SECTION 4 — LIVE DEMO (2:30 – 4:00)

> ⚠️ This is your most important section. Practice this 3-4 times before recording.

### Demo Step 1: Show the Dashboard (15 seconds)

> 📺 **SCREEN:** Open `http://localhost:8000/demo/index.html` in your browser

**Say:**

> "Let me show you the live system. This is the demo dashboard running locally through Docker."

*Point to the different sections:* "Request data on the left, vision verification in the middle, results on the right."

---

### Demo Step 2: Legitimate Return (30 seconds)

> 📺 **SCREEN:** Click "Load Random Sample" or type in values for a legitimate-looking return
> - order_value: 1200, account_age: 500, prior_returns: 1, return_velocity: 0
> Upload: `data/images/catalog/smartphone.png` as catalog, `data/images/returns/smartphone_legitimate.png` as return
> Click **"Score Request"**

> 🖼️ **Hold up your phone or a printed image** of two matching phones side by side

**Say:**

> "Here's a legitimate return. Old account, low return velocity, and look — the catalog photo and the return photo are the **same smartphone**. DINOv2 gives a high similarity score.
>
> The system says **Auto-Approve**. Instant refund. Happy customer, no delay."

---

### Demo Step 3: Fraud — Substitution (30 seconds)

> 📺 **SCREEN:** Keep order_value high (8000), set account_age low (30), prior_returns high (5), return_velocity high (3)
> Upload: `data/images/catalog/smartphone.png` as catalog, upload a **mismatched** return image (e.g., `headphones_substitution.png`)
> Click **"Score Request"**

> 🖼️ **Hold up two different objects** — e.g., a phone box and a random different item — to visually show substitution

**Say:**

> "Now watch this — same smartphone was ordered, but the return photo is a completely different item. The similarity score drops to near zero. Combined with the suspicious behavioral signals — new account, high return velocity — the system routes this to **Manual Review**.
>
> Notice — it doesn't reject. It sends it to a human. Defense only."

---

### Demo Step 4: Circuit Breaker (15 seconds)

> 📺 **SCREEN:** In the dashboard, click the **"Simulate Failure"** button (timeout or checkpoint mismatch), then score again

**Say:**

> "What if the vision system crashes? The circuit breaker catches the failure, logs it to the audit trail, and the system **still gives a decision** using tabular data alone. It raises the approval threshold so it won't auto-approve without visual verification. It degrades, but it never dies."

---

## SECTION 5 — HONEST METRICS + CLOSE (4:00 – 5:00)

> 📺 **SCREEN:** Show the metrics section of your README or the PR curve from `evaluation/results/pr_curve_fusion.png`

> 🖼️ **VISUAL TIP:** If possible, show the PR curve image on screen — the two curves (blue tabular, red fusion) make a strong visual impact

**Say:**

> "Let me be honest about the numbers — because the contest asks for honest metrics.
>
> The tabular-only model gets a **PR-AUC of 0.447** on a held-out test set. That's realistic — this is synthetic data with heavy overlap between fraud and legitimate behavior. No single feature separates them easily.
>
> But when I add the vision pipeline, the fusion model jumps to **PR-AUC of 0.858**. That's the power of multi-modal fusion.
>
> The cost-optimal threshold gives **83% precision and 76% recall**. That means — for every 100 returns the system flags, 83 are actually fraud. And it catches 76 out of every 100 fraud cases.
>
> I also included a **full validity boundaries document** that explains exactly what these numbers do and do not prove. The data is synthetic, the absolute numbers would change with real merchant data, but the **architecture and methodology are production-ready**.
>
> Every single decision is logged in a full **audit trail** — what scores were computed, what thresholds were active, whether the vision pipeline failed, and why the system made that specific decision. This is essential for compliance in fintech.
>
> To summarize: Risk Manager is a defense-only, multi-modal AI system that protects merchants from return fraud. It works when images are available. It works when images are missing. It never crashes, it never auto-denies, and every decision is fully traceable.
>
> Thank you."

---

## 🎯 KEY PHRASES TO DROP NATURALLY

Use these when they fit — judges love hearing these:

| Phrase | When to say it |
|---|---|
| "Defense-only — we never auto-deny" | During architecture + demo |
| "Degrades gracefully, never dies" | During circuit breaker demo |
| "Late fusion lets any modality fail without crashing the system" | During architecture |
| "The merchant never sees the product" (returnless refunds) | During problem statement |
| "Honest metrics — this is what it can and cannot prove" | During metrics |
| "Every decision is fully auditable" | During close |
| "Amazon processes millions of returns monthly — you can't inspect every box by hand" | During problem statement |

---

## 📋 PRE-RECORDING CHECKLIST

- [ ] Docker container running (`docker-compose up`)
- [ ] Dashboard loads at `http://localhost:8000/demo/index.html`
- [ ] Have these images ready to upload during demo:
  - `data/images/catalog/smartphone.png`
  - `data/images/returns/smartphone_legitimate.png`
  - `data/images/returns/headphones_substitution.png` (or any mismatched image)
- [ ] Practice the demo flow 3-4 times to hit timing
- [ ] Keep the PDF report / GitHub README open in another tab for quick switching
- [ ] Total time should be under 5:00 — aim for 4:30 to leave breathing room

---

## 🖼️ VISUAL AIDS CHEAT SHEET

| Moment in Script | What to Show |
|---|---|
| "Three types of fraud" | 📺 README on screen |
| "Returnless refunds" flow | 🖼️ Quick sketch on paper: Customer → Refund (no inspection) |
| Architecture explanation | 📺 Architecture diagram from PDF/docs |
| Legit demo | 📺 Dashboard + 🖼️ Two matching product photos |
| Fraud demo | 📺 Dashboard + 🖼️ Two mismatched products |
| Circuit breaker | 📺 Dashboard failure simulation |
| Metrics | 📺 PR curve image (`pr_curve_fusion.png`) |
| Close | 📺 Back to GitHub README |
