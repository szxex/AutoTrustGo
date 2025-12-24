## 🇺🇸 English README

# AutoTrustGO

**Trust-Driven Optimizer for SDXL LoRA / LyCORIS**

AutoTrustGO is a custom optimizer designed to
**maximize style fidelity and fine detail preservation**,
while **preventing overfitting, instability, and “AI-like” artifacts**
commonly seen in LoRA training.

---

## Key Features

* ✅ **Step-independent design**

  * No step count, warmup, or decay schedules
* ✅ **Confidence-driven updates**

  * Gradient trust is evaluated by rank & stability
* ✅ **Natural overfitting suppression**

  * Confidence saturation decays automatically
* ✅ **Preserves fine details**

  * Eyes, thin lines, subtle jitter remain intact
* ✅ **Universal**

  * Style, character, concept, photorealistic training
* ✅ **High generalization when applied as LoRA**
* ✅ **Avoids flattening seen in AdamW / Lion**

---

## Core Concept

AutoTrustGO continuously asks:

> *“How much can this gradient be trusted?”*

Instead of relying on step-based heuristics.

### Core Internal Signals

* **confidence**

  * Rank component: structural relevance
  * Stability component: safe variability
* **danger**

  * Continuous instability indicator
* **mask_ema**

  * Stability memory over time
* **trust_level**

  * User-defined aggressiveness

👉 All controls are **continuous, not if-based heuristics**.

---

## Recommended Use Cases

* SDXL LoRA / LyCORIS training
* Illustration style learning
* Character learning
* Concept learning (pose, structure)
* Photorealistic LoRA
* Fine-tuning–like scenarios

---

## Comparison (Summary)

| Aspect             | AdamW  | Lion   | AutoTrustGO |
| ------------------ | ------ | ------ | ----------- |
| Stability          | Medium | Medium | High        |
| Fine detail        | Medium | High   | High        |
| Overfit resistance | Medium | Medium | High        |
| Generalization     | Medium | Good   | Excellent   |
| AI artifacts       | Likely | Less   | Minimal     |
| Tuning difficulty  | Low    | Medium | Low         |

---

## Parameters

### learning_rate

* Base LR
* Internally rescaled by AutoTrustGO
  → slightly higher than AdamW is optimal

### trust_level (Main behavior control)

| Value | Behavior                        |
| ----- | ------------------------------- |
| 0.3   | Very conservative               |
| 0.4   | **Recommended (balanced)**      |
| 0.5   | Aggressive (character emphasis) |

---

## Notes

* ❌ Do NOT use step-based schedulers
* ❌ Do NOT drastically lower learning rate
* ❌ Do NOT assume Adam-style behavior

---

This optimizer differs from existing optimizers like Adam or Lion in that it gradually progresses toward the features of the target image. (The concept is different.)
It does not make rapid progress early in training, so while the final quality may be high, the learning effect during training is difficult to discern.
Please note that unlike other optimizers, even with a short step size (regardless of whether it's good or bad), it does not necessarily produce any noticeable change.
With too few steps, training may end before capturing key features, so a slightly higher number of steps tends to provide more stable results.
(Overfitting suppression is included.)

This optimizer was created using AI. However, we do not rely solely on AI; we continuously review the underlying philosophy, direction, and modification details.
Therefore, while the copyright status is somewhat ambiguous, you are generally free to modify or create derivatives. We welcome improvements.
(As long as you don't outright plagiarize it or claim it as your own original idea after just borrowing the concept.)
※This text alone was written in Japanese by a human.

---
