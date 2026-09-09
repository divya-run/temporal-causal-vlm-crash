# Diagnosing and Repairing Temporal-Causal Reasoning Failures in Compact VLMs for Roadside Crash Understanding

**Team Members:** Divya Rajasekar,Claude

**Selected Track:** Research Track (frontier question: VLM reasoning) + Safety & Evaluation Track (targeted failure-mode analysis, edge-deployable monitoring)

## Abstract

Vision-Language Models (VLMs) are increasingly proposed as scene-understanding

components for cooperative and infrastructure-assisted autonomous driving, but

recent benchmarks show they struggle specifically with causal and temporal

reasoning about crash events, not just visual description. This project uses

the CrashSight benchmark (2026) to isolate one concrete failure mode in a

compact, edge-deployable VLM, diagnoses its root cause using a temporal-order

probe, and tests whether a lightweight intervention (phase-conditioned

prompting or targeted LoRA fine-tuning) closes the gap. We further quantize

and deploy the resulting model on an NVIDIA Jetson to measure whether the fix

survives real-time, on-device constraints — a dimension the source benchmark

does not evaluate. The project targets a submission to the 6th LLVM-AD

workshop (WACV 2027).

##  Literature & SOTA Survey Results
**CrashSight (Gan et al., Apr 2026)** — Introduces a phase-aware, infrastructure-centric crash video benchmark with 250 videos and 13K multiple-choice QA pairs across three reasoning tiers. Benchmarks 8 VLM configurations zero-shot and fine-tuned, and shows models struggle most on causal and temporal reasoning despite strong scene description. Provides an error taxonomy we build directly on.
**RiskCueBench (2026)** — Shows that shuffling or reversing video frames barely changes VLM accuracy on crash and risk-cue understanding, indicating current models rely on salient frames rather than genuine temporal order. Supplies the diagnostic method this project reuses to isolate the failure mode.
**The Abstraction Gap in Vision-Language Causal Reasoning (2026)** — Surveys how VLMs handle causal reasoning from synthetic to naturalistic scenes, and finds a persistent gap between formal causal inference and real-world visual abstraction. Motivates why crash scenes are a particularly hard causal-reasoning setting.
**2COOOL (WACV 2026, LLVM-AD)** — An evaluation benchmark for generating incident reports on out-of-distribution hazards in driving. Relevant as a companion evaluation angle if crash-specific data proves limited, and as evidence the workshop values narrow hazard/incident-focused submissions.
**Efficient Visual Question Answering Pipeline for Autonomous Driving via Scene Region Compression (WACV 2026, LLVM-AD)** — Demonstrates a compute-efficient VQA pipeline for driving scenes, representative of the workshop's stated preference for efficiency-focused rather than frontier-scale work.
**FROST-Drive (WACV 2026, LLVM-AD)** — A scalable, efficient end-to-end driving approach built on a frozen vision encoder, used here as a reference point for how much can be achieved without full-model fine-tuning.
**LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)** — The parameter-efficient fine-tuning method this project uses to adapt a compact VLM without full retraining, keeping the project within a single-team, single-GPU compute budget.
**Qwen2.5-VL Technical Report** — Describes the compact, open-weight vision-language model family this project selects as its base model for both the diagnostic probe and the fine-tuned intervention.


## Project Proposal Document
**Problem Formulation**
Current VLMs used for driving-scene understanding perform well at describing what is visible in a crash scene but perform substantially worse at explaining why a crash happened and how it unfolded over time. This gap matters for cooperative/infrastructure-assisted autonomous driving, where downstream systems may rely on a VLM's causal account of an incident. This project narrows that broad problem to one measurable question: for the specific reasoning categories where an open benchmark shows the largest model failure, can a lightweight, edge-deployable intervention meaningfully close the gap?

Input: Roadside crash video clips (CrashSight dataset) plus associated multiple-choice reasoning questions spanning scene perception, event-level reasoning (temporal sequence, crash mechanics), and inference-level judgment (fault determination, evidence sufficiency).
Output: A predicted answer to each reasoning question, plus (for the diagnostic phase) a classification of why the model failed, following the benchmark's own error taxonomy.
Target Metrics / Success Criteria:
Accuracy on CrashSight Tier 2 (causal/temporal) questions, before and after intervention.
Temporal-sensitivity score: accuracy delta between original, shuffled, and reversed frame order (RiskCueBench-style probe) — success means this delta increases after intervention, indicating genuine temporal reasoning rather than frame pattern-matching.
Post-quantization accuracy retention (4-bit/8-bit) and on-device inference latency/FPS on Jetson — success means less than a defined threshold (e.g., 5%) accuracy drop with real-time-viable latency.
Proposed Technical Approach
Baseline & diagnosis: Evaluate a compact open VLM (Qwen2.5-VL, ~3B parameters) zero-shot on CrashSight's Tier 2 questions; reproduce the benchmark's error taxonomy on our model specifically.
Root-cause probe: Apply the RiskCueBench frame-shuffle/reverse test to the same questions to quantify how much of the failure is due to lack of genuine temporal reasoning versus other causes.
Intervention: Implement one lightweight fix targeted at the diagnosed cause — either (a) explicit phase-conditioned prompting using CrashSight's four annotated crash phases (pre-crash, collision dynamics, aftermath, causes), or (b) a LoRA fine-tune using phase labels as auxiliary supervision.
Edge validation: Quantize the resulting model to 4-bit/8-bit and deploy on the team's NVIDIA Jetson, measuring accuracy retention, latency, and memory footprint.
Deliverables: Reproducible codebase, experiment logs comparing baseline vs. intervention, and a written failure/success analysis suitable for workshop submission.
Device & Maintainer
Device available: NVIDIA Jetson (team-owned hardware); cloud GPU (Colab/RunPod) for the LoRA fine-tuning step, since training will not run on the Jetson itself.
Maintainer (for potential Claude Code / ongoing repo upkeep): [Divya — or designated team member], responsible for the Jetson deployment pipeline and quantization; remaining team members own the benchmark evaluation and fine-tuning experiments respectively.


## AI Novelty & Feasibility Audit
**AI Critique Summary:**

**Novelty Score:** Medium-High. The source benchmark (CrashSight) is only months old, and no published work yet targets its specific error taxonomy with a lightweight, diagnosis-driven fix rather than a general-purpose fine-tune. Combining a temporal-sensitivity probe (from a separate, also-recent benchmark) with a targeted intervention is a genuinely underexplored combination.
**Red Ocean Risks:** Low-Medium. General "VLMs for driving scene understanding" is an active area, but this project's specific niche — targeted causal/temporal failure-mode repair plus edge-quantized validation — has essentially no direct prior work as of this writing. The main risk is that other teams could build on CrashSight quickly given its visibility (it has already been cited by at least one follow-up paper within months of release), so timely execution matters.
**Feasibility Audit:** High. The dataset is public and modestly sized (250 videos, 13K QA pairs), the base model is small enough for LoRA fine-tuning on a single cloud GPU, and the team already owns Jetson hardware for the deployment/quantization phase. The main feasibility risk is the relatively small fine-tuning split, which raises overfitting risk and should be monitored with a held-out validation split.




