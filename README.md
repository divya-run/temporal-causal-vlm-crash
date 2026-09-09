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


