# temporal-causal-vlm-crash
This project examines why compact VLMs fail at causal and temporal crash reasoning in CrashSight (2026). We use a temporal-order probe to diagnose the failure, test phase-conditioned prompting or LoRA fine-tuning, and quantize/deploy the model on NVIDIA Jetson to evaluate real-time edge performance. Target: LLVM-AD Workshop, WACV 2027.
