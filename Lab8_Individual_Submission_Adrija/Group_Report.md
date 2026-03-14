# Group Report
**Lab 8 – Fine-Tuning and Domain Adaptation for GenAI Systems**

**Team Members:** Adrija Ghosh
**Project Title:** GenAI-Driven Uncertainty-Aware Forecasting and Recovery Simulation for Decision Support

## 1. Domain Task Definition
The domain task is **uncertainty-aware disaster forecast explanation generation**. The model receives forecast indicators (such as probability of impact, confidence interval, rainfall intensity, surge level, exposure of vulnerable populations, and historical analog information) and generates a clear, decision-support snippet for emergency planners and humanitarian stakeholders.

## 2. Instruction Dataset Description
The instruction dataset uses an instruction-input-response format to teach the model how to convert structured forecast signals into narratives.
Examples cover scenarios like:
- Storm track uncertainty explanation
- Flood response planning
- Evacuation planning
- Resource allocation recommendations

## 3. Adaptation Method Used
Parameter-efficient fine-tuning via **LoRA (Low-Rank Adaptation)** was used. LoRA adapters were attached to self-attention target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of the base model (`TinyLlama-1.1B-Chat-v1.0`).

## 4. System Integration Description
The adapted model integrates into our project pipeline as an inference layer. Once forecast data is retrieved (e.g., via our RAG/database back-end), the data is synthesized by the model through a FastAPI endpoint and presented to the user via a Streamlit interface.

## 5. Evaluation Results
The baseline model output was compared with the LoRA-adapted model. The adapted model consistently provided better domain-specific logic, specifically addressing disaster-planning parameters natively.

## 6. Impact on Project Performance
The integration of a specialized, fine-tuned model reduces hallucinations regarding disaster uncertainty. Instead of providing generic weather information, the model provides actionable statements relevant to emergency planning.

## 7. Individual Contribution Table

| Student Name | Contribution | Percentage |
|--------------|--------------|------------|
| Adrija Ghosh | Instruction dataset creation, fine-tuning implementation, Streamlit & FastAPI integration, evaluation | 100% |
