#  MitoChem-AI: Specialized Large Language Model for Mitochondrial Biology and Chemical Inference

**MitoChem-AI** is a fine-tuned version of **Qwen2.5-7B-Instruct** specialized in molecular mechanisms, chemical-biological inference, and structured data extraction within mitochondrial biology, cellular metabolism, and skin health.

##  Key Capabilities

* **Mechanistic Reasoning:** Predicts complex, multi-organelle signaling cascades (e.g., ER-Mitochondria crosstalk).
* **Structured Output:** Highly accurate JSON extraction of biological entities.
* **Domain Expertise:** Specialized knowledge in mitochondrial dynamics ($\text{Drp}1$, $\text{Mfn}2$), Sirtuins ($\text{SIRT}3$), and metabolic pathways ($\text{MPC}$, $\text{OXPHOS}$).

##  CRITICAL: The Naming Conflict (MPC)

The base model has a strong bias for defining the abbreviation $\text{MPC}$ as the **Mitochondrial Protein Import Complex**, even though this model is fine-tuned for the **Mitochondrial Pyruvate Carrier**.

* **Status:** The current model has this known bug (as seen in Test 4).
* **Fix:** We are deploying an update ($\text{v1.1}$) with contrastive training to explicitly correct this fundamental error with abbreviations.
* **Mitigation (User Action):** You **MUST** provide strong context and NO abbreviations (see Usage Guide) to force the correct interpretation.

##  Deployment and Access

### 1. Model Weights (Hugging Face Hub)
The LoRA adapter weights and tokenizer configuration are hosted here:
`https://huggingface.co/kayceesamuel/Mitochem_AI`

### 2. Inference API (Recommended Access)
You can use the model directly via the **Hugging Face Inference $\text{API}$ endpoint** that automatically loads when you upload your model and weights.

* **API URL:** `https://huggingface.co/kayceesamuel/Mitochem_AI`
* **Quick Start:** The simplest way to use the model without coding is through the **Inference Widget** directly on the Hugging Face model page.

##  Optimal Usage Guide (How to Get the Best Results)

To ensure high-quality, domain-specific answers, follow these rules:

| Rule | Example (Good) | Example (Bad) | Benefit |
| :--- | :--- | :--- | :--- |
| **1. Always Add Context** | "In the context of **pyruvate metabolism**, what is the function of **MPC**?" | "What is $\text{MPC}$?" | Resolves the naming ambiguity instantly. |
| **2. Specify the Cell Type/Condition** | "In a **UVB-stressed epidermal keratinocyte**, how does $\text{SIRT}3$ act?" | "What does $\text{SIRT}3$ do?" | Forces the model to use the most relevant training data. |
| **3. Control Output Style** | Set $\text{temperature}=0.3$ and use the instruction: "Return the entities as a **JSON list**." | Requesting JSON with $\text{temperature}=0.8$. | Ensures structured output stability. |
| **4. Use the Instruction Format** | Frame your query with the prefix: **Instruction: [Your task here]** | Just inputting a raw question. | Aligns with the model's fine-tuning and maximizes performance. |

##  Reproducibility and Development

* **Training Script:** See `scripts/01_train_model.py` for the full training loop.
* **Testing Script:** See `scripts/02_test_model.py` for the inference helper function and sample tests.
* **Dependencies:** See `requirements.txt`.
* **Configuration:** See `configs/lora_config.yaml`.
