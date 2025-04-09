# 🔧 DPO + Quantization: Fine-tuning and Compressing LLMs with LoRA

This project demonstrates a full pipeline of fine-tuning and compressing a large language model (LLM) using Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and model quantization (GPTQ), with Hugging Face's `transformers`, `peft`, and `trl` libraries. The final model is optimized for efficient inference and compatible with customized evaluation APIs.

本项目展示了一个完整的大语言模型优化流程，涵盖监督式微调（SFT）、偏好学习（DPO）、以及模型量化（GPTQ）。我们使用了 Hugging Face 社区的工具库如 `transformers`、`peft` 和 `trl`，最终生成的模型可以在资源受限环境中高效推理，并符合项目定制化评估接口的要求。

---

## 📌 Project Structure 项目结构

```
llm-dpo-lora-project/
├── dpo_train_with_lora.py         # DPO训练脚本
├── quantization_colab.ipynb       # GPTQ量化实验
├── saved_model/                   # 保存的模型
├── sft_test/                      # LoRA适配器权重（用于加载）
├── README.md
```

---

## 🚀 Pipeline Overview 项目流程概览

### 🔹 Milestone 1 – Supervised Fine-tuning (SFT)

- Dataset: [`Tachi67/sft_dataset`](https://huggingface.co/datasets/Tachi67/sft_dataset)
- Base model: `bigscience/bloom-1b7`
- Fine-tuned using `LoRA` adapter to reduce training cost.
- Although this part was led by a teammate, I also participated in training improvements due to their absence during key phases.

使用 SFT 数据集对 Bloom 模型进行初步微调，引入了 LoRA 以降低参数规模。该阶段虽由队友主导，但由于其阶段性不在本地，我协助进行了训练流程的调整和调优。

---

### 🔹 Milestone 2 – Direct Preference Optimization (DPO)

- Dataset: [`Tachi67/mnlp_dpo_data_7k`](https://huggingface.co/datasets/Tachi67/mnlp_dpo_data_7k)
- Framework: `trl.DPOTrainer`
- Strategy: Load SFT-trained LoRA adapter and continue optimizing using DPO loss.
- Result: Successfully adapted the SFT model to better reflect preference-style learning.

在 LoRA 适配器基础上继续微调，使用 DPO 方法替代传统 RLHF，以偏好数据优化模型对生成质量的判断能力。

---

### 🔹 Milestone 3 – Quantization (GPTQ)

- Methods tested: SmoothQuant, AWQ, AQLM, Quanto, GPTQ
- Final solution: [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)  
- Issues with other methods:
  - LoRA compatibility
  - Saving format incompatible with evaluation API
- GPTQ used mixed int8/fp16 quantization, guided with DPO dataset
- Performed on [Google Colab notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing)

最终我们选用 GPTQ 方法完成模型压缩，因其对 LoRA 支持较好且兼容 Hugging Face API，可顺利保存并用于下游推理任务。

---

## 🧠 Key Insights 项目收获

- Learned how to integrate LoRA with different fine-tuning paradigms.
- Understood the structural differences between SFT and DPO training logic.
- Gained hands-on experience debugging quantization toolkits in constrained environments.
- Developed the ability to choose between engineering trade-offs in model deployment.

---

## 🤖 Environment 运行环境

```bash
transformers==4.36+
peft==0.7+
trl==0.7+
accelerate
auto-gptq
torch==2.0+
```

---

## 👩‍💻 Author 作者

**Zimu Zhao**  
MSc in Digital Humanities, EPFL  
Focus: AI model tuning, resource-efficient training, and cross-modal interaction design.  
个人关注方向：AI 模型微调、低资源条件下的效率优化，以及人机交互设计在语言任务中的实现。

---

## 📬 Contact 联系方式

欢迎与我联系讨论 LoRA、DPO 或模型压缩等相关话题！  
Feel free to reach out if you'd like to discuss anything related to LoRA, DPO, or quantization strategies.
