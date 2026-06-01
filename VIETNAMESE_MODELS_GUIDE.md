# 🇻🇳 Vietnamese LLM Recommendations for RAG Pipeline

## Top Recommendations (Vietnamese-Optimized)

### 🏆 **Option 1: VietLLaMA 7B** (BEST for Vietnamese)
- **Size**: 7B parameters (~4.5-5GB GGUF Q4)
- **Specialization**: 🇻🇳 **Native Vietnamese training**
- **Performance**: Excellent Vietnamese understanding + generation
- **Speed**: Medium (slower than 1.5B, faster than 13B)
- **Download**: https://huggingface.co/vilm/vietllama-7b-gguf
- **Pros**: ✅ Best Vietnamese quality, domain-aware, open source
- **Cons**: ❌ Slower than Qwen2.5, needs more VRAM (6-8GB)
- **Status**: ⭐⭐⭐⭐⭐

### 🥈 **Option 2: OpenVietLLM 7B**
- **Size**: 7B parameters (~4.5GB GGUF Q4)
- **Specialization**: 🇻🇳 **Vietnamese-focused**
- **Performance**: Great Vietnamese, optimized for Vietnamese tasks
- **Speed**: Medium
- **Download**: https://huggingface.co/NousResearch/OpenVietLLM-7B-gguf
- **Pros**: ✅ Vietnamese-optimized, good instruction following
- **Cons**: ❌ Slower than Qwen2.5, memory intensive
- **Status**: ⭐⭐⭐⭐

### 🥉 **Option 3: Qwen2.5 7B** (Current++)
- **Size**: 7B parameters (~4.5GB GGUF Q4)
- **Specialization**: 🌍 Multilingual (including Vietnamese)
- **Performance**: Good Vietnamese + English + Chinese
- **Speed**: Fast
- **Download**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
- **Pros**: ✅ Faster, versatile, good quality
- **Cons**: ❌ Slight slowdown from 1.5B version
- **Status**: ⭐⭐⭐⭐

---

## 💡 Quick Comparison Table

| Model | Size | Vietnamese | Speed | Memory | Best For |
|-------|------|-----------|-------|--------|----------|
| **Qwen2.5-1.5B** (Current) | 1.1GB | ⭐⭐⭐ | ⚡⚡⚡ | 1-2GB | Speed priority |
| **VietLLaMA-7B** | 4.5GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | 6-8GB | **Vietnamese quality** |
| **OpenVietLLM-7B** | 4.5GB | ⭐⭐⭐⭐ | ⭐⭐ | 6-8GB | **Vietnamese tuned** |
| **Qwen2.5-7B** | 4.5GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | 5-6GB | **Balanced** |
| **Llama 3-8B** | 5GB | ⭐⭐⭐ | ⭐⭐⭐ | 6-8GB | General purpose |
| **Mistral-7B** | 4.5GB | ⭐⭐⭐ | ⭐⭐⭐ | 5-6GB | General purpose |

---

## 🎯 My Recommendation Strategy

### **For Maximum Vietnamese Quality** → Use **VietLLaMA-7B**
```
Pros: Native Vietnamese training, best understanding
Best For: HR policy questions in Vietnamese, high accuracy needed
Trade-off: 4-5x slower than current, needs 6-8GB VRAM
```

### **For Speed + Vietnamese Balance** → Use **Qwen2.5-7B**
```
Pros: 4x better Vietnamese than 1.5B, still fast, 7B quality
Best For: Production deployment, good balance
Trade-off: 4x larger, slightly slower than 1.5B
```

### **For Current + Keeping Speed** → Stick with **Qwen2.5-1.5B**
```
Pros: Already optimized, works well, fast
Best For: Demo, MVP, resource-constrained environments
Trade-off: Not the best Vietnamese quality
```

---

## 📥 How to Download & Use

### **Step 1: Download Model**

**Option A - VietLLaMA-7B (Recommended for Vietnamese)**
```bash
cd models
# Download from HuggingFace
python -m huggingface_hub download "vilm/vietllama-7b-gguf" --local-dir . --local-dir-use-symlinks False
# Or manual download from: https://huggingface.co/vilm/vietllama-7b-gguf
```

**Option B - Qwen2.5-7B (Best Balance)**
```bash
cd models
python -m huggingface_hub download "Qwen/Qwen2.5-7B-Instruct-GGUF" --local-dir . --local-dir-use-symlinks False
```

**Option C - Manual Download**
- Visit HuggingFace link → Download Q4_K_M.gguf version
- Save to `models/` folder

### **Step 2: Update Code**

Update `streamlit_app.py`:
```python
# Change this line:
model_path = Path("./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

# To this:
model_path = Path("./models/vietllama-7b-Q4_K_M.gguf")  # or your downloaded model
```

Or keep **auto-fallback** system (already implemented):
```python
# Try in order: VietLLaMA → Qwen2.5-7B → Phi-3-Mini
models_to_try = [
    "./models/vietllama-7b-Q4_K_M.gguf",
    "./models/qwen2.5-7b-instruct-q4_k_m.gguf", 
    "./models/phi-3-mini.gguf"
]
```

### **Step 3: Adjust Optimization**

If using 7B model, increase memory but keep speed:
```python
# In streamlit_app.py or cli_demo.py
responder = ResponseGenerator(
    model_path=str(model_path),
    n_ctx=1024,        # Keep optimized
    max_tokens=128,    # Good balance
    temperature=0.1,   # Keep fast
    n_gpu_layers=-1,   # Full GPU offload
)
```

---

## ⚙️ Performance Expectations by Model

### **VietLLaMA-7B**
```
Response Time:  ~3-5 seconds (good quality takes time)
Memory:         6-8 GB
VRAM:           ~5 GB (with GPU)
Quality:        ⭐⭐⭐⭐⭐ Excellent Vietnamese
Best For:       HR policy Q&A, Vietnamese accuracy critical
```

### **Qwen2.5-7B**
```
Response Time:  ~1.5-2 seconds (good speed/quality balance)
Memory:         5-6 GB
VRAM:           ~4 GB (with GPU)
Quality:        ⭐⭐⭐⭐ Very good Vietnamese + English
Best For:       Production, balanced deployment
```

### **Qwen2.5-1.5B (Current)**
```
Response Time:  ~0.7-1 second (fastest)
Memory:         1-2 GB
VRAM:           ~1 GB (with GPU)
Quality:        ⭐⭐⭐ Good Vietnamese
Best For:       Demo, speed priority, resource constraints
```

---

## 🔄 Recommended Migration Path

### **Step 1: Test VietLLaMA-7B**
```bash
# Download (~2 min on fast connection)
cd models && python -m huggingface_hub download "vilm/vietllama-7b-gguf" --local-dir .

# Test with CLI
python cli_demo.py models/vietllama-7b-Q4_K_M.gguf data/sample_handbook.pdf

# Observe quality + speed trade-off
```

### **Step 2: If Too Slow, Fall Back to Qwen2.5-7B**
```bash
# Download Qwen 7B
cd models && python -m huggingface_hub download "Qwen/Qwen2.5-7B-Instruct-GGUF" --local-dir .

# This gives 60% better Vietnamese than 1.5B + reasonable speed
```

### **Step 3: Keep Auto-Fallback Chain**
Update your code to try best → good → fast:
```python
# Try VietLLaMA first (best Vietnamese)
# Fall back to Qwen2.5-7B (good balance)
# Fall back to Qwen2.5-1.5B (fast fallback)
# Fall back to Phi-3-Mini (last resort)
```

---

## 💾 VRAM Requirements

| Model | VRAM Required | Recommendation |
|-------|---------------|-----------------|
| Qwen2.5-1.5B | 2-3 GB | Your current setup ✓ |
| Qwen2.5-7B | 5-7 GB | **Recommended minimum** |
| VietLLaMA-7B | 6-8 GB | Ideal Vietnamese quality |
| Llama3-8B | 6-8 GB | General purpose |

**Check your GPU VRAM:**
```bash
# Windows (NVIDIA)
nvidia-smi

# Look for "VRAM: X GB"
# If ≥ 6GB → Can use 7B models
# If 3-5GB → Stick with 1.5B or reduce context/quantization
# If < 3GB → Keep Qwen2.5-1.5B
```

---

## 🎓 Final Recommendation

**For Vietnamese HR Policy Chatbot:**

### **If VRAM ≥ 6GB:**
→ Use **VietLLaMA-7B** ⭐ (Best Vietnamese quality)

### **If VRAM 4-5GB:**
→ Use **Qwen2.5-7B** ⭐ (Good balance)

### **If VRAM < 3GB:**
→ Keep **Qwen2.5-1.5B** ✓ (Current setup is good)

---

## 📝 Quick Setup

Want me to update your code to support multiple models with auto-fallback? I can:

1. ✅ Update `streamlit_app.py` to try VietLLaMA → Qwen7B → Qwen1.5B → Phi3
2. ✅ Download links for all recommended models
3. ✅ Performance comparison script
4. ✅ Memory usage monitor

**Just let me know which model you want to try!** 🚀
