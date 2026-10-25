# Modelos abliterated / uncensored: estado del arte (21-oct-2026)

**Método.** Datos sacados de la API de Hugging Face el 21-10-2026:
- Búsquedas `abliterated`, `heretic` y `uncensored`, ordenadas por descargas, likes y trending. Salen **752 repos únicos**.
- Fichas y `config.json` de los modelos base.
- Tamaños reales de los ficheros de cada cuantización, sumando shards.
- UGI Leaderboard (CSV de 1.329 filas, último test 16-10-2026).
- Papers y READMEs de las herramientas.

> ⚠️ El campo `downloads` de la API cuenta **los últimos 30 días**. Las cifras de refusals y KL son **autodeclaradas por quien sube el modelo** salvo donde se indica.

> 🧭 **La referencia es la §11 (v2 auditada) y `profiles.yaml`.** Las §5, §6, §9 y §10 son análisis previo y contienen afirmaciones que la auditoría corrigió (ver `data/verification/`). Ante una contradicción, manda la §11.

### Revisión del 21-10-2026: erratas corregidas

| # | Error | Corrección | Comprobado contra |
|---|---|---|---|
| 1 | "Flash-Next no cabe en 48 GB (mínimo 58–68 GB)" | Solo tiene que estar en VRAM el shard 1 (Q2_0 37,6 · IQ2_XS 39,2 · IQ3_XXS 47,0 · IQ3_S 54,8 GB). La tabla n-gram de 28,8 GB va mmap a RAM/disco. **Cabe en una L40S**, y en una RTX PRO 6000 a IQ3_S sobra sitio para KV. | Card de ISTA-DASLab y de SC117 |
| 2 | NVFP4 de Flash-Next "solo en H200" | El H200 es Hopper y **no tiene NVFP4**. dealignai lo valida en 2× DGX Spark (Blackwell). Son 135 GB: no cabe en 1× RTX PRO 6000. | Card de dealignai |
| 3 | "HauhauCS Balanced" para Qwen3.8-27B | Para Qwen3.8 **solo existe Aggressive**. Balanced existe para Gemma4-12B/26B. | Búsqueda HF |
| 4 | Coder390 con los benchmarks del Qwen3.8-27B base | Es huihui aplicado sobre un SFT/RLOO de nerkyor. ⚠️ **Re-corregido:** nerkyor **sí** publica benchmarks (FP8, mismo protocolo): LCB 90/100 frente a 83 del Qwen3.8-27B original, GPQA 178 frente a 177, MMLU 450 frente a 444, menos truncados a 94K, y ~180 tok/s con DFlash2. Lo que no está medido es el daño de la abliteration de huihui (capas 2–52). Vuelve a ser la opción principal de coding. | Card de nerkyor y de huihui |
| 5 | "Qwen3-Coder-30B: huihui pierde poco (−0,4)" | Ese −0,4 es de Qwen3-30B-A3B-**Instruct**. El Coder no está medido. | ugi.csv |
| 6 | Gemma-4-31B heretic "#3–4 del UGI" | **#6 (coder3101, 65,69) y #8 (llmfan46, 64,99) de 1.329** en el ranking global. | ugi.csv |
| 7 | "Gemma y Llama tienen AUP" | Gemma 4 es **Apache-2.0**. Tienen AUP Llama y Gemma ≤3. | Card de Google |
| 8 | Bucoid IQ4_XS 14,4 GB | 14,0–14,3 GB, con un KLD medio de 0,028–0,033 frente a BF16. | API de HF y card |
| 9 | Nemotron-Bolt "cabe en 24 GB" | 22 GB + 1,7 GB de mmproj: sin visión y con contexto corto. NVIDIA publica recetas de 28/32 GB → mejor en RTX 5090. | Card de NVIDIA y de BoldingBuilds |
| 10 | DS-V4.1 en 4× H200 "justo" | **Re-corregido:** dealignai lo valida en 4× H200 (TP4) con el motor de referencia, a 14,36 $/h. | Card de dealignai |
| 11 | Stock de B200/B300 "sin stock" | Con 1 GPU hay stock Low. Multi-GPU no está disponible. Los pods de RunPod exigen RAM mínima: 4090 ≥ 31 GB, L40S ≥ 62 GB, RTX PRO 6000 ≥ 157 GB. | API de RunPod |

**Sin verificar, y marcado como tal en el texto:**
- Las métricas de KL y refusals las publican los propios autores.
- La abliteration sobre Flash-Next (dirección de huihui) no tiene KL publicado.
- mrfakename/DeepSeek-V4.1-1017-GGUF tiene el card vacío. dealignai advierte que cuantizar sus pesos de forma ingenua degrada el resultado.
- La compatibilidad de llama.cpp con arquitecturas muy nuevas cambia rápido: usa siempre la última release.

---

## 1. Panorama en una página

**1. El modelo base dominante es Qwen3.8-27B.** Es denso, multimodal, tiene 262K de contexto y "thinking" activable.
- Acumula unos **13,8 M de descargas al mes** y ~17,6 K likes repartidos en ~90 derivados abliterated/heretic/uncensored.
- Le siguen Qwen3.8-Flash-Next (125B MoE, 6B activos), Qwen3.5-9B, Qwen3.6-35B-A3B y la familia Gemma 4.

**2. En tendencia ahora mismo:**
- Variantes de **Qwen3.8-Flash-Next**: cuantizaciones GSQ-RCO de ISTA-DASLab con abliteration encima, y el runtime Strata.
- **DeepSeek-V4.1-Flash-1017**, publicado el 17-oct y abliterado/"uncensored" en menos de 24 h.
- **GLM-5.3 / 5.3-Flash.**

**3. Técnica.** La abliteration "clásica" (una dirección de rechazo) está siendo sustituida por métodos que minimizan el daño:
- **Heretic** de p-e-w: optimización automática del trade-off refusals↔KL.
- **MPOA / norm-preserving biprojected** de grimjim.
- **Heretic-ARA**: optimización directa de pesos con L-BFGS, ya fusionada en Heretic.
- Ejemplo: `heretic-org/Qwen3.8-27B-heretic-ara` pasa de **99/100 a 0/100 refusals con KL 0,054**, prácticamente sin pérdida.

**4. Ya hay datos de calidad (UGI NatInt) y todos los métodos pueden dañar.**
- En 153 pares variante↔base del UGI, la pérdida mediana de NatInt va de +0,2 (MPOA) a −3,0 (Derestricted).
- **Dentro de cada método la dispersión es enorme**: con Heretic va de −18,8 a +5,6.
- huihui en concreto cae fuerte en la arquitectura híbrida Qwen3.5/3.6 con thinking (−11 a −13), y apenas pierde en Llama, Qwen2.5 o Qwen3-Next.
- Detalle en la §7.
- **Las matemáticas son lo más sensible**: hasta −18,8 pp en GSM8K (arXiv 2512.13655).

**5. Prácticamente todo lo de 2026 razona.**
- Qwen3.8, Gemma 4, DeepSeek V4, GLM-5.3, Kimi K3 y Nemotron 3.5 tienen thinking por defecto o configurable.
- Los no-razonadores relevantes son Qwen3-Coder-30B-A3B / Coder-Next y los modelos legacy (Llama 3.x, Qwen2.5, Gemma 3).

---

## 2. Rankings Hugging Face (descargas/30 días)

### 2.1 Popularidad agregada por familia base (abliterated + heretic + uncensored)

| Familia base | Repos | Descargas/30d | Likes | Trending (suma) |
|---|---:|---:|---:|---:|
| **Qwen3.8-27B** (+ Ternary-Bonsai-2-27B) | 90 | 13,8 M | 17.609 | 732 |
| **Qwen3.8-Flash-Next** | 42 | 4,8 M | 2.620 | 478 |
| Qwen3.5-9B / Qwythos-9B | 38 | 3,3 M | 5.367 | 141 |
| Qwen3.6-35B-A3B (+ Ornith) | 37 | 3,0 M | 6.381 | 69 |
| Gemma-4-12B | 18 | 1,3 M | 1.477 | 58 |
| Gemma-4-E4B/E2B | 13 | 1,2 M | 2.476 | 52 |
| Qwen3.6-27B | 21 | 1,0 M | 5.908 | 25 |
| **DeepSeek-V4/V4.1-Flash** | 29 | 0,84 M | 2.053 | **446** |
| Gemma-4-26B-A4B | 25 | 0,84 M | 3.008 | 35 |
| Gemma-4-31B | 18 | 0,80 M | 1.180 | 23 |
| GLM-5.3-Flash | 12 | 0,76 M | 1.124 | 68 |
| GLM-5.3 / 5.2 (754B) | 7 | 13 K | 1.426 | 50 |
| Kimi-K3 (2,8T) | 2 | 2 K | 378 | 6 |

### 2.2 Top "abliterated"

| # | Repo | Descargas | Likes | Trending |
|---|---|---:|---:|---:|
| 1 | huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF | 2.431.853 | **1.415** | 68 |
| 2 | SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF | 1.950.149 | 332 | **70** |
| 3 | 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF | 1.070.113 | 720 | 33 |
| 4 | mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF | 670.081 | 18 | – |
| 5 | Bahushruth/Qwen3.6-35B-A3B-abliterated-v4 | 473.143 | 13 | – |
| 6 | huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 343.899 | 184 | 7 |
| 7 | huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF | 341.330 | 224 | – |
| 8 | wangzhang/gemma-4-31B-it-abliterated | 225.130 | 39 | – |
| 9 | alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF | 207.818 | 164 | 39 |
| 10 | huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF | 184.228 | 57 | 4 |

- **Más likes (histórico):** Huihui-Qwen3.8-27B (1.415 + 597 la versión safetensors), DavidAU Llama-3.2-8X3B Dark-Champion (657), DavidAU gpt-oss-20b abliterated (555), audnai/penclaw-GLM-5.3 (430), Huihui-Qwen3.5-35B-A3B (401), mlabonne/gemma-3-27b (345), mlabonne/NeuralDaredevil-8B (298).
- **Trending esta semana:**
  - drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated (58)
  - DavidAU/Qwen3.8-Flash-Next-…-50x-Reasoning (48)
  - Huihui-Qwen3.8-Flash-Next-abliterated-**v2** (43)
  - Huihui-Qwen3.8-27B-**Coder390-EfficientThink**-abliterated (14)
  - BoldingBuilds/Nemotron-3.5-Bolt-74B-A9B-Abliterated-24GB (8)

### 2.3 Top "heretic" / "uncensored" (otras técnicas)

| Repo | Descargas | Likes | Trending | Técnica |
|---|---:|---:|---:|---|
| HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF | 2.359.355 | 1.968 | 122 | propietaria, conserva MTP |
| JonathanColetti/Qwen3.8-27B-Uncensored-GGUF | 2.575.500 | 1.651 | 49 | – |
| DavidAU/Qwen3.5-9B-The-Defiant-Fable-…-Heretic | 1.895.000 | 1.144 | 58 | heretic + fine-tune/merge |
| DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-…-Heretic-NEO-CODER | 1.737.441 | 1.918 | **127** | heretic + merge "coder" |
| HauhauCS/Qwen3.6-35B-A3B-Uncensored-Aggressive | 909.645 | **3.970** | 44 | propietaria |
| orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF | 869.081 | 1.000 | 112 | gated, sin detalles |
| orcarouter/Qwen3.8-27B-Uncensored-FP8 | 126.807 | 1.832 | 18 | gated |
| dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 | 2.803 | 195 | **178** | "CRACK" (propietaria) |
| dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 | 52.502 | 733 | 72 | "CRACK" |
| llmfan46/gemma-4-31B-it-uncensored-heretic(-GGUF) | 79.496 | 197+158 | 3 | Heretic |

---

## 3. Estado del arte de las técnicas

| Técnica | Cómo funciona | Calidad / evidencia |
|---|---|---|
| **Abliteration clásica** (Arditi 2024, FailSpy, mlabonne, **huihui-ai**) | Calcula la dirección de rechazo (diferencia de medias de activaciones) y la ortogonaliza en o_proj/down_proj. | Funciona, pero degrada: UGI NatInt de Qwen3.5-27B pasa de 35,8 a 23,1 y gpt-oss-120b de 33,7 a 16,7. huihui reconoce disclaimers residuales. |
| **Abliteration + DPO "healing"** (NeuralDaredevil, Llama-3.3-70B-abliterated-finetuned) | Abliterar y después reentrenar con DPO/SFT. | Recupera casi todo menos matemáticas (GSM8K). |
| **Norm-preserving biprojected / MPOA** (grimjim) | Elimina solo la componente ortogonal a la "harmfulness" y restaura la norma de cada fila. | Gemma-3-12B: NatInt 18,7 → **21,3** (sube). |
| **Heretic** (p-e-w, AGPL) | Optuna optimiza un kernel de pesos por capa y componente, minimizando refusals + KL a la vez. Soporta MoE e híbridos. | Gemma-3-12B: 3/100 refusals con **KL 0,16**, frente a 0,45 de huihui y 1,04 de mlabonne. NatInt ≈ base en Gemma-4-31B y gpt-oss-120b. |
| **Heretic-ARA** (Arbitrary-Rank Ablation, fusionado sep-2026) | No usa dirección: L-BFGS ajusta pesos para mantener las salidas inocuas y acercar las dañinas a sus vecinas inocuas. | Qwen3.8-27B-heretic-ara: **0/100 refusals, KL 0,054**. Inestable en algunos modelos. |
| **Strata** (runtime) | Proyecta la dirección en inferencia (h ← h − (h·r)r) sin tocar los pesos. Se combina con cuantizaciones GSQ-RCO. | Flash-Next IQ3_S: KL 0,075 frente a BF16 abliterated, con 93 % de top-1 idéntico. |
| **Expert-specific (MoE)** (MoTE, expert silencing) | Silenciar los expertos que concentran el rechazo. | DeepSeek-R1: −52 % refusals sin pérdida en MT-Bench, pero el CoT pasa a chino en el 10 % de los casos. |
| **Propietarias** (HauhauCS "Aggressive/Balanced", dealignai "CRACK", orcarouter) | No publican el método. | HauhauCS dice 0/465 refusals, sin KL ni benchmarks. dealignai DS-V4.1: HarmBench 100 %, **MMLU −4,2 pp** (moral_scenarios −39,9 pp). |

**Formatos que conviene conocer** (no son técnicas de uncensoring):
- **GSQ-RCO** (ISTA-DASLab): cuantización 2–3,5 bpw de altísima calidad. Flash-Next IQ3_S a 83,6 GB "≥ base en todas las tareas".
- **APEX** (mudler): GGUF consciente de MoE.
- **MTP-preserved**: conserva la cabeza de multi-token prediction. Sin ella pierdes el speculative decoding, del orden de 2–3× de velocidad.
- **NVFP4**: solo Blackwell.
- **Ternary-Bonsai-2** (prism-ml): 27B ternario de 5,9–7,2 GB que retiene el ~98 % del FP16.

**Razonamiento y rechazo.**
- En modelos que razonan, el rechazo **reaparece dentro del thinking**. DeepSeek base cumple el 42,8 % con el razonamiento apagado y solo el 1,6 % con reasoning=max.
- Conviene validar con el thinking ON y con el thinking OFF.

**Riesgos.**
- Más optimismo y sobreconfianza, y menos incertidumbre expresada (arXiv 2607.17427, "Abliteration Is Not a Scalpel").
- Pérdida de la capacidad de decir "no" ante premisas falsas, lo que trae más alucinación y sycophancy.
- Licencias heredadas del base: Llama y Gemma ≤3 tienen AUP (Gemma 4 es Apache-2.0); Qwen3.8-Flash-Next usa "qwen-community-1.0"; GLM-5.3 y Kimi K3 tienen licencias propias.

---

## 4. Modelos base (verificados en HF)

| Base | Tipo | Total / activos | Contexto | Razona | Visión | Licencia |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | Denso híbrido (DeltaNet) | 27,8B | 262K (→1M) | ✅ on por defecto, toggle + reasoning_effort | ✅ | Apache-2.0 |
| Qwen3.8-Flash-Next | MoE 512 exp. + n-gram emb. | 125B / **6B** (+51B n-gram, 4B MTP) | 262K (→1M) | ✅ toggle | ✅ | qwen-community-1.0 |
| Qwen3.6-35B-A3B | MoE | 35B / 3B | 262K | ✅ | ✅ | Apache-2.0 |
| Ornith-1.5-35B-A3B | MoE (fine-tune de Qwen3.6) | 35B / 3B | 262K | ✅ siempre `<think>` | – | MIT |
| Qwen3-Coder-30B-A3B / Coder-Next | MoE | 30B/3B · 80B/3B | 256K | ❌ **no-thinking** | – | Apache-2.0 |
| Gemma-4-31B / 26B-A4B / 12B / E4B | Denso / MoE | 31B · 25B/3,8B · 12B · 8B | 256K / 128K | ✅ configurable | ✅ (+audio en 12B/E*) | Apache-2.0 (Gemma terms) |
| Nemotron-3.5-Bolt-74B-A9B-24GB | Mamba2-Transformer LatentMoE, 2-bit nativo | 74B / 9B | 256K | ✅ toggle | ✅ | OpenMDW-1.1 |
| LFM2.5-24B-A2B | MoE conv+GQA | 24B / 2,45B | 131K | ✅ **siempre** | – | LFM1.0 |
| DeepSeek-V4-Flash-0731 | MoE | 284B / 13B | 1M | ✅ effort low/high/max | – | MIT |
| DeepSeek-V4.1-Flash-1017 | MoE + Engram | 552B (+196B Engram, 763B en disco) / ~8–16B | 1M | ✅ | ✅ | MIT |
| GLM-5.3-Flash | MoE, atención sparse + lineal | 321B / 18B | 1M | ✅ effort (max por defecto) | ✅ | MIT |
| GLM-5.3 | MoE DSA | ~744B / ~40B | 1M | ✅ | – | glm-5.3 |
| Kimi-K3 | MoE KDA + MLA | 2,8T / 104B | 1M | ✅ | ✅ | kimi-k3 |

### Benchmarks oficiales del modelo base (antes de abliterar)

| Modelo | SWE-bench Pro | LiveCodeBench v6 | GPQA-D | HLE (sin tools) | Terminal-Bench 2.1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next | **62,5** | **91,9** | **91,7** | 35,9 | – |
| Qwen3.8-27B | 61,7 | 90,3 | 89,2 | 30,8 | 73,0 |
| Ornith-1.5-35B-A3B | 59,6 (SWE-V 79) | – | 89,2 | 25,6 | 67,8 |
| Qwen3.6-35B-A3B | 49,5 (SWE-V 73,4) | 80,4 | 86,0 | 21,4 | 52,5 |
| Nemotron-3.5-Bolt 74B (IQ2) | – | – | 86,4 | 25,3 | 51,0 |
| Gemma-4-31B | 35,7 (SWE-V 52) | 80,0 | 84,3 | 19,5 | 42,1 |
| Gemma-4-26B-A4B | 13,8 | 77,1 | 82,3 | 8,7 | – |
| DeepSeek-V4-Flash-0731 | 56,0 | 90,6 | 90,8 | 33,8 | 82,7 |
| DeepSeek-V4.1-Flash-1017 | DeepSWE **77,0** | – | – | 42,6 | – |
| GLM-5.3 | DeepSWE 66,9 | – | – | 42,0 (HLE+tools 62,5) | 88,2 |
| Kimi-K3 | DeepSWE 67,5 | – | – | 43,5 | 88,3 |

---

## 5. Recomendaciones por máquina y uso

> ⚠️ **HISTÓRICO, superado por la §11 v3.** Contiene recomendaciones que la auditoría corrigió. No lo uses para decidir.


Tamaños = ficheros reales en HF. Deja un 10–25 % libre para KV-cache/contexto.

### 5.1 Local 16–24 GB (RTX 4080/4090/5080/5090, 3090)

| Uso | Recomendado | Archivo / tamaño | Razona | Notas |
|---|---|---|---|---|
| **Generalista (24 GB)** | `heretic-org/Qwen3.8-27B-heretic-ara` (cuantizado) o `HauhauCS/Qwen3.8-27B-…-Aggressive-MTP-GGUF` (para Qwen3.8 solo existe la versión Aggressive) | Q4_K_P 17,9 GB · IQ4_XS 15,7 GB | ✅ | El mejor 27B disponible. Con heretic-ara la pérdida es mínima (KL 0,054). Con MTP ganas velocidad. |
| Generalista (alternativa) | `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF` | Q4_K_M 18,7 GB | ✅ | **#3–4 del UGI de los open uncensored** (65,0), NatInt intacto. La mejor para escritura y conocimiento general. |
| **Generalista (16 GB)** | `Bucoid/Qwen3.8-27B-Heretic-Ara-16GB-VRAM-IQ4-XS-MTP-GGUF` (KLD medio 0,028–0,033 frente a BF16) o huihui UD-IQ4_XS | 14,0–14,3 GB | ✅ | Contexto corto. |
| 16 GB "milagro" | `Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF` | **PQ2_0 7,2 GB** | ✅ | 27B ternario; prism-ml declara un 98 % del FP16. Necesita el fork de llama.cpp de prism. |
| **Coding** | `huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated` | GGUF Q4 ≈ 17 GB | ✅ ("efficient think") | En tendencia. Es huihui (capas 2–52) sobre el SFT/RLOO de nerkyor. Benchmarks del Coder390 sin abliterar: LCB 90 frente a 83 del base, GPQA 178/198 y menos bucles de razonamiento. La abliteration no está medida. |
| Coding rápido (MoE) | `gbuzhf/Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF` o `huihui-ai/…Qwen3.6-35B-A3B-abliterated-MTP-GGUF` | Q4_K 21,7 GB · Q3_K 17,2 GB | ✅ | 3B activos: muy rápido, y en 16 GB funciona con expertos en CPU. |
| Coding sin razonamiento | `mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF` | Q4_K_M 18,6 GB | ❌ | Respuestas directas, ideal para autocompletado. Es más antiguo (2025). |
| **Ciencia** | Qwen3.8-27B heretic-ara (GPQA 89,2), o `BoldingBuilds/Nemotron-3.5-Bolt-74B-A9B-Abliterated-24GB-GGUF` | 22 GB (+1,7 mmproj) | ✅ | Nemotron-Bolt tiene GPQA 86,4 y HLE 25,3 pero va muy justo en 24 GB (22 GB de modelo + 1,7 GB de mmproj: sin visión y con contexto corto; NVIDIA publica recetas de 28/32 GB). Evita la abliteration cruda en mates (GSM8K es lo que más cae). |
| Ligeros (8–12 GB) | `HauhauCS/Gemma4-12B-QAT-Uncensored-Balanced` (7,4 GB), `lukey03/Qwen3.5-9B-abliterated`, `DavidAU/Qwen3.5-9B-Defiant-Fable-Heretic` | 6–8 GB | ✅ | – |
| Experimental | Qwen3.8-Flash-Next GSQ-RCO con el motor **Strata** (cache de expertos en RAM) | 12–24 GB de VRAM + ≥64 GB de RAM | ✅ | Según ISTA/SC117 funciona "en una GPU de 12 GB con 64 GB de RAM". No lo he verificado; es un runtime de terceros. |

### 5.2 NVIDIA L40S (48 GB, Ada: FP8 sí, NVFP4 no)

| Uso | Recomendado | Tamaño | Razona |
|---|---|---|---|
| **Generalista** | Qwen3.8-27B en **FP8** (`orcarouter/Qwen3.8-27B-Uncensored-FP8`, 30,9 GB, gated) o heretic-ara en Q8_0 (~29 GB) con vLLM y 128K+ de contexto | 29–31 GB | ✅ |
| Generalista (escritura/UGI) | gemma-4-31B heretic Q8_0 | 32,6 GB | ✅ |
| **Coding** | Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated en FP8/Q8, o Ornith-1.5 / Qwen3.6-35B-A3B abliterated en Q8_0 (37,8 GB) para máximo throughput | 29–38 GB | ✅ |
| **Ciencia** | Qwen3.8-27B heretic-ara Q8, o Nemotron-Bolt-24GB con contexto largo | 22–29 GB | ✅ |
| **Generalista/agentes (alternativa)** | **Qwen3.8-Flash-Next** GSQ-RCO **Q2_0** (`SC117/…-GSQ-RCO-abliterated-GGUF`): 37,6 GB residentes; la tabla n-gram de 28,8 GB va mmap a RAM/disco (`-lm mmap --lazy-mode on`) | 37,6 GB + 28,8 GB de RAM | ✅ |
| No cabe | GLM-5.3-Flash (≥93 GB) | – | – |

### 5.3 RTX PRO 6000 Blackwell (96 GB) / H200 (141 GB)

| Uso | RTX PRO 6000 96 GB | H200 141 GB | Razona |
|---|---|---|---|
| **Generalista** | **Qwen3.8-Flash-Next abliterated**: `huihui-ai/…-Flash-Next-abliterated-v2-GGUF` IQ3_S 83,7 GB, o `SC117/…GSQ-RCO-abliterated` IQ3_S 83,9 GB | Flash-Next IQ4_XS 93,7 GB, (⚠️ corregido: el NVFP4 de dealignai **no** funciona en H200, que es Hopper; está validado en 2× DGX Spark Blackwell) o `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` (135 GB, muy justo) | ✅ |
| Generalista (máx. calidad 27B) | Qwen3.8-27B heretic-ara **BF16** (55,6 GB) con 1M de contexto | igual | ✅ |
| **Coding** | `alesha-pro/…Flash-Next-…-Coder-Strata` / ISTA GSQ-RCO-Coder (DeepSWE 58,7, SWE-ML 81) · `huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF` Q2 (86,7 GB) | DeepSeek-V4-Flash abliterated Q4 (156 GB) con un poco de offload, o GLM-5.3-Flash abliterated IQ2_XXS–Q2_K_XL (102–109 GB) | ✅ |
| **Ciencia** | Flash-Next (GPQA 91,7 / HLE 35,9) | GLM-5.3-Flash Q2_K_XL (HLE ≈ GLM-5.2+) o Flash-Next NVFP4 | ✅ |

### 5.4 Multi-GPU obligatorio (frontera open-weights)

| Modelo uncensored | Formato / tamaño | Hardware mínimo razonable | Uso estrella | Razona |
|---|---|---|---|---|
| `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` / `mrfakename/…-1017-abliterated` | FP8/FP4 nativo, **510 GB** | 4× H200 · 8× H100 · 4× B200 · 6× RTX PRO 6000 | **Coding/agentes** (DeepSWE 77,0, mejor open). Generalista. | ✅ |
| `huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF` / `orcarouter/GLM-5.3-Flash-Uncensored-FP8` | Q4_K_XL 200 GB · FP8 328 GB | 2–4× RTX PRO 6000 · 3× H200 | Generalista multimodal. | ✅ |
| `dealignai/GLM-5.3-ABLITERATED-NVFP4` / `Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw` | NVFP4 465 GB · EXL3 293 GB | 4× B200 / 6× RTX PRO 6000 · EXL3: 4× RTX PRO 6000 | **Coding y ciberseguridad** (mejor open en CyberGym 84,5, Terminal-Bench 3.0). | ✅ |
| `Uniboshi/Kimi-K3-Abliterated-V1` | MXFP4 **1,56 TB** | 8× B300 · 16× H200 (2 nodos) | Ciencia y agentes de largo horizonte (HLE 43,5). | ✅ |
| `audnai/penclaw-GLM-5.3-abliterated` | BF16 3,8 TB (sí, 3,8 TB) | Clúster | Solo investigación. | ✅ |

---

## 6. Veredicto rápido

> ⚠️ **HISTÓRICO, superado por la §11 v3.** Contiene recomendaciones que la auditoría corrigió. No lo uses para decidir.


| | Generalista | Coding | Ciencia |
|---|---|---|---|
| **16–24 GB** | Qwen3.8-27B **heretic-ara** / HauhauCS Aggressive (Q4) · Gemma-4-31B heretic | Qwen3.8-27B Coder390-EfficientThink abliterated · Ornith-1.5-35B-A3B abliterated (MoE rápido) · Qwen3-Coder-30B abliterated (sin razonar) | Qwen3.8-27B heretic-ara · Nemotron-3.5-Bolt abliterated (24 GB) |
| **L40S 48 GB** | Qwen3.8-27B FP8/Q8 | Qwen3.8-27B Coder FP8 / Qwen3.6-35B-A3B Q8 | Qwen3.8-27B Q8 |
| **RTX PRO 6000 / H200** | Qwen3.8-Flash-Next abliterated (GGUF IQ3_S/IQ4_XS; el NVFP4 de 135 GB necesita 2× Blackwell) | Flash-Next Coder GSQ-RCO · DeepSeek-V4-Flash abliterated · GLM-5.3-Flash (H200) | Flash-Next |
| **Multi-GPU** | DeepSeek-V4.1-Flash-1017 uncensored · GLM-5.3-Flash | DeepSeek-V4.1-Flash-1017 · GLM-5.3 abliterated | Kimi-K3 abliterated · DeepSeek-V4.1 |

**Regla de oro (corregida en la §7).**
- **Elige por la evidencia de cada repo, no por el autor ni por la técnica.** La evidencia es KL publicado, refusals y, si existe, el par en UGI.
- Como orientación general: MPOA/norm-preserving y Heretic con KL < 0,1 suelen ser los más seguros.
- huihui es el primero en publicar y cubre todo, pero en la familia Qwen3.5+ con thinking ha perdido mucho.
- Los modelos de DavidAU son fine-tunes/merges con otro estilo: más creativos y menos predecibles.
- Comprueba siempre que la variante conserva el MTP si vas a usar speculative decoding.

**Protocolo de validación propio, antes de producción:**
1. Refusals con un juez LLM (thinking ON y OFF).
2. KL y top-1 frente al base en prompts inocuos.
3. MMLU, más GSM8K/AIME con un presupuesto de tokens amplio.
4. Prueba de calibración/alucinación (SimpleQA/TruthfulQA).
5. Guardrail externo si hay usuarios finales.

---

## 7. ¿Es solo huihui? Pérdida de calidad por método (datos UGI)

Hay 153 pares de modelo uncensored frente a su base en el UGI Leaderboard, comparados en el mismo modo de prompt (thinking/no-thinking). La métrica es NatInt, inteligencia natural. Script y datos: `data/ugi/method_deltas.py` y `data/ugi/method_deltas_2026-10-21.csv`.

| Método | Pares | ΔNatInt mediana | Peor caso | Mejor caso | % pares con pérdida > 3 pts |
|---|---:|---:|---:|---:|---:|
| norm-preserving / MPOA (grimjim, YanLabs, vprilepskii) | 15 | **+0,24** | −2,0 | +4,1 | **0 %** |
| otras abliteration (mlabonne, failspy, grimjim v1…) | 15 | −0,56 | −10,6 | +2,9 | 27 % |
| fine-tune "uncensored" | 3 | −1,12 | **−24,7** | −0,5 | 33 % |
| **Heretic** (incluye ARA y v2) | 86 | −1,24 | **−18,8** | +5,6 | 36 % |
| **huihui** (abliteration clásica) | 22 | −2,46 | −12,8 | +3,8 | 36 % |
| ArliAI Derestricted | 12 | −3,00 | −14,5 | +3,3 | 50 % |

**Conclusiones:**

1. **Todos los métodos están afectados.** La pérdida depende sobre todo de la ejecución concreta y del modelo base.
   - Con Heretic hay casos de −18,8 (Mistral-Small-3.2 "ultra") y de +5,6 (GLM-4.7-Flash).
   - El mismo autor puede pasar de −11,0 (llmfan46 Qwen3.5-27B heretic-v2) a −1,0 (heretic-v3).
2. **huihui no es malo en general.**
   - En Llama-3.3-70B gana +2,2, en Qwen2.5-14B +0,7, en Qwen3-Next-80B pierde solo −1,0 y en GLM-4.7-Flash con thinking gana +3,8.
   - **Su problema está en la familia Qwen3.5/3.6 híbrida con thinking:** Qwen3.5-27B −12,8, 9B −11,2 y 35B-A3B −10,8, mientras que los Heretic del mismo modelo pierden entre −0,3 y −2.
   - **Actualización (auditoría):** Qwen3.8 hereda esa arquitectura, pero en Qwen3.8-27B **el método de huihui está medido de forma independiente y sale bien** (abliterlitics: 98 % de cumplimiento en peticiones dañinas sin copyright y sin pérdida de capacidad). La cautela que extrapolé desde Qwen3.5 **no se confirma en Qwen3.8**.
   - (Según abliterlitics, la edición de huihui sobre el Qwen3.8-27B base abarca las capas 17–51 y está medida. La de su Coder390 abarca las 2–52 y no está medida.)
3. **Lo más consistente es MPOA / norm-preserving**: ningún par pierde más de 2 puntos. A cambio, a veces queda menos "dispuesto" (W/10 medio 6,4 frente a 8–9).
4. **Hay un trade-off disposición↔inteligencia.** Las variantes con W/10 = 10 son las que más se arriesgan a perder NatInt.
5. **En la práctica:**
   - Prefiere repos que publiquen KL < 0,1 (Heretic-ARA 0,054 en Qwen3.8-27B, llmfan46 MPOA 0,024).
   - Mira si hay un par en UGI.
   - Haz tu propia validación (§6).
   - huihui sigue siendo válido cuando no hay alternativa, por ejemplo en DeepSeek-V4, GLM-5.3-Flash o Flash-Next v2, que solo tienen huihui o métodos propietarios.

---

## 8. Precios RunPod por GPU (API GraphQL, 21-10-2026)

Precio por GPU y hora en USD. "Community" es más barato pero sin SLA; "Secure" es datacenter certificado. El coste 24/7 está calculado con 730 h/mes. Datos: `data/runpod/`.

| GPU | VRAM | Community $/h | Secure $/h | 24/7 mes (community) | Stock 1 GPU | Formatos |
|---|---:|---:|---:|---:|---|---|
| RTX A5000 | 24 | 0,16 | 0,27 | 117 $ | Low | GGUF/AWQ (sin FP8) |
| RTX 3090 | 24 | 0,22 | 0,50 | 161 $ | – | GGUF/AWQ |
| **RTX 4090** | 24 | **0,34** | 0,89 | 248 $ | **High** | GGUF, FP8 |
| RTX 5090 | 32 | 0,69 | 1,19 | 504 $ | Low | GGUF, FP8, **NVFP4** |
| RTX A6000 | 48 | **0,33** | 0,59 | 241 $ | Low | GGUF/AWQ (Ampere, sin FP8) |
| A40 | 48 | – | 0,59 | 431 $ (secure) | High | GGUF/AWQ |
| RTX 6000 Ada | 48 | 0,74 | 0,99 | 540 $ | Low | FP8 |
| **L40S** | 48 | 0,79 | 1,09 | 577 $ | Medium | FP8 |
| RTX PRO 5000 | 48 | 0,82 | 0,96 | 599 $ | Low | FP8, NVFP4 |
| A100 SXM 80GB | 80 | 1,39 | 1,79 | 1.015 $ | Low | GGUF/AWQ/INT8 |
| H100 SXM | 80 | 2,69 | 3,99 | 1.964 $ | High | FP8 |
| H100 NVL | 94 | 2,59 | 3,19 | 1.891 $ | Low | FP8 |
| **RTX PRO 6000** | 96 | **1,69** | 2,49 | 1.234 $ | **High** | FP8, NVFP4 |
| **H200 SXM** | 141 | 3,59 | 5,29 | 2.621 $ | Low | FP8 |
| H200 NVL | 143 | – | 3,79 | 2.767 $ (secure) | Low | FP8 |
| MI300X | 192 | – | 2,39 | 1.745 $ (secure) | – | ROCm (soporte de los MoE nuevos sin verificar) |
| B200 | 180 | 5,98 | 7,99 | 4.365 $ | Low (1 GPU); multi-GPU sin stock | FP8, NVFP4 |
| B300 | 288 | 6,94 | 8,99 | 5.066 $ | Low (1 GPU); multi-GPU sin stock | FP8, NVFP4 |

**Multi-GPU: precio mínimo por pod y disponibilidad en este momento (`lowestPrice` de la API):**

| Configuración | VRAM total | $/h | Disponibilidad |
|---|---:|---:|---|
| 2× RTX PRO 6000 | 192 GB | 3,38 | High |
| **4× RTX PRO 6000** | 384 GB | **6,76** | **High** |
| 4× H100 SXM | 320 GB | 10,76 | Medium |
| 4× H200 SXM | 564 GB | 14,36 | Low |
| **8× H100 SXM** | 640 GB | **21,52** | Low (la única de 8 GPUs disponible ahora) |
| 8× B300 (precio de lista) | 2,3 TB | 55,5 community / 71,9 secure | sin stock ahora |

> La API de RunPod no ofrece ahora mismo pods de 8× H200, 8× RTX PRO 6000 ni B200/B300. Para Kimi-K3 harían falta Instant Clusters o esperar a que haya stock.

---

## 9. Recomendación final con coste (RunPod)

> ⚠️ **HISTÓRICO, superado por la §11 v3.** Contiene recomendaciones que la auditoría corrigió. No lo uses para decidir.


| Máquina | Uso | Modelo recomendado (variante) | Por qué esta variante | Razona | GPU RunPod más rentable | $/h |
|---|---|---|---|---|---|---:|
| **16–24 GB** | Generalista | `heretic-org/Qwen3.8-27B-heretic-ara` → GGUF Q4 (~16–18 GB) | KL 0,054 publicado, 0/100 refusals | ✅ | RTX 4090 | 0,34 |
| | Generalista / escritura | `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF` Q4_K_M 18,7 GB | Par UGI: NatInt 34,4 → 34,7 (sin pérdida), UGI 65,0 (#8 de 1.329; coder3101 es #6 con 65,7) | ✅ | RTX 4090 | 0,34 |
| | Coding | `huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated` Q4 | Coder390 sin abliterar: LCB 90 frente a 83 del base. La abliteration de huihui no está medida: valídala frente a nerkyor original y llmfan46 MPOA | ✅ | RTX 4090 | 0,34 |
| | Coding (sin razonar) | `mradermacher/Huihui-Qwen3-Coder-30B-A3B-…-abliterated-i1-GGUF` Q4_K_M 18,6 GB | Indicio indirecto: huihui en Qwen3-30B-A3B-**Instruct** pierde −0,4 en UGI (el Coder no está medido) | ❌ | RTX 4090 | 0,34 |
| | Ciencia | Qwen3.8-27B heretic-ara Q4/Q5 · alternativa `Nemotron-3.5-Bolt-74B-A9B-Abliterated-24GB` (22 GB) | GPQA 89,2 / 86,4 en el base | ✅ | RTX 4090 (Nemotron justo) · RTX 5090 si hace falta | 0,34–0,69 |
| **L40S 48 GB** | Generalista | Qwen3.8-27B heretic-ara en **FP8** (vLLM, ~30 GB, 128K de contexto) | Calidad casi BF16 con FP8 nativo en Ada | ✅ | L40S · más barato: RTX A6000 con GGUF Q8 | 0,79 · 0,33 |
| | Coding | Coder390 FP8, o Qwen3.6-35B-A3B **heretic** (llmfan46, par UGI +0,6/+3,0) en Q8 37,8 GB | MoE con 3B activos, mucho throughput | ✅ | L40S | 0,79 |
| | Ciencia | Qwen3.8-27B heretic-ara Q8/FP8 | – | ✅ | L40S | 0,79 |
| **RTX PRO 6000 / H200** | Generalista | Qwen3.8-Flash-Next abliterated (huihui v2 o SC117 GSQ-RCO) IQ3_S 84 GB · (el NVFP4 de dealignai, 135 GB, **no** sirve en H200 ni cabe en 1× RTX PRO 6000) | Base GPQA 91,7, SWE-Pro 62,5. Solo existen huihui/propietarias: valida | ✅ | **RTX PRO 6000** (H200 si NVFP4) | **1,69** (3,59) |
| | Coding | Flash-Next GSQ-RCO-Coder (+Strata, KL 0,075) · DeepSeek-V4-Flash-0731 abliterated Q2 87 GB | Strata no toca los pesos y se puede desactivar | ✅ | RTX PRO 6000 | 1,69 |
| | Ciencia | Flash-Next IQ4_XS 93,7 GB, o Qwen3.8-27B heretic-ara BF16 con 1M de contexto | – | ✅ | RTX PRO 6000 | 1,69 |
| **Multi-GPU** | Generalista | GLM-5.3-Flash abliterated: huihui Q4_K_XL 200 GB, o orcarouter FP8 328 GB | 18B activos, multimodal, MIT | ✅ | 2× / 4× RTX PRO 6000 | 3,38 / 6,76 |
| | Coding | **DeepSeek-V4.1-Flash-1017 uncensored** (dealignai FP8 510 GB, MMLU −4,2 pp publicado) | Mejor DeepSWE open (77,0) | ✅ | 8× H100 SXM (4× H200: 510 GB de 564, sin margen) | 21,52 (14,36) |
| | Coding / ciberseguridad | GLM-5.3 uncensored **EXL3 3.0bpw** (293 GB) | Mejor CyberGym open (84,5) | ✅ | 4× RTX PRO 6000 | 6,76 |
| | Ciencia | Kimi-K3-Abliterated (1,56 TB) | HLE 43,5. Sin stock ahora; DS-V4.1 (HLE 42,6) es la alternativa real | ✅ | 8× B300 | ~55,5 |

**Mejores relaciones calidad/precio:**
1. **RTX 4090 a 0,34 $/h** con Qwen3.8-27B heretic-ara o Gemma-4-31B heretic.
2. **RTX PRO 6000 a 1,69 $/h** con Qwen3.8-Flash-Next: 96 GB por menos de la mitad que un H200.
3. **4× RTX PRO 6000 a 6,76 $/h** con GLM-5.3 EXL3 o GLM-5.3-Flash FP8: la frontera open por menos de 7 $/h.

- La L40S solo compensa frente a una RTX A6000 (0,33 $/h) si necesitas FP8/vLLM a alto throughput. Para GGUF, la A6000 sale a menos de la mitad.
- Los precios y el stock son una foto del 21-10-2026. Para actualizarlos, vuelve a ejecutar la consulta GraphQL `gpuTypes` (ver `data/runpod/`).

---

## 10. Formatos de pesos: qué es cada uno y cuál usar

### 10.1 Guía rápida de formatos

| Formato | Qué es | bits/peso típicos | GPU / hardware | Runtime | Cuándo usarlo |
|---|---|---|---|---|---|
| **safetensors BF16/FP16** | Pesos originales sin cuantizar | 16 | Cualquiera con VRAM suficiente | Transformers, vLLM, SGLang | Máxima calidad, fine-tuning, o como base para cuantizar tú mismo. Ocupa unos 2 GB por cada 1B de parámetros. |
| **FP8** (E4M3, "FP8-dynamic") | Float de 8 bits por tensor o bloque | 8 | **Ada (L40S, 4090, RTX 6000 Ada), Hopper (H100/H200), Blackwell**. No hay FP8 nativo en Ampere (A100/A6000/3090) | vLLM, SGLang, TensorRT-LLM | Producción en servidor: calidad ≈ BF16 a la mitad de memoria. Es el formato nativo de DeepSeek-V4 y GLM-5.3. |
| **NVFP4** | FP4 de NVIDIA con escalas FP8 cada 16 valores | ~4,5 | **Solo Blackwell** (RTX 5090, RTX PRO 5000/6000, B200, B300) | vLLM, SGLang, TensorRT-LLM, NInfer | Máximo throughput en Blackwell. Su calidad suele quedar entre la de AWQ y Q5. "Experts-Only" = solo los expertos MoE en FP4. |
| **MXFP4** | FP4 estándar OCP con escala cada 32 | ~4,25 | Blackwell nativo. Hopper y otras mediante kernels emulados | vLLM, SGLang, llama.cpp | Formato nativo de Kimi-K3 y de los expertos de gpt-oss y DeepSeek-V4.1. |
| **AWQ** (W4A16) | INT4 de solo pesos, calibrado por activaciones | 4 (g128) | Cualquier NVIDIA desde Ampere; también AMD | vLLM, SGLang, TGI | Servir 4 bits en GPUs sin NVFP4 (A6000, A100, L40S, H100). |
| **GPTQ / AutoRound** (W4A16, Int4/Int8) | INT4/INT8 de solo pesos, cuantización por segundo orden | 4–8 | Ampere y superiores | vLLM (Marlin), SGLang, ExLlama | Equivalente a AWQ. AutoRound suele dar algo más de calidad. |
| **W8A8 INT8 / SmoothQuant** | Pesos y activaciones en INT8 | 8 | Ampere y superiores | vLLM | Alternativa a FP8 en Ampere (A100/A6000). |
| **GGUF** (Q2_K…Q8_0, IQ1–IQ4, UD, APEX, GSQ-RCO, PTQ1_0/PQ2_0) | Formato de llama.cpp con muchos niveles de cuantización | 1,6–8,5 | **Todo**: NVIDIA, AMD, Apple, CPU, con offload GPU+RAM | llama.cpp, LM Studio, Ollama, KoboldCpp | **Local y máquina única**, MoE con expertos descargados a RAM, y modelos que no caben enteros en VRAM. Q4_K_M/IQ4_XS es el punto dulce; IQ2/IQ3 GSQ-RCO para gigantes. |
| **EXL3** (EXL2 = versión anterior) | Cuantización de ExLlamaV3 (QTIP/trellis) con bpw fraccionario | 2–6 | NVIDIA desde Ampere | ExLlamaV3, TabbyAPI | **La mejor calidad por GB en GPU NVIDIA** a 2,5–3,5 bpw. Ideal para meter un modelo grande en pocas GPUs, por ejemplo GLM-5.3 en 4× RTX PRO 6000. Throughput multiusuario inferior a vLLM. |
| **MLX** (4/6/8 bit, oQ*, MTPLX) | Formato de Apple MLX | 2–8 | **Solo Apple Silicon** | mlx-lm, LM Studio Mac | Solo si usas Mac. No sirve en RunPod. |
| **bnb** (bitsandbytes 4/8-bit) | Cuantización al vuelo | 4/8 | NVIDIA | Transformers | Solo para experimentar o hacer fine-tuning QLoRA. Es lento para servir. |

**Reglas para elegir:**
- **Local y 1 GPU de consumo:** GGUF. Q4_K_M o IQ4_XS; Q5/Q6 si sobra memoria. Si cuentas con MTP para el speculative decoding, verifica que el GGUF diga "MTP".
- **L40S, H100 o H200 con vLLM:** FP8. Si no cabe, AWQ o GPTQ-Int4.
- **Blackwell (RTX PRO 6000, 5090, B200):** NVFP4 da el máximo rendimiento. FP8 si buscas la máxima calidad.
- **Ampere (A6000, A100, 3090):** GGUF, AWQ/GPTQ o W8A8. No hay FP8 ni NVFP4 nativos.
- **Modelos gigantes en pocas GPUs:** EXL3 (2,5–3,5 bpw) o GGUF IQ2/IQ3 GSQ-RCO.

### 10.2 Formatos disponibles para cada modelo recomendado (HF, 21-10-2026)

Entre paréntesis, las descargas de los últimos 30 días. Datos: `data/formats/format_search_2026-10-21.json`.

| Modelo (variante) | BF16 safetensors | FP8 | NVFP4 | AWQ / GPTQ / INT | GGUF | EXL3 | MLX |
|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B heretic-ara** | `heretic-org/Qwen3.8-27B-heretic-ara` (55,6 GB) | – (cuantízalo tú con `llm-compressor`) | `lyf/Qwen3.8-27B-Heretic-ARA-NVFP4-MTP-VL` (20,9K) · `megashuckle/…-Heretic-ARA-NVFP4-NInfer` | – | `Bucoid/Qwen3.8-27B-Heretic-Ara-16GB-VRAM-IQ4-XS-MTP-GGUF` (35,6K) | – | – |
| Qwen3.8-27B heretic MPOA (llmfan46, KL 0,024) | `llmfan46/Qwen3.8-27B-Uncensored-Heretic-Native-MTP-Preserved` | – | `…-NVFP4` (3,8K) · NVFP4-GGUF | `…-GPTQ-Int4` (5,1K) | `llmfan46/…-Native-MTP-Preserved-GGUF` (23,5K) | – | – |
| Qwen3.8-27B HauhauCS | – | – | – | `twolven/…-HauhauCS-Aggressive-AWQ-MTP` (19K) | `HauhauCS/…-Aggressive-MTP-GGUF` (**2,36M**; IQ2_M 10,3 → Q8_K_P 31,5 GB) | – | – |
| Qwen3.8-27B orcarouter (gated) | `orcarouter/Qwen3.8-27B-Uncensored` | `…-Uncensored-FP8` (127K, 30,9 GB) | `…-Uncensored-NVFP4` (110K) | `…-Uncensored-INT8` | `bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF` (133K) | `dr-housemd/…-exl3-3.07bpw` | `orcarouter/…-MLX` |
| Qwen3.8-27B huihui | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | `munekazu/…-FP8` | `Barding-Defense/…-NVFP4-NInfer` (46K) · `sakamakismile/…-NVFP4` | `shawnw3i/…-AWQ-MTP` · `TelperionAI/…-INT4-AWQ-GPTQ` · `letechlead/…-W4A16-AutoRound` | `huihui-ai/…-abliterated-GGUF` (**2,43M**; PTQ1_0 6,6 → Q8_0 29 GB) | – | varios |
| Qwen3.8-27B Coder390 (huihui) | `huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated` | – | – | `alpha2beta/…-GPTQ-Int4-…-MTP` · `…-MXFP4` | `huihui-ai/…-GGUF` · `mradermacher/…-i1-GGUF` | – | oQ8e |
| **Gemma-4-31B heretic** | `llmfan46/gemma-4-31B-it-uncensored-heretic` · `coder3101/gemma-4-31B-it-heretic` | `kakrotto/…-heretic-FP8` · `vwdubb/…-FP8` | `AEON-7/…-DECKARD-HERETIC-NVFP4` (variante DavidAU) | `alonsoko/gemma-4-31b-it-abliterated-heretic-AWQ-W4A16` (4,6K) · `valoomba/…-W8A8-INT8` | `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF` (79K; Q4_K_M 18,7 GB) | – | `mlx-community/…-heretic-4bit` |
| **Qwen3.6-35B-A3B heretic** (llmfan46) | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic(-Native-MTP-Preserved)` | `protoLabsAI/…-heretic-FP8` | `llmfan46/…-NVFP4-Experts-Only` | `llmfan46/…-GPTQ-Int4` (22K) · `sahilchachra/…-AWQ` | `SC117/…-heretic-Native-MTP-Preserved-APEX-GGUF` (65K) · `llmfan46/…-GGUF` | – | `froggeric/…-MLX-4/6/8bit` |
| Ornith-1.5-35B-A3B abliterated | `huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated` | – | `pottokao/…-abliterated-NVFP4-DFlash` | `nuoram/…-W8A8` | `gbuzhf/…-Abliterated-MTP-UD-APEX-GGUF` (42K) | – | `PocketAiHub/…-MLX-4bit` |
| Qwen3-Coder-30B-A3B (huihui) | `huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated` | – | – | – | `mradermacher/…-i1-GGUF` (41K; Q4_K_M 18,6 GB) | `ArtusDev/…-EXL3` | – |
| Nemotron-3.5-Bolt-74B abliterated | – | – | – | – | `BoldingBuilds/…-Abliterated-24GB-GGUF` (**solo GGUF**, 2-bit nativo, 22 GB) | – | – |
| Ternary-Bonsai-2-27B abliterated | – | – | – | – | `Hikari07jp/…` PQ2_0 7,2 GB · `BoldingBuilds/…-PTQ1_0` · `…-PQ2_0-MTP` (requiere el fork de llama.cpp de prism) | – | `KridgeDookie/…-MLX` |
| **Qwen3.8-Flash-Next abliterated/uncensored** | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` · `orcarouter/…-Uncensored` | `dealignai/…-ABLITERATED-FP8` · `orcarouter/…-Uncensored-FP8` (75K) | `dealignai/…-ABLITERATED-NVFP4` (135 GB) · `mazinb/…-Uncensored-NVFP4` (21K) · `edp1096/…-NVFP4-QAD` | `leoncca/…-AWQ-g32` · `Mochgolf/…-W4A16-AutoRound` | `SC117/…-GSQ-RCO-abliterated-GGUF` (**1,95M**; IQ2_XS 68,5 → IQ3_S 83,9 GB) · `huihui-ai/…-v2-GGUF` · `orcarouter/…-Uncensored-GGUF` (869K) · Strata (`alesha-pro`, `jimmy31chen`) | `ghost-actual/…-EXL3-2.50bpw` · `alesha-pro/…-exl3-3bit/4bit` | varios |
| **DeepSeek-V4-Flash-0731 abliterated** | – (el nativo es FP8+FP4) | `lovesenko/DeepSeek-V4-Flash-0731-Abliterated` (14,6K, nativo) · `apetersson/…-FP8` | `sakamakismile/…-Abliterated-NVFP4` · `cbert33/…-abliterated-vision` | `Jon-Nielsen/…-MXFP4-INT8` | `huihui-ai/…-0731-abliterated-GGUF` (341K; Q2 86,7 · Q4 156 GB) | – | `apetersson/…-MLX-Mixed-2bit-3bit` |
| **DeepSeek-V4.1-Flash-1017 uncensored** | `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated` (nativo FP8/FP4, 510 GB) | `dealignai/…-1017-UNCENSORED-FP8` (510 GB) · `pqhaz/…-1017-heretic` | `pqhaz/…-1017-heretic-NVFP4` · `Infatoshi/…-NVFP4-uncensored` | `Infatoshi/…-1017-MXFP4-uncensored` | `mrfakename/…-1017-abliterated-GGUF` · `AtomicChat/…-Uncensored-GGUF` | `dealignai/…-EXL3-2.9bpw-UNCENSORED` · `Infatoshi/…-EXL3-2.875bpw/3.5bpw` | – |
| **GLM-5.3-Flash abliterated/uncensored** | `zai-org` (base) | `orcarouter/GLM-5.3-Flash-Uncensored-FP8` (163K, 328 GB) · `dealignai/…-UNCENSORED-FP8` (37K) | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` (7,6K) | `hell0ks/GLM-5.3-Flash-Uncensored-AWQ` (9,8K) | `BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF` (**491K**) · `huihui-ai/…-abliterated-GGUF` (IQ1_S 93 → Q6_K_XL 292 GB) · `GCSA-AiLab/…-RCO-GSQ-GGUF` | `Mia-AiLab/…-EXL3-4bpw-TensorFold-Ablit` | `grant-ai/…-MLX-4bit` |
| **GLM-5.3 (754B) abliterated/uncensored** | `audnai/penclaw-GLM-5.3-abliterated` (BF16 3,8 TB) | `dealignai/GLM-5.3-UNCENSORED-FP8` | `dealignai/GLM-5.3-ABLITERATED-NVFP4` (465 GB) | – | – | `Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw` (293 GB) | – |
| **Kimi-K3 abliterated** | `Uniboshi/Kimi-K3-Abliterated-V1` (nativo **MXFP4**, 1,56 TB) · `SHSLab/Kimi-K3-Abliterated` | – | – | – | `Blackfrost-AI/KIMI-K3-Q2_K-GGUF-ABLITERATED` · `Uniboshi/…-V1-GGUF` | – | – |

### 10.3 Formato recomendado en cada caso de la §9

| Máquina | GPU RunPod | Formato recomendado | Motivo |
|---|---|---|---|
| 16–24 GB local | RTX 4090 / 3090 / 5080 | **GGUF Q4_K_M / IQ4_XS** (con MTP si está disponible) · Bonsai PQ2_0 en 16 GB | llama.cpp/LM Studio; deja margen para el contexto |
| 16–24 GB, RTX 5090 (32 GB) | RTX 5090 | GGUF Q6_K, o **NVFP4** con vLLM | Blackwell: NVFP4 nativo |
| L40S 48 GB | L40S / RTX 6000 Ada | **FP8** con vLLM (27B) · GGUF Q8_0 para Qwen3.6-35B-A3B | Ada tiene FP8 nativo y no tiene NVFP4 |
| A6000 48 GB (alternativa barata) | RTX A6000 | **GGUF Q8_0** o AWQ/GPTQ-Int4 | Ampere: sin FP8 |
| RTX PRO 6000 96 GB | RTX PRO 6000 | **GGUF IQ3_S/IQ4_XS GSQ-RCO** (Flash-Next) · **NVFP4** (27B/35B con vLLM) · EXL3 3–4 bpw (Flash-Next) | Blackwell. Flash-Next IQ3_S: 54,8 GB residentes + n-gram en RAM, deja ~35 GB para KV |
| H200 141 GB | H200 SXM | **FP8** (27B, 35B) · NVFP4 no (Hopper) · Flash-Next: GGUF IQ3_S/IQ4_XS o AWQ-g32 | Hopper: FP8 nativo, sin NVFP4 |
| 2–4× RTX PRO 6000 | 2×/4× RTX PRO 6000 | GLM-5.3-Flash **FP8** (328 GB en 4×) o **NVFP4** · GLM-5.3 **EXL3 3.0bpw** (293 GB) | NVFP4/EXL3 maximizan lo que cabe |
| 4× H200 / 8× H100 | H200 / H100 | DeepSeek-V4.1-Flash-1017 **FP8 nativo** (510 GB) con vLLM/SGLang (EP+DP) | Es su formato oficial; Hopper no tiene NVFP4 |
| 8× B300 | B300 | Kimi-K3 **MXFP4 nativo** | Formato oficial; Blackwell lo ejecuta nativamente |

> ⚠️ NVFP4 **solo es nativo en Blackwell**. En Hopper y Ampere, vLLM puede ejecutarlo vía Marlin como W4A16 de solo pesos, más lento; en L40S/A6000/H100/H200 es preferible FP8 o AWQ. Los GGUF "MTP" requieren un llama.cpp reciente para el speculative decoding. Los formatos NInfer, DFlash y Strata necesitan su runtime específico. MLX es solo para Mac.

## 11. Perfiles modelo/pod: v3 (dos rondas de auditoría; resultado final)

Hice una auditoría adversarial con cuatro auditores, uno por nivel, y verifiqué en las fuentes los hallazgos clave. Los detalles están en `data/verification/*/AUDIT.md`; el fichero `profiles.yaml` tiene los comandos, el sampling y las alternativas.

Origen de cada dato: **[IND]** = medición independiente del mismo protocolo · **[AUT]** = lo publica el autor · **[NV]** = sin medir.

| Perfil | Modelo | Fichero / formato | Runtime | Evidencia | Pod · $/h |
|---|---|---|---|---|---|
| **basic-general** | orcarouter/Qwen3.8-27B-Uncensored | bartowski GGUF Q4_K_M **17,77 GB** (+ mmproj 0,93) | llama.cpp, 64K | [IND] 1º de 12: cumple el 82 %, sin pérdida medible, KL 0,068 | RTX 4090 · 0,34 |
| **basic-coding** | **TIE-VALIDATE** · por defecto: orcarouter 27B | orcarouter Q4_K_M 17,77 GB · nerkyor Coder390 Q3LynnStyle-MTP 17,30 GB · huihui Coder390 i1 Q4_K_M ~17 GB | llama.cpp + MTP, 48K | orcarouter [IND] en los 2 ejes. nerkyor: capacidad [AUT], refusals [NV]. Método huihui en el base [IND]: delivered 58,8 %, bucles 40 % | RTX 4090 · 0,34 |
| **basic-science** | orcarouter/Qwen3.8-27B-Uncensored | Q4_K_M 17,77 GB · Q6_K 23,46 GB (5090) | llama.cpp, KV q8_0, effort xhigh | [IND] igual que en general | 4090 · 0,34 / 5090 · 0,69 |
| **medium-general** | orcarouter/Qwen3.8-27B-Uncensored | FP8 **30,87 GB** (espejo sin gating `id-2/…-FP8`) · Q8_0 29,12 GB | vLLM FP8, 128K | [IND] | L40S · 0,79 |
| **medium-coding** | **TIE-VALIDATE** · por defecto: orcarouter 27B | orcarouter FP8 30,87 GB · nerkyor FP8 31,26 GB + DFlash2 · huihui Coder390 (FP8 hecho por ti o Q8_0 29,05 GB) | vLLM FP8 / SGLang (nerkyor) / llama.cpp | Igual que basic-coding | L40S · 0,79 |
| **medium-science** | orcarouter/Qwen3.8-27B-Uncensored | FP8 30,87 GB | vLLM, effort xhigh | [IND] | L40S · 0,79 |
| **big-general** | orcarouter Flash-Next (build de SC117) | SC117 IQ3_S: **51,3 GiB en VRAM** (+ n-gram 28,8 GB vía mmap) + MTP Q8_0 4,14 GB (de spiritfather) | llama.cpp master `-ngl 99 -lm mmap --spec-draft-model … --spec-draft-n-max 4` | [IND CaliperBench, thinking on] willingness_v2 0,8 (heretic-2: 0,0) · [IND AtomicChat] KLD 0,022, MMLU 86,65 | RTX PRO 6000 · 1,69 |
| **big-coding** | ídem | ídem, 192K | ídem, effort medium | Base DeepSWE 58,7 · SWE-Pro 62,5 · SWE-bench Multilingual 81,0 | RTX PRO 6000 · 1,69 |
| **big-science** | ídem | ídem | effort xhigh | Base GPQA 91,7 · HLE 35,9 | RTX PRO 6000 · 1,69 |
| **huge-general** | dealignai/GLM-5.3-Flash-UNCENSORED-**FP8** | **328,3 GB** FP8 | vLLM TP4; max_tokens ≥ 6000 | [AUT] MMLU absoluto 87,33 frente a 85,28 de la NVFP4 (mismo harness). Alternativa barata: NVFP4 194,7 GB en 4× RTX PRO 6000 (6,76) | 4× H200 · 14,36 |
| **huge-coding** | base DeepSeek-V4.1-Flash-1017 · variante **TIE-VALIDATE**: dealignai FP8 frente a pqhaz heretic | 510 GB · pqhaz también como parche de 1,6 GB | SGLang dsv4.1 TP4+EP4, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` | dealignai [AUT]: MMLU −3,86 pp. pqhaz [AUT]: KL 0,136, refusals 1/100, sin capacidad | 4× H200 · 14,36 |
| **huge-science** | ídem (TIE-VALIDATE) | ídem | effort max | Base HLE 42,6 | 4× H200 · 14,36 |

**Ronda 2 (v3): falsos positivos eliminados.** Detalle en `data/verification/round2/`.
- **basic/medium-coding:** pasan a **TIE-VALIDATE**. huihui ofrece mucho cumplimiento, pero en la medición independiente el 40 % de sus respuestas acaba en bucle (delivered 58,8 %). En nerkyor, los refusals están sin medir. orcarouter queda por defecto hasta hacer el A/B.
- **basic-general:** la abliteration de orcarouter es en la capa 38, no la 24. La alternativa de deriva baja pasa a ser RVN (15/400 refusals). heretic-ara queda como opción conservadora.
- **big-\*:** pasa de heretic-2 a la ablación de orcarouter (build de SC117). Con thinking activado, heretic-2 tiene willingness_v2 0,0, igual que el stock [IND CaliperBench]. Además, varias builds GGUF traen `compress_ratios` a cero (atención densa); SC117 lo trae correcto.
- **huge-general:** pasa a la FP8 (+2 puntos de MMLU absoluto en el mismo harness). La NVFP4 queda como opción barata.
- **huge-coding/science:** se mantiene el base DeepSeek-V4.1-1017. La variante es TIE-VALIDATE entre dealignai (−3,86 de MMLU) y pqhaz (KL 0,136, sin datos de capacidad).
- En el tier huge no hay ninguna evaluación independiente.

**Qué ha cambiado en la v2 respecto a la v1, y por qué:**
1. **basic y medium (general/science) pasan a usar orcarouter.**
   - En la comparativa independiente de abliterlitics (12 variantes, mismo pipeline), orcarouter es el 1º: cumple el 82 % sin perder capacidad.
   - heretic-ara es el más limpio, pero todavía rechaza ~1/3 de las peticiones.
   - El llmfan46 MPOA que yo había elegido tiene un **KL real de 0,30** (12× lo que declara su card), pierde en TruthfulQA y entra en bucles de razonamiento en el 37 % de los casos.
2. **Coding: Coder390, en la versión de huihui** (corregido después de la v2). Al principio puse nerkyor por delante porque su calidad está medida. Pero el método de huihui sobre Qwen3.8-27B está medido de forma independiente (98 % de cumplimiento en peticiones dañinas, sin pérdida de capacidad), mientras que los rechazos del Coder390 de nerkyor no lo están. nerkyor queda como alternativa con calidad medida.
2b. **Notas sobre Coder390 (originalmente: "pasa a nerkyor"):**
   - Viene de un padre "EfficientThink-Uncensored" con 4/140 refusals.
   - Publica benchmarks en todos sus formatos.
   - La edición de huihui (capas 2–52) es más amplia que la medida (17–51) [NV].
   - La subida de 83 a 90 en LCB es sobre todo por truncar menos: el valor oficial del base ya es 90,3.
3. **big pasa a heretic-2.**
   - La abliteration de SC117 viene de orcarouter, no de huihui, pero sus tensores son casi iguales a los de la v1 de huihui (coseno 0,9).
   - Una medición independiente da a huihui un desvío ~6× mayor que heretic-2.
   - El comando de la v1 fallaba: `--draft-max` ya no existe en llama.cpp.
4. **huge-coding y huge-science pasan a 4× H200 (14,36 $/h, validado por dealignai) en vez de 8× H100.** El repo no tiene gating. El MMLU es el de la versión 1017 (−3,86), no el de la anterior.
5. **huge-general:** el NVFP4 también funciona en Hopper vía Marlin, contra lo que dije. El "0 % refusals" solo vale con effort off/max.
6. **Errores de precio:** Secure no cuesta "+40–50 %" más, sino entre +30 % y +162 %. La RAM mínima de los pods cambia con cada consulta.

**Lo que sigue sin estar medido y conviene validar tú:**
- Los refusals del Coder390.
- heretic-2 en IQ4_XS: el KL está medido en BF16.
- La velocidad de DFlash2 en una L40S.
- Que el espejo `id-2` sea idéntico al original de orcarouter (puedes comparar hashes con el repo gated).
- El rendimiento de DeepSeek-V4.1-1017 uncensored más allá de MMLU.

---

## Fuentes principales

- **Comparativa independiente** de 12 abliterations de Qwen3.8-27B: https://abliterlitics.dev/models/qwen38-27b/ (copia en `data/verification/basic/abliterlitics/`)
- Medición de satellitedown (Flash-Next: heretic-2 frente a huihui): `data/verification/big/cards/`

- HF API: `https://huggingface.co/api/models?search=abliterated|heretic|uncensored&sort=downloads|likes|trendingScore`
- Fichas: Qwen/Qwen3.8-27B, Qwen/Qwen3.8-Flash-Next, deepseek-ai/DeepSeek-V4-Flash-0731, deepseek-ai/DeepSeek-V4.1-Flash-1017, zai-org/GLM-5.3(-Flash), moonshotai/Kimi-K3, google/gemma-4-*, nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF, ornith-ai/Ornith-1.5-35B-A3B, prism-ml/Ternary-Bonsai-2-27B-gguf
- Heretic: https://github.com/p-e-w/heretic (ARA: PR #211 → #446)
- grimjim, norm-preserving biprojected: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
- mlabonne, abliteration: https://huggingface.co/blog/mlabonne/abliteration
- UGI Leaderboard: https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard
- arXiv 2512.13655 (comparativa de herramientas), 2607.17427 ("Abliteration Is Not a Scalpel"), 2502.11096 (MoTE)
- heretic-org/Qwen3.8-27B-heretic-ara · dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 · alesha-pro/…-Strata-GGUF · ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- Parámetros de DeepSeek V4.1 Flash: https://pasqualepillitteri.it/en/news/15529/deepseek-v41-flash-748-billion-native-vision · https://www.marktechpost.com/2026/10/17/deepseek-just-released-v4-1-flash-1017-scores-77-0-on-deepswe-opens-search-api/
- GLM-5.3 (39–40B activos): https://recipes.vllm.ai/zai-org/GLM-5.3 · https://www.datalearner.com/en/benchmark-compare/glm-4-7/glm-5-3
