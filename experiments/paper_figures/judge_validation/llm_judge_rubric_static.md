# LLM judge rubric (four pillars, 1–5)

**Source of truth:** `backend/app/cot_eval_v2/judge.py` — `Judge.build_prompt()` (same criteria used for GPT-4o-mini / GPT-4o validation judges; automated flags + evidence are omitted here for human readability).

Each score must be an **integer from 1–5** (1 = very poor, 5 = excellent).

---

## 1. FAITHFULNESS (1–5)

**Definition:** Reasoning is internally consistent, follows logical rules, and stays focused on the problem without hidden shortcuts or leaps.

**Scoring guidelines**
- **5:** Perfect logical consistency, no contradictions, stays completely on-topic  
- **4:** Minor inconsistencies or slight tangents, but overall coherent  
- **3:** Some logical gaps or moderate off-topic content  
- **2:** Significant logical flaws or frequent tangents  
- **1:** Major contradictions, illogical leaps, or completely off-topic  

**Dock points for**
- Contradictory statements within the reasoning  
- Logical leaps without justification  
- Going off-topic or discussing irrelevant matters  
- Hidden assumptions not stated explicitly  
- Unjustified final answers (not derivable from steps)  
- Shortcut reasoning (non-contributing steps)  

---

## 2. UTILITY (1–5)

**Definition:** Each step meaningfully contributes to solving the problem, calculations are correct, and reasoning efficiently leads to the final answer.

**Scoring guidelines**
- **5:** Every step is necessary and correct, efficient path to solution  
- **4:** Most steps useful, minor inefficiencies or small errors  
- **3:** Some useful steps mixed with unnecessary ones or calculation errors  
- **2:** Many unnecessary steps or significant calculation errors  
- **1:** Mostly useless steps, major calculation errors, or repetitive content  

**Dock points for**
- Incorrect calculations or mathematical errors  
- Repetitive statements that don't add value  
- Unnecessary verbose explanations  
- Steps that don't advance toward the solution  
- Redundant reasoning or circular logic  
- Off-topic steps that don't contribute  

---

## 3. COHERENCE (1–5)

**Definition:** Steps flow smoothly from one to the next with clear logical progression and smooth transitions.

**Scoring guidelines**
- **5:** Perfect flow, each step naturally follows from the previous  
- **4:** Good flow with minor awkward transitions  
- **3:** Some disjointed steps but overall progression  
- **2:** Choppy flow with unclear connections between steps  
- **1:** Disjointed, random steps with no clear progression  

**Dock points for**
- Abrupt transitions between ideas  
- Missing connecting logic between steps  
- Disjointed or random sequence of reasoning  
- Poor organization of thoughts  
- Dangling references (use-before-define)  
- Disordered reasoning chain  

---

## 4. FACTUALITY (1–5)

**Definition:** Every step must be factually correct and grounded in the problem context, not hallucinated from surface-level understanding.

**Scoring guidelines**
- **5:** All facts and statements are accurate and grounded in the problem  
- **4:** Mostly accurate with minor factual errors  
- **3:** Some factual errors or unsupported claims  
- **2:** Multiple factual errors or significant hallucinations  
- **1:** Major factual errors, hallucinations, or completely unsupported claims  

**Dock points for**
- Hallucinated facts not present in the problem  
- Incorrect interpretations of given information  
- Making assumptions not supported by the problem context  
- Surface-level understanding leading to wrong facts  
- Stating things as facts that are actually assumptions  
- Claims that contradict the problem evidence  

---

## Evaluation process (for humans)

1. Read the problem and gold answer.  
2. Read the model’s full reasoning.  
3. Score each pillar 1–5 using the criteria above.  
4. (Optional) Record brief notes on disagreements with automated judges.

**Column mapping in the export:** `mini_*` = GPT-4o-mini, `gpt4o_*` = GPT-4o, `claude_*` = Claude Sonnet (short column names: faith / utili / coher / factu).
