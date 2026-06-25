# NexaBank Business Case

## Use Case
Card A: Multilingual email triage

## 1. Value Driver
- Current state: 8,000 customer emails arrive daily in 12 languages, and agents spend about 7 minutes manually categorising each email before replying.
- Business value: automate classification, language detection, and routing so agents can respond faster and focus on higher-value customer issues.
- Expected impact: reduce triage time by about 68%, improve productivity, and lower support handling cost.
- Estimated benefit: roughly GBP 320k per year in cost reduction, with a quality lift of about 22%.

## 2. Data Requirements
- Historical customer emails with labels such as intent, urgency, language, and final routing outcome.
- Multilingual samples covering the main languages used by customers.
- Agent resolution history to learn which routes and responses worked best.
- Privacy controls to remove or mask personal data before model use.
- Synthetic data can help expand rare intent classes, but real labelled examples are still needed for validation and evaluation.

## 3. Model Selection
- Recommended model: Claude 3.5 Sonnet.
- Why: strong text understanding, good multilingual performance, and suitable quality for email classification and drafting.
- Alternate model: GPT-4o if the team prefers a different vendor ecosystem or wants a strong general-purpose multimodal model.
- Best fit architecture: rule engine for deterministic routing plus an LLM for intent detection, summarisation, and response drafting.

## 4. Risk Register
- Hallucination risk: the model may generate an incorrect category or reply. Mitigation: constrain outputs to approved labels and use human review for low-confidence cases.
- Bias risk: some languages or customer groups may be underrepresented. Mitigation: balance training data and test performance by language and region.
- Privacy risk: emails may contain sensitive customer data. Mitigation: mask PII and keep processing within approved security boundaries.
- Vendor lock-in risk: dependency on one model provider. Mitigation: keep prompt and routing logic model-agnostic where possible.

## 5. Success Metric
- 90-day pilot KPI: reduce average email resolution time by at least 30% while maintaining or improving first-pass routing accuracy.

## Recommendation
Start with a pilot on one or two high-volume email categories, then expand to more languages and intents once accuracy and agent satisfaction are proven.
