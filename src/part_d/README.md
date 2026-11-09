# Part D — Policy RAG Assistant and Refund Agent

## Overview

Part D implements policy-document chunking, TF-IDF retrieval, grounded policy answers, refund-related tools and a ReAct agent. It also evaluates retrieval/fallback behaviour and demonstrates safe stopping and checkpoint recovery. All model interactions and refund actions are simulated locally; no external API key or real payment transaction is used.

## Source files

| File | Responsibility |
|---|---|
| `document_chunking.py` | Loads policy documents and creates overlapping chunks |
| `retrieval.py` | Retrieves policy chunks using the configured retrieval implementation |
| `rag_evaluation.py` | Evaluates document retrieval, fact grounding and unsupported-question fallback, including a no-overlap comparison |
| `refund_tools.py` | Defines order lookup, policy search, refund calculation, refund issuance and human escalation tools |
| `react_agent.py` | Runs the ReAct agent, scenario runner and safe-stopping demonstrations |
| `checkpoint_resume.py` | Saves agent state and resumes an interrupted run |
| `agent_comparison.py` | Produces the agent comparison report required by the project workflow |

## Policy RAG workflow

Policy text files are loaded from `data/policies/` and split into chunks. The retrieval component uses TF-IDF vectorization and cosine similarity to rank relevant chunks. The RAG assistant supplies retrieved policy context to the simulated model and instructs it to answer only from that context. Unsupported questions use the configured fallback response.

The evaluation uses 14 questions: 12 answerable and 2 unsupported. Recorded results:

| Metric | With overlap | Without overlap |
|---|---:|---:|
| Document accuracy | 100% | 100% |
| Fact accuracy | 100% | 83.33% |
| Fallback accuracy | 100% | 100% |

The evaluation checks document-source matching, expected-fact substring presence and the required fallback prefix. These are deterministic benchmark checks; they do not independently prove semantic correctness for arbitrary questions. The evaluation set contains only 14 questions.

## Refund tools and policy rules

Exactly five tools are registered:

1. `get_order` — retrieves order status, order amount and delivery duration.
2. `search_policy` — retrieves relevant refund policy context.
3. `calculate_refund` — calculates the refund amount under the policy.
4. `issue_refund` — records an eligible automatic refund in the simulated ledger.
5. `escalate_to_human` — records a refund requiring human approval.

The implemented rules are:
- Only completed orders qualify for refunds.
- Delivery exceeding 45 minutes qualifies for a full refund of the order amount.
- Delivery in 45 minutes or less receives no late-delivery refund.
- Refunds up to and including INR 1,000 may be issued automatically.
- Refunds above INR 1,000 require human approval/escalation.

Refund actions are simulated and appended to `outputs/refund_ledger.jsonl`; no money is transferred.

## ReAct scenarios

The scenario runner processes seven cases from `data/raw/agent_scenarios.csv`.

| Scenario | Expected outcome | Verified outcome |
|---|---|---|
| S1 | REFUND — INR 128.25 | PASS |
| S2 | REFUND — INR 135.00 | PASS |
| S3 | ESCALATE — INR 1,999.00 | PASS |
| S4 | DENIED | PASS |
| S5 | DENIED | PASS |
| S6 | NOT_FOUND | PASS |
| S7 | FALLBACK_MAX_ITERATIONS | PASS |

The saved `agent_runs.jsonl` was checked programmatically and contained all seven expected scenario statuses.

## Safe stopping

The agent demonstrates three safety fallbacks:
- Token limit: `FALLBACK_TOKEN_LIMIT`
- Timeout: `FALLBACK_TIMEOUT`
- Maximum iterations: `FALLBACK_MAX_ITERATIONS`

The recorded safe-stopping run verified that execution stops at the configured limit and prevents additional tool execution where required.

## Checkpoint and resume

The S1 demonstration simulates an interruption after two completed tools. The checkpoint stores the run/scenario identifiers, conversation messages, token usage, completed step and executed tools. Resume continues with the remaining tools without re-executing the first two.

Verified tool sequence across the interrupted and resumed execution:

```text
get_order
search_policy
calculate_refund
issue_refund
```

The verified result was `REFUND: INR 128.25 issued.`, with five model calls and four tool executions in total.

## Generated outputs

Part D outputs are stored in `outputs/part_d/`:
- `policy_chunks.csv`
- `policy_chunks_no_overlap.csv`
- `rag_evaluation_details.csv`
- `rag_evaluation_summary.csv`
- `agent_runs.jsonl`
- `safe_stopping_demo.json`
- `checkpoint_resume_result.json`
- `agent_comparison.txt`

The simulated refund ledger is stored separately at `outputs/refund_ledger.jsonl`. Checkpoint state is stored under `state/`.

## Run commands

Run from the repository root after Parts A–C:

```bash
python -m src.part_d.document_chunking
python -m src.part_d.retrieval
python -m src.part_d.rag_evaluation
python -m src.part_d.refund_tools
python -m src.part_d.react_agent
python -m src.part_d.checkpoint_resume
python -m src.part_d.agent_comparison
```

Some commands write or append generated outputs. In particular, running scenario and checkpoint demonstrations can append additional simulated refund actions to the ledger. Repeated runs should therefore be treated as new simulation runs, not as independent real transactions.

## Verification scope and limitations

The recorded checks cover tool registration and validation, RAG evaluation, seven scenario outcomes, token-limit/timeout/iteration fallbacks, and checkpoint recovery. The model is the supplied mock implementation, retrieval is TF-IDF/cosine similarity, and refund processing is simulated. These results are for the supplied synthetic benchmark and do not constitute production deployment validation.
