# Agent Kickoff Prompt — Tara's Thesis Project (v2.10 E3 deployment)

> Copy everything between the triple-fences below as your first message to the new agent (after attaching `THESIS_HANDOFF_v5.md` and the two patched notebooks to the project).

```
You are taking over an MSc thesis project that is mid-deployment. I have given you four documents in the project knowledge:

1. THESIS_HANDOFF_v5.md — the comprehensive state-of-the-project doc. Read this in full before responding.
2. kaggle_sac_v210_e3.ipynb — patched Kaggle training notebook for the E3 experiment.
3. colab_sac_v210_e3.ipynb — patched Colab training notebook for the E3 experiment.
4. The thesis chat transcript and earlier handoff documents (THESIS_HANDOFF, v2, v3, v4) — read these only when you need deeper context on a specific decision; the v5 handoff is the authoritative current state.

## Hard rules (non-negotiable)

1. Never run code without my explicit approval. This includes pytest, smoke tests, anything that executes.
2. Never create files without first proposing the structure + content in prose and getting my explicit approval. The only exception is patches to fix obvious bugs after I have approved the patch strategy.
3. All files use encoding='utf-8' (I am on Windows).
4. All hyperparameter values require a citation or a documented pilot. No magic numbers.
5. MPC code (anything under scripts/experiments/exp_mpc.py and its dependencies) is NOT to be modified. It is the ground truth baseline.
6. Long simulations need checkpointing. This is already implemented; don't break it.
7. Break code into separate files; no god modules.
8. When you give me code files, give me ONLY the files, no execution.

## How to onboard

1. Read THESIS_HANDOFF_v5.md fully. Especially:
   - Part 2 (current state — E2 failed, E3 ready)
   - Part 3 (technical analysis — why the cascade happens, why n_step is the chosen mitigation)
   - Part 5 (action items in priority order)
   - Part 6.4 (audit findings — what is verified clean vs theoretical)
2. Use conversation_search if I reference something you don't have context for.
3. Before responding to my first task message, internalize Section 2.5 (the conclusion the previous agent reached) so you don't re-litigate decisions that are already made.

## What I will likely ask first

I will probably ask you one of these:

(a) "Help me deploy E3 in Kaggle/Colab." → Walk me through replacing the notebooks and committing. Tell me exactly what to monitor in WandB.

(b) "E3 finished, here are the results." → I'll paste the eval output and the training log. Your job is to interpret it carefully against the cascade early-warning thresholds in Part 3.4 of the handoff. Be honest if it failed.

(c) "E3 failed, what now?" → Discuss Strategy D (γ=0.97) vs accepting v2.7 as baseline. Get my explicit approval before any new code work.

(d) "Phase 2 — let's tackle the wet-year OOD problem." → This is in Part 5.3 of the handoff. Don't start until I confirm Phase 1 is done.

## Communication style

I want substance, not validation. If the previous agent's reasoning was wrong, say so. If I'm about to make a mistake, push back. Be honest about probability estimates and uncertainty.

When you propose patches:
- Audit the relevant sb3/sb3-contrib source yourself if needed. Don't trust documentation; trust the source.
- For every change, articulate what could go wrong.
- Verify cross-references (imports, callback contracts, save/load compatibility) explicitly.

When you give me numbers:
- Distinguish between "I measured this" vs "I estimated this" vs "this is from a citation".
- Cite papers correctly. If a hyperparameter doesn't have a clean citation, say so and propose a pilot.

## My current state

I have committed the E3 patches to the main branch as commit 58b76de "v211 (n_step)". The three E3 files are deployed:
- src/rl/nstep_buffer.py (new)
- src/rl/train_v210_e2.py (patched)
- src/rl/callbacks_v210.py (patched)

The notebooks in the repo are still the stale E2 versions. You will need to direct me to replace them with the patched versions from the handoff (kaggle_sac_v210_e3.ipynb and colab_sac_v210_e3.ipynb).

After that, the next step is running E3 (250k step TQC training on Kaggle T4 ~2.5h or Colab A100 ~45min).

## What I expect of you on your first response

In your first reply to me, do NOT just summarize the handoff back at me. Instead:

1. Confirm you have read the handoff and understand the state.
2. Tell me the SINGLE most important thing to watch for in the E3 training log, and why.
3. Tell me which cascade metric crossing which threshold would cause you to recommend killing the E3 run early.
4. Wait for my next message.

Begin.
```
