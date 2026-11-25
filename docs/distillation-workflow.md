# Distillation Workflow for Speculative Decoding

Your work is to build reproducible datasets, measure which predictions depend on distant evidence, and train drafts that test whether giving those predictions more weight helps speculative decoding. Student A owns model loading, decoder correctness, and inference benchmarks. Your first handoff is a small set of paired retrieval prompts that A can use to measure draft–target disagreement before the speculative decoder is ready.

DistillSpec provides a precedent for using knowledge distillation (KD) to improve draft–target alignment. This project asks whether **teacher sensitivity to distant context adds value beyond ordinary KD and balanced short/long training**. [DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)

## Four Week Deadline

Submit the full research report, figures, and reproducibility package by **Day 28 counting from kickoff**. Finish training by Day 19 and timed evaluation by Day 24 so the final four days are available for analysis, writing, and review. Start the report in Week 1, then fill in methods and results as runs finish.

| Due | Student B | Student A | Shared outcome |
| --- | --- | --- | --- |
| **Day 2** | Start the prompt generator and verify access to PG-19 and UltraChat. | Verify GPU access and check that both candidate models can load. | Agree on data schema, tokenizer, chat template, GPU time blocks, and the minimum evaluation matrix. |
| **Day 7** | Hand off paired pilot prompts with token positions, split manifests, and a reloadable tiny KD checkpoint. | Verify model compatibility, greedy output equivalence on smoke prompts, and that the benchmark logs acceptance and latency. Start the mismatch pilot once prompts arrive. | **G1** · Usable prompts, checkpoint handoff, and a working benchmark on a small batch of prompts. |
| **Day 14** | Validate full/local sensitivity, finish held-out PG-19 and UltraChat subsets, and start M2 as soon as the pilot and trainer are usable. Use run timings to set a shared token budget. | Finish the mismatch and teacher-accuracy pilot. Time target-only and M0 while B prepares checkpoints. | **G2–G3** · Record what the pilot supports, choose the final experiment size, and decide whether to run M3. Freeze test inputs and short-context regression criteria. |
| **Day 19** | Hand off M2 and, if the sensitivity check supports it, M3. Fit M1 into the same budget if feasible. Give A each checkpoint and its manifest as soon as it is ready. | Benchmark target-only, M0, and each completed checkpoint as it arrives. Check greedy output equivalence for each draft before timing. | **G4** · Freeze checkpoint selection and training configs. Run M4 and extra ablations only if they fit before this cutoff without delaying the core runs. |
| **Day 24** | Verify training and scoring costs, review paired results and failure cases, and complete the methods draft. | Finish repeated timing and acceptance measurements on the held-out workloads. | Freeze results and rerun only measurements with validity problems within the reserved benchmark window. |
| **Days 25–28** | Write the data, training, and interpretation sections. Verify that the manifests identify the data and checkpoints used in the tables. | Write the decoder and systems sections and finalize tables and figures. | **G5** · Review the full draft by Day 27 and submit by Day 28. |

The initial candidates are Qwen2.5-3B-Instruct as target and Qwen2.5-0.5B-Instruct as draft. Begin GPU smoke tests at batch size 1 with short inputs and expand after A verifies token-ID compatibility and memory use. Use the first profiling and training runs to estimate how much work fits before the cutoffs. [Two-person plan](../Speculative_Decoding_Two_Person_Research_Plan%281%29.docx) · [Experiment protocol](experiment-protocol.md)

## Minimum Experiment and Scope Cuts

- Protect **target-only, M0, M2, and M3** when the sensitivity check supports M3. M0 gives the untuned reference. M2 is the uniform-KD control on the same training mix as M3. M1 adds a separate estimate of the effect of changing the training mix. Keep controlled retrieval, a small PG-19 continuation set, and a small UltraChat set for each evaluated draft.
- If the Week 2 timing estimate exceeds the GPU budget, drop LongBench, NoLiMa, M4, and extra ablations first. Reduce the retrieval sweep to two lengths and two shared evidence distances and use a smaller held-out set per workload. For example, use 2K and 4K with evidence near 128 and 1,024 tokens if both lengths fit. Use a smaller, equal training-token budget for M2 and M3 and retain repeated timing. Record the reduced sample sizes and uncertainty. If needed, drop M1 and limit claims about the effect of changing the training mix.
- If the Day 14 pilot gives weak evidence of a distance effect, use the remaining time for a small follow-up only if the controls are sound and it fits before Day 19. If the sensitivity score still fails its controls by Day 14, finish the target-only/M0/M2 comparison and report where the sensitivity hypothesis could and could not be tested. A weak or inconclusive result still belongs in the final report.

## Start With Paired Retrieval Prompts

1. Agree with A on the model and tokenizer revisions, chat template, first answer-token boundary, and seed ranges. Use the fields in the shared protocol and add a `pair_id` so A can join two prompt variants for the same underlying example.
2. Build a deterministic key–value retrieval generator. Start with 8–10 held-out prompt families as a pilot setting. Within a family, hold the query, evidence, answer, and distractor pool constant. Move evidence at fixed length, or add filler while holding evidence distance fixed. Randomize the key and value across families to reduce answer guessing. Work at 1K–2K first to debug and prepare 8K prompts for the first distance comparison.
3. Measure actual positions after final chat formatting. Use the first answer prediction boundary minus the evidence start index as the distance. Check that the evidence appears once, the answer is absent from distractors, variants meet the planned length and distance within a documented tolerance, and pair members stay in the same split. Hand A the pilot JSONL, seed manifest, and measured lengths and distances.
4. Once the pilot is sound, use the roadmap’s sweeps. For the length sweep, start at 1K, 2K, 4K, and 8K tokens with evidence about 128 tokens away. For the distance sweep, hold length at 8K and start near 128, 512, 2K, 4K, and 7K tokens. Expand toward 50–100 paired prompts per feasible cell if the pilot and GPU budget support it.

Keep generated training families separate from validation and test families. Split PG-19 by book before constructing windows and UltraChat by conversation before extracting prompts. Record dataset revisions, source IDs, splits, and token budgets. Use your generated prompts for controlled RULER-style tests and keep any official RULER evaluation held out and labeled separately. Keep LongBench and NoLiMa for the extension if the core runs fit. [Research proposal](../Speculative_Decoding_Research_Proposal.docx) · [Experiment protocol](experiment-protocol.md)

## Validate the Sensitivity Signal

1. For the same next-token target, compute `s_t = KL(p_target(full prefix) || p_target(recent prefix))`. Compare recent windows of 128, 512, and 1,024 tokens on validation data. Use the same answer prefix for both conditions. Start by scoring the first answer token and label any extension to more token positions.
2. Compare examples where the relevant span is inside and outside the recent window. Test length-matched replacement of the evidence and of an irrelevant span at the same position where feasible. Include a position-preserving context-removal control to check whether changed token positions drive the score. Log teacher retrieval accuracy, draft–target KL, and top-1 agreement. Ask A to join greedy rejection data once the decoder is ready. Show paired results with uncertainty for all examples and for the teacher-correct subset.
3. If the teacher frequently misses the answer, adjust the pilot difficulty before interpreting the distance effect. If scores react similarly to evidence and irrelevant-span replacement, investigate the control or revise the score before scaling. At the Day 14 review, decide with A whether the evidence supports full M3 training or a smaller follow-up. Freeze the window, target-token mask, and normalization and clipping settings using train/validation data. Cache scores by dataset and teacher revision and scoring configuration, and measure the cost of producing them.

## Train the Draft Comparisons

Start with LoRA, batch size 1, and gradient accumulation. The smoke run should produce finite losses, show that the draft can learn a tiny batch, and save an adapter that A can reload. For the full experiment, keep the base draft, optimizer settings, valid training-token budget, and evaluation setup comparable across trained variants. Log every departure from those settings.

| Draft | Training recipe | Comparison |
| --- | --- | --- |
| **M0** | Original draft checkpoint | Reference for the effect of ordinary KD and for short-context retention. |
| **M1** | Uniform forward KL from full-context target to draft with conventional data sampling | M1 vs. M0 measures the effect of ordinary KD. |
| **M2** | Same loss with a fixed short/long mix | M2 vs. M1 tests the effect of training exposure. |
| **M3** | Same examples, sampling, and valid token positions as M2, with dependency-sensitive token weights | **M3 vs. M2 is the main test of weighting.** |
| **M4** | Explicitly combine short-context, long-context, and dependency-weighted KD losses | Compare with M3 at matched exposure and with M0/M2 on held-out short prompts. |

For M3, use `w_t = 1 + alpha * normalized_clipped(s_t)` and divide the weighted KL sum by the sum of weights over valid training positions. Set `alpha = 0` as a check that the objective recovers M2 under identical inputs and masks. If the pilot scores only answer tokens, document which tokens receive an extra weight and use identical position masks for M2 and M3. For M4, define the groups and normalize each component before combining them. Use a small validation sweep to choose `alpha` and the M4 coefficients, then lock the final recipe before evaluating on the test sets.

Use M2 as the weighting-removal control. If the required runs fit before the deadline, add a random-weight control with the same weight distribution. If M4 fits, compare it with M3 at matched exposure to test explicit balancing. Add draft–target disagreement weighting only if another run fits. Use shorter contexts and sampled scoring positions if memory is tight. Alternate training and profiling blocks if the team shares one GPU.

## Handoff and Results

Give A a dataset manifest, checkpoint or adapter path, base model and tokenizer revisions, code commit, config, trained token counts and short/long exposure, validation scores, peak memory, and GPU-hours for scoring and training. Start with the shared [run template](../configs/experiment.template.json). Keep datasets, caches, and checkpoints outside Git and commit the code, manifests, and artifact references needed to reproduce the runs.

Review three findings together. First, compare M3 with M2 for held-out long-context KL, top-1 agreement, greedy acceptance, and accepted draft tokens per verification. Second, compare short UltraChat acceptance and timing against M0 and M2. Agree on what counts as a material short-context regression before the final test. Third, use A’s synchronized, repeated throughput and latency measurements on identical inputs to test actual speedup against both M2 and target-only decoding. Check greedy output equivalence and report prefill separately from decode. Use PG-19 for sustained continuation timing and retrieval for the controlled distance analysis. If alignment or acceptance improves while speed stays flat, look at draft cost, verification cost, and memory and report the result as measured. [Experiment protocol](experiment-protocol.md) · [Speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html)
