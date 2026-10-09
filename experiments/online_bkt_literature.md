# Online / incremental estimation of BKT *parameters*: literature check

Compiled 2026-10-09. Scope: work that updates the **global** BKT parameters (prior/init, learn, forget, guess, slip) as new data arrives, without refitting on the full history. It excludes (a) per-student forward filtering of mastery with fixed parameters, which is ordinary BKT inference, and (b) generic online EM for HMMs (Cappé 2011; Cappé & Moulines 2009; Mongillo & Denève 2008), except where (b) is needed to place (a) in context.

## How each claim was verified (read this first)

The session's egress policy **blocked** arxiv.org (and its mirrors ar5iv, export, alphaxiv), educationaldatamining.org, jedm.educationaldatamining.org, learninganalytics.upenn.edu, people.csail.mit.edu, fi.muni.cz, huggingface.co and semanticscholar.org. Only GitHub was reachable. Each claim below therefore carries one of these tags:

- **[V-code]**: I read the primary artifact myself (source code, README, or GitHub issue text).
- **[V-index]**: I confirmed the title, authors, venue and abstract-level content through search-engine indexes of the publisher or arXiv page. I did **not** read the full text, so method details beyond the abstract are not verified.
- **[UNVERIFIED]**: secondary sources only, or nothing found. Treat as a lead, not a fact.

---

## 1. Works on BKT parameter learning, with online/incremental relevance

### 1.1 KT² (Gao et al., 2025): the closest BKT-family work to incremental EM
- **Citation:** Xinyi Gao, Qiucheng Wu, Yang Zhang, Xuechen Liu, Kaizhi Qian, Ying Xu, Shiyu Chang. *A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings.* arXiv:2506.09393 (June 2025), cs.CL. I found no evidence of a peer-reviewed venue.
- **URLs:** https://arxiv.org/abs/2506.09393 · code: https://github.com/UCSB-NLP-Chang/KT2
- **What it does:**
  - **[V-index]** Knowledge-Tree KT (KT²) is a Hidden Markov *Tree* Model over a hierarchy of knowledge concepts. It is BKT-like but not plain per-skill BKT. Parameters are fit with EM, and an "incremental update mechanism" runs as responses arrive. The target setting is low-resource classrooms.
  - **[V-code]** From `KT2/KT/KT.py` and `KT2/KT/graph_update.py`:
    - A **burn-in** batch EM runs to convergence on the first `burn_in_size` (default 10) responses per student, pooled across students.
    - After that, for each new response of the current student, the code:
      1. runs an E-step for that student only (`update_e_step`, an upward/downward pass touching the new leaf);
      2. **re-runs a full M-step over the cached posteriors of all students** (`m_step(graphs, datas, uids+train_uids, ...)`), with one EM step per arrival (`max_step = 1`).
    - This is Neal & Hinton (1998)-style *incremental EM*: refresh one unit's sufficient statistics, then M-step on the pooled total. It is not stochastic-approximation online EM.
    - It has **no discounting, no step size and no pseudo-count prior**.
    - Note that `KT.py` reloads the burn-in parameter graph **for each test student** ("Reset parameter graph for each student"). In the evaluation loop, parameter updates therefore do not carry over from one student to the next. The online adaptation is per-student-trajectory, starting from a shared batch anchor.
- **Update after reading the full text (arXiv v1, uploaded by the owner) [V-paper]:**
  - **The model has no learning over time.** The hidden variables K_ci (mastery of each concept) sit on a tree of concepts. The "transitions" γ_c are parent-to-child entailment: mastering a parent implies mastering its children (eq. 8). They are not transitions across time steps. Emission (eq. 10): P(correct | mastered) = φ_n, which takes one of r_easy, r_med or r_hard according to the item's difficulty bin, and P(correct | not mastered) = ε (a guess).
  - **Section 2.6 "Communal Burn-In":** the first answers of every student (10 per student in the experiments) are pooled into Q_init, and EM runs to convergence to give θ_init.
  - **"Personalized Update":** each student i gets their **own** θ_i, estimated on Q_init ∪ Q_i, with **one EM iteration** per new answer.
  - So KT² is *per-student personalisation from a shared batch anchor*. It is **not** online learning of shared global parameters.
  - Evaluation: 100 students per module, 3 modules each of XES3G5M and MOOCRadar. Average AUC is 0.733 and 0.776, ahead of AKT, SAINT, qDKT and their "-Online" retrained variants, and of Qwen-2.5-7B and Llama-3.2-3B prompting (Table 2).
- **Evaluation:** **[V-index/V-code]** on the XES3G5M and MOOCRadar datasets, with simulated classroom subsets. The abstract says KT² "consistently outperforms strong baselines in realistic online, low-resource settings." I could not read the metric tables (arXiv was blocked).
- **Relation to the proposed design:**
  - KT² shares the *batch anchor, then incremental E/M* structure.
  - It differs on every other component: it keeps per-student posteriors in memory (no forward-only statistics), runs an exact full M-step over all students at every arrival (no periodic M-step), and has no forgetting factor or priors.
  - It is the best published precedent for "batch refit as anchor plus incremental EM" in a KT model. It is not BKT proper and it is a preprint.

### 1.2 pyBKT `partial_fit`: warm-start batch EM, not online EM
- **Citation:** Anirudhan Badrinath, Frederic Wang, Zachary Pardos. *pyBKT: An Accessible Python Library of Bayesian Knowledge Tracing Models.* EDM 2021. arXiv:2105.00385.
- **URLs:** https://educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_237.pdf · https://github.com/CAHLR/pyBKT
- **What it does:**
  - **[V-index]** Batch EM over historical logs, with random restarts. The paper evaluates parameter-recovery accuracy on simulated data and runtime against earlier implementations.
  - **[V-code]** (`source-py/pyBKT/models/Model.py`, lines 82–113 and 394–436 in the local checkout):
    - `partial_fit(data)` copies the previously fitted parameters into the EM initialisation, then runs **full EM to convergence on only the data passed in**.
    - Old sufficient statistics are not retained, so earlier data is "remembered" only through the starting point.
    - All `num_fits` restarts are overwritten with the same warm-start values, so the restarts are redundant in this mode.
    - The M-step (`fit/M_step.py`) is plain maximum likelihood (normalised soft counts) with no Dirichlet/pseudo-count terms. Parameters can only be *fixed*, not regularised.
- **Relation to the proposed design:** this is the only "incremental" parameter API in the main BKT library. It is semantically a warm-started refit on a new batch. It does not consolidate old and new evidence and can drift to whatever the new batch alone supports. It is a useful baseline to compare against, not a precedent.

### 1.3 pyBKT GitHub issue #55: per-student inference, not parameter learning
- **Citation:** keikurono7, "Feature Request: Support for Incremental / Online Updates in pyBKT," CAHLR/pyBKT issue #55, opened 2025-10-04. Status: open, no comments, no maintainer reply.
- **URL:** https://github.com/CAHLR/pyBKT/issues/55
- **What it asks for:** **[V-code]** I read the full issue text. It requests `model.update_single_step(skill, prior_state_prob, observation_success)`, which applies the BKT update equations "using the pre-fitted model parameters (P(G), P(S), P(T))".
- **Classification:** this is category **(a)**, per-student forward filtering. It says nothing about updating the global parameters, so it should not be cited as a request for online parameter estimation.

### 1.4 hmm-scalable (Yudelson): batch only, with gradient, conjugate-gradient and Baum-Welch solvers
- **Citations:**
  - Michael V. Yudelson, Kenneth R. Koedinger, Geoffrey J. Gordon. *Individualized Bayesian Knowledge Tracing Models.* AIED 2013, LNCS 7926, pp. 171–180.
  - Tool: https://github.com/myudelson/hmm-scalable. The older repo, https://github.com/IEDMS/standard-bkt, says "development moved" there.
- **What it does:**
  - **[V-code]** From the README: `trainhmm -s structure.solver`, with solvers 1 = Baum-Welch, 2 = gradient descent, 3 = conjugate gradient (Polak-Ribière, Fletcher-Reeves, Hestenes-Stiefel, Dai-Yuan). The gradient approach follows Levinson, Rabiner & Sondhi (1983). All solvers run full-batch.
  - Other options: `-0` (initial parameters, so warm starts are possible), `-l`/`-u` (box bounds; by default **guess and slip are capped at 0.3**), `-c` (L2 penalty toward centroids, i.e. a Gaussian-style prior), `-B` (block re-estimation of PI, A or B).
  - The README contains **no stochastic/minibatch solver and no incremental or online mode**. The candidate "BKT with SGD in hmm-scalable" is **not supported** by the README.
  - **[V-index]** The AIED 2013 paper fits student-level parameters with gradient-based optimisation on KDD Cup 2010 data (more than 20M rows). Its finding is that individualising learning *speed* helps more than individualising the prior.
- **Relation to the proposed design:**
  - It is the source for practical guardrails: parameter bounds (G, S ≤ 0.3), L2-to-centroid regularisation (comparable to pseudo-count priors), and warm-starting.
  - It is evidence that large-scale BKT is fit by batch refits, not online updates.

### 1.5 StanBKT (Pradhan et al., 2026): full Bayesian, batch
- **Citation:** Siddhartha Pradhan, Yanping Pei, Morgan Lee, Puyuan Zhang, Erin Ottmar, Adam C. Sales. *StanBKT: Rethinking Parameter Estimation in Bayesian Knowledge Tracing.* arXiv:2605.23048 (May 2026).
- **URL:** https://arxiv.org/abs/2605.23048
- **What it does:** **[V-index]** A Stan package for BKT that offers HMC, variational inference, Pathfinder and optimisation. It supports standard, grouped and hierarchical BKT. Evaluated on ASSISTments 2020: the methods have similar predictive performance but differ in cost and in how faithfully they capture the posterior.
- **Online aspect:** none found in the abstract.
- **Relation to the proposed design:** a principled source of priors and posterior uncertainty for the batch anchor. A posterior from the anchor could set the pseudo-counts for online updates, though that last step is my inference, not their claim.

### 1.6 Spectral learning for KT (Falakmasir et al., 2013): method of moments
- **Citation:** Mohammad H. Falakmasir, Zachary A. Pardos, Geoffrey J. Gordon, Peter Brusilovsky. *A Spectral Learning Approach to Knowledge Tracing.* EDM 2013. The Pardos publication list says it won the best student paper award.
- **URL:** https://people.csail.mit.edu/zp/papers/EDMPaper2013fpgb.pdf
- **What it does:** **[V-index]** Spectral (moment-based) estimation of BKT parameters. It reports much faster fitting than EM at the same accuracy, or better accuracy at EM-equivalent time.
- **Online aspect:** not claimed. However, the method-of-moments statistics (low-order co-occurrence counts) are **additive across students**, so they could in principle be accumulated in a stream. That is my inference; I have not seen it published for BKT.
- **Related work:** Falakmasir, Yudelson, Ritter, Koedinger, *Spectral Bayesian Knowledge Tracing*, EDM 2015 short paper, pp. 360–363 (https://www.educationaldatamining.org/EDM2015/proceedings/short360-363.pdf). **[V-index]** Its abstract describes feature- and model-compensation on KDD Cup 2010, not online fitting.

### 1.7 Empirical-probabilities estimation (Hawkins, Heffernan & Baker, 2014)
- **Citation:** William J. Hawkins, Neil T. Heffernan, Ryan S. J. d. Baker. *Learning Bayesian Knowledge Tracing Parameters with a Knowledge Heuristic and Empirical Probabilities.* ITS 2014, LNCS 8474, pp. 150–155.
- **URL:** https://learninganalytics.upenn.edu/ryanbaker/paper_143.pdf
- **What it does:** **[V-index]** A non-EM estimator. A heuristic labels each opportunity as known or unknown, and parameters are then computed as empirical frequencies. It is presented as avoiding EM's local optima, degeneracy and cost.
- **Online aspect:** not claimed. Because the parameters are ratios of counts, a streaming count-based version is easy to imagine. I could not read the full text to check whether the heuristic needs a student's *future* responses. If it does, as with Baker et al. 2008, it is not purely forward-only.

### 1.8 BKT-BF (Baker et al.): grid search
- **What it is:** **[V-index]** Baker's group distributes BKT-BF ("BKT-Brute Force (Grid Search)"). The literature conflicts on how it compares with EM: Pavlik et al. report it is comparable or better; Gong et al. report EM is better.
- **[UNVERIFIED]** Whether its objective is SSR, RMSE or log-likelihood.
- **Relation to the proposed design:** the grid-search objective is a sum over students, so a cached grid of per-cell loss totals could be updated incrementally. This has not been published as such.

### 1.9 Bayesian-Bayesian KT (B²KT; Tschiatschek, Knobelsdorf & Singla, 2022)
- **Citation:** *Equity and Fairness of Bayesian Knowledge Tracing.* EDM 2022 poster, pp. 578–582. arXiv:2205.02333.
- **URL:** https://arxiv.org/abs/2205.02333
- **What it does:** **[V-index]** Puts a posterior over *per-student* BKT parameters and updates it online during interaction, which enables "online individualization."
- **Classification:** this is online **per-student parameter inference**, between categories (a) and the target. It does not update the global population parameters. The abstract names scalability through approximate inference as future work.
- **Relation to the proposed design:** shows that online Bayesian updating of BKT parameters (per student) has been done and evaluated. It is useful for the pseudo-count prior idea, since a population prior is updated by individual evidence.

---

## 2. Identifiability, degeneracy and multiple optima (why online updates are risky)

| Work | Status | Key point for online updating |
|---|---|---|
| J. E. Beck & K.-m. Chang, *Identifiability: A Fundamental Problem of Student Modeling*, User Modeling 2007, LNCS 4511, pp. 137–146. https://doi.org/10.1007/978-3-540-73078-1_17 | [V-index] | Very different (L0, G, S, T) sets fit the same performance curve. The proposed fix is **Dirichlet priors** that pull parameters toward population means. This is the direct BKT precedent for **pseudo-count priors**. |
| R. S. J. d. Baker, A. T. Corbett, V. Aleven, *More Accurate Student Modeling through Contextual Estimation of Slip and Guess Probabilities in BKT*, ITS 2008, LNCS 5091, pp. 406–415. https://learninganalytics.upenn.edu/ryanbaker/BCA2008W.pdf | [V-index] | Dirichlet priors remain vulnerable to **model degeneracy** (e.g. the student is more likely to answer correctly when not knowing the skill). The contextual G/S estimates use a student's *subsequent* responses, so they are a smoothing/batch method, not forward-only. |
| Z. A. Pardos & N. T. Heffernan, *Navigating the parameter space of BKT models: visualizations of the convergence of the EM algorithm*, EDM 2010, pp. 161–170. | [V-index] (via the author's poster) | Synthetic-data maps of EM convergence show a **"dual global maxima"** structure in KT, so the starting point decides which optimum EM reaches. An online learner with noisy steps could switch basins. |
| Z. A. Pardos & N. T. Heffernan, *Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing*, UMAP 2010. https://people.csail.mit.edu/zp/papers/UMAP_final.pdf | [V-index] | Prior-per-student (individualised L0) in BNT-SM, fit with batch EM. Not online. (KT-IDEM, the item-difficulty extension, is Pardos & Heffernan, UMAP 2011.) |
| Y. Gong, J. E. Beck, N. T. Heffernan, EDM 2010 work on multiple Dirichlet priors | [V-index] (secondary description only) | Dirichlet priors "might be hurt by outliers"; trimming helps. |
| B. van de Sande, *Properties of the Bayesian Knowledge Tracing Model*, JEDM 5(2):1–10, 2013. https://jedm.educationaldatamining.org/index.php/JEDM/article/view/35 | [V-index] | Closed-form solution of the population-averaged Markov chain (exponential curve with 3 effective parameters). Fixed-point analysis gives the parameter region with sensible behaviour. Useful for defining the **feasible region / projection step** after each online M-step. |
| S. Doroudi & E. Brunskill, *The Misidentified Identifiability Problem of Bayesian Knowledge Tracing*, EDM 2017 (JEDM track). | [V-index] title/abstract start; the claim that "identifiable under G+S<1" comes from a **secondary** source, so [UNVERIFIED] | The practical problem is *semantic degeneracy*, not mathematical non-identifiability. |
| D. Shchepakin, S. Sankaranarayanan, D. Zimmaro, *Parametric Constraints for BKT from First Principles*, EDM 2024, pp. 18–29. https://educationaldatamining.org/EDM2024/proceedings/2024.EDM-long-papers.2/index.html (arXiv:2401.09456) | [V-index] | Derives necessary constraints on BKT parameters and a constraint-respecting estimator. The authors are from Amazon, which shows production interest in non-degenerate fitting. Its constraints are a good candidate for the projection/clamping step in an online M-step. |
| S. Ritter, T. Harris, T. Nixon, D. Dickison, R. C. Murray, B. Towle, *Reducing the Knowledge Tracing Space*, EDM 2009 (best paper). | [V-index] | Carnegie Learning data (more than 8000 students, 4 courses): the BKT parameter space can be reduced drastically (clustered parameter sets) without changing system behaviour. This supports **shrinkage/tying** to stabilise online estimates for sparse skills. |

Implication for the proposed design. Online EM converges to a stationary point of the likelihood, and BKT has known multiple, semantically opposite maxima. So:
- the batch anchor should select the basin;
- pseudo-counts (Beck & Chang style) and feasibility constraints (hmm-scalable's G, S ≤ 0.3 default; Shchepakin et al. 2024; van de Sande 2013) should keep the iterates in that basin;
- Baker et al. 2008 warn that priors alone do not prevent degeneracy, so an explicit degeneracy check is still needed after each M-step.

---

## 3. Production systems: what is documented about online parameter updating

- **Carnegie Learning MATHia / Cognitive Tutor.** **[V-index]** MATHia's DataShop documentation says mastery uses BKT (Corbett & Anderson 1994). **[UNVERIFIED]** I found nothing public on how parameters are re-estimated (batch or online, and how often).
- **Duolingo Birdbrain.** **[V-index]** blog title only: https://blog.duolingo.com/learning-how-to-help-you-learn-introducing-birdbrain. **[UNVERIFIED]** details come from a secondary source (IEEE Spectrum, https://spectrum.ieee.org/duolingo). The original Birdbrain was an IRT-style logistic model that updates learner ability *and* exercise difficulty after every exercise (an Elo-like online update of global item parameters); a later version is reported to be a recurrent neural network. **It is not BKT.** It is the clearest production example of online updates to *global* item parameters, but in a logistic/IRT model.
- **Pelánek's Elo line of work.** These are online updates of item parameters in logistic models, and the literature that argues for online updates over batch fitting of global parameters:
  - **[V-index]** R. Pelánek, *Applications of the Elo rating system in adaptive educational systems*, Computers & Education 98 (2016), doi:10.1016/j.compedu.2016.03.017. Preprint: https://www.fi.muni.cz/~xpelanek/publications/CAE-elo.pdf.
  - **[V-index]** Pelánek, Papoušek, Řihák, Stanislav, Nižnan, *Elo-based learner modeling for the adaptive practice of facts*, UMUAI 27(1) 2017, doi:10.1007/s11257-016-9185-7.
  - **[V-index]** Nižnan, Pelánek, Řihák, EDM 2015 (https://www.educationaldatamining.org/EDM2015/proceedings/full109-116.pdf): basic Elo predicts well, and more complex models improve only slightly.
  - **[V-index]** Pelánek, *BKT, logistic models, and beyond*, UMUAI 27 (2017) 313–350, doi:10.1007/s11257-017-9193-2.
  - **[UNVERIFIED]** whether any of these runs a head-to-head of online Elo against online-updated BKT. I could not read the full texts. The comparison in this literature is Elo against *batch-fitted* models.
- **ALEKS, Khan Academy, ASSISTments production pipelines.** **[UNVERIFIED]** I found no primary source describing online updates of global BKT parameters. ALEKS uses knowledge-space theory, not BKT.

## 4. Deep KT / continual learning (adjacent)

- **[V-index]** *Deep Trustworthy Knowledge Tracing*, arXiv:1805.10768. It names catastrophic forgetting and state-update failure in DKT and proposes regularisation.
- **[V-index]** LefoKT, AAAI 2025 (https://ojs.aaai.org/index.php/AAAI/article/view/34998). It addresses *growing sequence length*, not parameter streaming.
- I found no paper that evaluates replay, EWC or distillation for KT on streaming cohorts. **[UNVERIFIED]** that none exists, since search coverage was limited.

## 5. Generic HMM incremental/online estimation the design actually rests on

These are already known to the reader and are listed only to show which general result each design element borrows.

- **Neal & Hinton (1998)**, *A view of the EM algorithm that justifies incremental, sparse, and other variants.* This is the incremental EM of KT²: replace one unit's contribution to the sufficient statistics. It fits BKT well because BKT data is **many short independent sequences**, so each student-skill sequence can be treated as one unit.
- **Cappé & Moulines (2009)**, online EM for i.i.d. latent-data models. This is the right framework when each *completed* student sequence is an i.i.d. draw. The step-size schedule γ_n replaces the "discounting" factor, and constant γ gives exponential forgetting.
- **Cappé (2011)** and **Mongillo & Denève (2008)**: forward-only recursion for smoothed additive functionals within one long chain. For BKT this lets a student's *exact* smoothed expected counts be maintained incrementally, without storing their history or rerunning the backward pass. Combined with Neal-Hinton replace-old-contribution bookkeeping, a student's contribution can be refreshed after each response.
- **Le Corff & Fort (2013)**, *Online EM based algorithms for inference in HMMs*, Electron. J. Statist. 7:763–792 (arXiv:1108.3968). **[V-index]** Block online EM with convergence proofs and Polyak averaging. This is the theoretical analogue of "periodic M-steps".
- **Foti, Xu, Laird, Fox (2014)**, *Stochastic Variational Inference for HMMs*, NIPS 2014 (arXiv:1411.1670). **[V-index]** Minibatch SVI with buffered subsequences. It is the Bayesian analogue: a Dirichlet posterior plays the role of the pseudo-count prior.
- **Khreich, Granger, Miri, Sabourin (2012)**, *A survey of techniques for incremental learning of HMM parameters*, Information Sciences 197:105–130. **[V-index]** citation only.

## 6. Mapping to the proposed design

| Design element | BKT-specific precedent | Generic precedent |
|---|---|---|
| Exact forward-only smoothed sufficient statistics per student | **None found** in the BKT literature. KT² runs a fresh upward/downward pass per update; pyBKT and hmm-scalable run full forward-backward. | Cappé 2011; Mongillo & Denève 2008 |
| Global statistics with discounting | **None found** for BKT. The closest analogue is Elo's constant-K updates (Pelánek), in a different model. | Cappé & Moulines 2009 (step size γ); exponential forgetting |
| Periodic M-steps | KT² runs an M-step on *every* arrival over pooled statistics (no batching) | Le Corff & Fort 2013 (block online EM) |
| Pseudo-count priors | Beck & Chang 2007 (Dirichlet priors, batch); Gong et al. 2010; hmm-scalable `-c` L2-to-centroid; StanBKT / hierarchical BKT priors | MAP-EM; SVI (Foti et al. 2014) |
| Constraints / degeneracy guard | hmm-scalable bounds (G, S ≤ 0.3); Shchepakin et al. 2024; van de Sande 2013; Baker et al. 2008 | projection in stochastic approximation |
| Periodic batch refit as anchor | KT² burn-in batch EM; pyBKT `partial_fit` (warm start); common practice of offline fitting (MATHia, pyBKT) | — |
| Evaluation protocol for online parameters | KT² (online, low-resource classroom simulation); no BKT paper found with a forward-in-time comparison of online vs batch parameters | — |

## 7. Verdict

**"Online EM for global BKT parameters" is thinly grounded in the BKT-specific literature. In substance it is an application of general online/incremental EM for HMMs to BKT.**

- I found **no** peer-reviewed BKT paper that maintains streaming sufficient statistics for the global (L0, T, F, G, S) parameters with a step size or discount and evaluates the result against batch refits.
- The nearest works are:
  1. **KT²** (2025 preprint). Incremental EM with a batch burn-in on a BKT-like hidden Markov tree. It recomputes a full M-step per arrival, has no discounting or priors, and resets parameters per test student.
  2. **pyBKT `partial_fit`**. A warm-started batch refit on new data only, with no carried statistics.
  3. **B²KT**. Online Bayesian updating of *per-student* parameters, not the population.
- **Elo-style systems** (Pelánek; Duolingo's original Birdbrain) are the strongest evidence that online updates of global item parameters work in education. They use logistic/IRT models, not BKT.
- The **identifiability/degeneracy literature** is well established and relevant: Beck & Chang 2007, Baker et al. 2008, Pardos & Heffernan 2010, van de Sande 2013, Doroudi & Brunskill 2017, Shchepakin et al. 2024. It is all batch-fitting work. It supports pseudo-count priors, constraints and a batch anchor, but says nothing about how online iterates behave near BKT's dual optima.
- **pyBKT issue #55** is about per-student state updates (category (a)) and should not be cited as precedent for parameter streaming.

**Gaps (open, as far as I could find):**
1. No BKT study of forward-only smoothed statistics (Cappé-style) and their exactness or cost compared with per-student forward-backward.
2. No analysis of how online-EM step sizes or discounting interact with BKT's dual maxima (switching basins, drift toward degenerate G > 0.5 or S > 0.5).
3. No forward-in-time (prospective, cohort-shift) evaluation of online-updated BKT parameters against periodic batch refits, on any public dataset (ASSISTments, KDD Cup 2010, Cognitive Tutor/MATHia).
4. No published treatment of pseudo-count strength as an online regulariser for BKT (Beck & Chang's Dirichlet priors are batch only).
5. No documentation of how production BKT systems (MATHia, ASSISTments) refresh parameters. **[UNVERIFIED]**
6. Streaming versions of the cheap estimators (spectral/moment, empirical probabilities, grid-search loss caches) look possible because their statistics are additive, but are unpublished.

**Full texts read afterwards ([V-paper]),** uploaded by the owner: Cappé & Moulines 2009; Cappé 2011; Mongillo & Denève 2008; Beck & Chang 2007; Pardos & Heffernan 2010; the SQUAREM R vignette (Varadhan); KT² v1; and Khajah's JEDM paper (text). How each maps onto the code is in README.md sections 2–3.

**Caveat on coverage:** egress restrictions blocked arXiv and EDM/JEDM full texts, so everything tagged [V-index] rests on abstracts and listings. A full-text check of KT² (arXiv:2506.09393), Hawkins et al. 2014 and Pardos & Heffernan 2010 is the most valuable follow-up.
