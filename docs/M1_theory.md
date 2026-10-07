# M1 — A closed-loop policy-gap bound as the architecture-ranking signal

## Setting
Episode of length T (control steps), binary success R in {0,1}. A chunked policy pi_{a} (architecture a with
inference schedule (K steps, horizon h)) observes s every h steps and executes the first h actions of its chunk.
Write J(pi) = P(success). Let pi_r be a reference policy (any architecture, executed with its own schedule).

## Bound (simulation lemma, decision-level MDP)
Treat each chunk decision as one macro-action. With N = T/h decisions and a per-decision coupling where the two
policies share the environment noise (common random numbers), the probability that the two trajectories ever
diverge is at most the sum of per-decision divergence probabilities along the reference's visited distribution:

    |J(pi_a) - J(pi_r)|  <=  sum_{n<N} E_{s ~ d_n^{pi_r}} [ TV( pi_a(.|s), pi_r(.|s) ) ]                 (1)

For Gaussian-like action-chunk distributions with shared noise, TV is upper-bounded by a Lipschitz function of the
mean executed-chunk discrepancy, giving the usable surrogate

    J(pi_a)  >=  J(pi_r)  -  L * (T / h) * D_h(a; r),
    D_h(a; r) = E_{s ~ d^{pi_r}} || A_a(s)[0:h] - A_r(s)[0:h] ||_1 / h                                       (2)

where A(s)[0:h] are the executed (denoised, with the candidate's own K) actions and L is a task-dependent constant
(calibrated, see below). Mismatch in the horizon between candidate and reference enters via A_r executed with the
candidate's h (the reference is re-queried at the candidate's decision points).

## Consequences (tested in the paper)
1. Flow-matching validation loss is not an upper bound on D_h: it averages velocity errors over noise levels t on
   expert states and is invariant to K and h. Hence it can be uninformative (observed tau = 0.165).
2. D_h is measured on d^{pi_r}, the states the *reference* visits -> the bound is tight only near pi_r. Ranking by
   D_h(.; r) therefore favours architectures that imitate r (observed anchor bias of v1).
3. **Bootstrapped reference.** During search, replace r by the incumbent (best arch by closed-loop evidence so far)
   and re-record d^{pi_r}. Each candidate's lower bound LB(a) = J_hat(r) - L * (T/h) * D_h(a; r) is then informative
   around the incumbent, removing the anchor bias.
4. **Pruning (M3).** The bound is symmetric: J(pi_a) <= J(pi_r) + L*(T/h)*D_h(a;r). With J_hat(r) estimated from
   paired rollouts, the incumbent's lower confidence bound LCB(r) is known. A candidate is pruned without any rollout
   when its upper bound J_hat(r) + L*(T/h)*D_h(a;r) cannot exceed LCB(r) by a margin; surviving candidates enter
   paired successive halving. When a candidate beats the incumbent with paired significance, it becomes the new
   reference (bootstrapping) and D_h is recomputed on its visited states.

5. **Certification of cheaper architectures (efficiency objective).** For a cost-constrained search (per-step compute,
   deploy memory), a candidate that is cheaper than the incumbent and satisfies LB(a) = J_hat(r) - L*(T/h)*D_h(a;r)
   >= target can be accepted with *no* rollouts; only uncertified candidates are raced. This turns the bound into a
   budget-saving device exactly where NAS-for-efficiency needs it (shrinking a known-good policy).

## Calibration of L
Fit L on the archs already raced (paired rollouts give J_hat(a) - J_hat(r)); use the 90th percentile of
|J_hat(a)-J_hat(r)| / ((T/h) D_h(a;r)) so the bound holds empirically for 90% of pairs. Report coverage.

## Implementation map
- `rnas/proxies.py --ref-arch --ref-ckpt --record-dir`: D_h(a; r) on r-visited states (records from `rnas/rollout.py
  --record-dir` run with the reference).
- `rnas/search.py: bound_race`: incumbent-bootstrapped racing with bound pruning (replay uses precomputed D for a set
  of references).
