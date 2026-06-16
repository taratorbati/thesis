#

SAC GRADIENT STEP (runs every env step):

  1. sample minibatch {(s,a,r,s′)} from replay buffer
  2. CRITIC:  ã′ ~ π(·|s′)                              # sample next action (stochastic)
              y = r + γ·[ min(Q1_t,Q2_t)(s′,ã′) − α·logπ(ã′|s′) ]   # soft target (+entropy)
              minimise (Q1−y)² + (Q2−y)²   over the VDN sum Σ_n Q_n
  3. ACTOR:   ã ~ π(·|s)                                # reparameterised sample
              maximise E[ min(Q1,Q2)(s,ã) − α·logπ(ã|s) ]           # value + entropy
  4. TARGETS: θ_target ← τ·θ + (1−τ)·θ_target   (τ=0.005, every step)
     (α is FIXED at 0.002 — no auto-tune in v2.18)

TD3 GRADIENT STEP (runs every env step):

  1. sample minibatch of EXACT 5-step transitions {(s, a, R_5, s′)} from buffer
  2. CRITIC (every step):
        a′ = π_target(s′) + clip(ε,−0.5,0.5),  ε~N(0,0.2)     # target smoothing
        y  = R_5 + γ^5 · min(Q1_target, Q2_target)(s′, a′)    # twin-min, no entropy
        minimise (Q1−y)² + (Q2−y)²   over the VDN sum Σ_n Q_n
  3. ACTOR (every policy_delay = 2 steps):                    # DELAYED
        maximise E[ Q1(s, π(s)) ]                             # Q1 only, deterministic
  4. TARGETS (every policy_delay steps):
        θ_target ← τ·θ + (1−τ)·θ_target   (τ=0.005)

