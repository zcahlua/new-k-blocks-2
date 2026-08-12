# Codex execution prompt

Implement `/Users/zcahlua/Desktop/8月work/新方法/algorithm/IMPLEMENTATION_PLAN.md` task by task.

Mandatory boundaries:

1. Keep every project source/data/artifact read and write inside `/Users/zcahlua/Desktop/8月work/新方法/algorithm/`. Python may load its standard-library runtime, but do not deliberately inspect, import, copy, or depend on any external project/data file.
2. Put test and staging scratch directories only under `algorithm/runs/.tmp/`; never use the operating-system default temporary directory.
3. Use Python 3.11+ standard library only; do not install dependencies or access the network.
4. Follow TDD: write the named failing test, run it, implement the smallest correct change, then rerun it.
5. Keep V1 strictly `STRUCTURE_ONLY`; ambient blocks and localization are `null` and `WITH_AMBIENT_BLOCKS` is rejected.
6. Treat full-separation enumeration as a bounded independent oracle; the primary algorithm is Elementary XP.
7. Never export an engine candidate unless the independent verifier returns `verified=true`.
8. Put all cases and outputs below a fresh `algorithm/runs/<run-id>/` and refuse overwrites.
9. Run the complete test suite and the real 18-case demo before claiming completion.
10. Final reporting must name files created, exact commands and results, verified case count, and any unmet gate. Do not claim FPT, ambient-block correctness, SPQR equivalence, or a general proof.

Start with Task 1. At the first failing gate, diagnose it, add a regression test, repair it, and continue only after the gate passes.
