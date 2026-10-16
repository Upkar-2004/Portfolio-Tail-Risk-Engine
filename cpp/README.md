# Planned C++ Simulation Phase

The validated Python implementation is the current research reference. A later
engineering phase will port selected numerical simulation components to C++20
and expose them through a thin pybind11 interface.

The planned C++ scope is deliberately limited to:

- numerical input validation;
- deterministic correlated-scenario generation;
- portfolio loss calculation;
- empirical VaR and Expected Shortfall extraction; and
- correctness and performance comparisons against the Python reference.

No C++ implementation is included in the current portfolio baseline. Source,
header, binding, CMake, and native-test files will be introduced only when they
contain working code and their behavior is covered by tests. The detailed
milestones remain in [`docs/project-checkpoints.md`](../docs/project-checkpoints.md).
