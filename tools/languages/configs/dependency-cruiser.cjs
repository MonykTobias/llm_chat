// Bundled dependency-cruiser config for the JS/TS import check. Mounted into the
// crav-node container at /config and run with `--output-type json`; the three
// forbidden rules below map directly onto the unified report sections:
//   no-unresolvable -> BROKEN, no-circular -> CIRCULAR, no-orphans -> notes.
// Project-local config is not used: this drives check_imports, not the project.
module.exports = {
  forbidden: [
    {
      name: "no-circular",
      severity: "error",
      from: {},
      to: { circular: true },
    },
    {
      name: "no-unresolvable",
      severity: "error",
      from: {},
      to: { couldNotResolve: true },
    },
    {
      name: "no-orphans",
      severity: "warn",
      from: {
        orphan: true,
        pathNot: [
          "(^|/)\\.[^/]+\\.(js|cjs|mjs|ts)$", // dotfiles like .eslintrc.js
          "\\.d\\.ts$",                        // type declarations
          "(^|/)(index|main)\\.(js|cjs|mjs|ts|tsx|jsx)$",
        ],
      },
      to: {},
    },
  ],
  options: {
    doNotFollow: { path: "node_modules" },
    exclude: { path: "node_modules|dist|build|coverage|\\.venv|venv|\\.git" },
    moduleSystems: ["es6", "cjs", "tsd"],
    tsPreCompilationDeps: true, // also see type-only imports in TS
  },
};
