// Bundled fallback ESLint flat config, used by the JS/TS linter only when the
// reviewed project ships no eslint config of its own. Intentionally uses ONLY
// built-in rules so it works against a bare `eslint` install (no plugins to
// resolve). Projects with their own config are always preferred.
export default [
  {
    ignores: [
      "node_modules/**",
      "dist/**",
      "build/**",
      "coverage/**",
      ".venv/**",
      "venv/**",
    ],
  },
  {
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": "warn",
      "no-debugger": "warn",
      "no-unreachable": "warn",
      "no-constant-condition": "warn",
      "no-dupe-keys": "error",
      "no-duplicate-case": "error",
    },
  },
];
