// Config mínima real (flat config, ESLint 9+) usada pela operação `lint` do checker
// (D-checker-node-server-backend). Regras de correção, não de estilo — pega bugs reais
// (variável não declarada, comparação solta, promise sem tratamento), não briga por formatação.
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      // Sem isso, `console`/`process`/`window`/etc. são marcados como no-undef —
      // achado real ao testar (D-checker-node-server-backend): faltavam globals.
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": "warn",
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  {
    ignores: ["dist/**", "node_modules/**"],
  },
);
