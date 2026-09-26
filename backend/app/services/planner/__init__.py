"""Analysis planner: a question becomes a validated JSON plan, run on Pandas.

  execute   plan grammar (ALLOWED_*), validation, execute_plan — no LLM
  fallback  rule-based plan (build_fallback_plan)
  llm_plan  planner prompt + repairs of the LLM's plan
  answer    charts + deterministic answer from a result frame

Import from the submodule that defines a name. Dependency order:
execute ← answer, fallback ← llm_plan.
"""
