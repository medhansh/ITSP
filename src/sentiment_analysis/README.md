# Sentiment analysis — not in scope for this pass

Per the project abstract, this module will eventually process Indian financial
news and regional-language sources to feed a sentiment signal into the
regime-aware strategy layer. It is intentionally left unimplemented in this
pass (regime detection + fundamental analysis first) — this stub exists only
so the rest of the codebase has a stable `src.sentiment_analysis` import path
to build against later.
