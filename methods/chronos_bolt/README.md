# Chronos-Bolt baseline

Chronos-Bolt is exposed as an independent benchmark method while reusing the
validated baseline path in the imported TS-RAG reproduction runner. This avoids
duplicating model-loading and data-window code. Model weights remain on shared
storage and are not committed.
