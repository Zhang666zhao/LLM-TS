# Data contract

No datasets are stored in this repository. `manifests/` contains only dataset
identity, split, shape, and provenance metadata needed to prove that methods
were evaluated on the same benchmark.

Large source files and prepared retrieval views remain on shared storage. A run
must record the manifest ID and SHA-256 value. Absolute user-specific NFS paths
must not appear in committed configuration.
