# Security policy

## Reporting

Use GitHub private vulnerability reporting for security issues once the repository is public. Until that channel is enabled, contact the maintainers privately rather than opening a public issue. Do not include sensitive FITS products, credentials, unpublished coordinates, or personal paths in a report.

## Scope

Security-sensitive surfaces include:

- parsing of untrusted FITS, JSON, CSV, gzip, and manifest inputs;
- file and artifact path validation;
- the loopback desktop bridge and per-job control token;
- child-process ownership and shutdown;
- archive downloads and resume behavior;
- packaged Python/native dependencies;
- PDF/CSV/JSON export paths.

The bridge must bind to loopback, must not expose a general shell, and must restrict reads and writes to configured data roots. Avril does not require root privileges.

## Operational guidance

Run third-party data in an unprivileged account. Keep the application, Python runtime, and scientific dependencies updated from trusted release artifacts. Do not put catalog credentials, SSH passwords, signing identities, API tokens, or Cloudflare credentials in manifests, logs, issue reports, or release archives.

## Supported versions

Only the latest tagged research-preview release receives security fixes. Historical snapshots are retained for scientific reproducibility but may contain unpatched dependencies.
