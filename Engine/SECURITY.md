# Security

Report security issues privately to the repository maintainers before opening a
public issue. Avril reads untrusted FITS and JSON inputs; run third-party data in
an unprivileged environment. The command-line engine does not require root.

The optional web bridge must bind to loopback, use an unguessable job token,
validate origins, restrict paths to configured data roots, and never expose a
general shell. Do not store catalogue credentials, SSH passwords, or Cloudflare
tokens in manifests, logs, or release archives.
