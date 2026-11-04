# Web package boundary

The web application is released separately from the scientific engine.

`Avril-V2.0-Algorithm` owns data acquisition, computation, job state, candidate
ledgers, and downloadable reports. `Avril-V2.0-Web` owns presentation,
responsive layout, the terminal-style progress view, and Cloudflare Pages
deployment.

The web package must not import or duplicate scientific kernels. It submits a
target and profile to a loopback bridge, polls structured progress events, and
renders the final result JSON. A disconnected bridge must produce a clear error
and a resettable job state; it must never show a fabricated demonstration result
as a completed real computation.

Public deployment is static. Real computation occurs on the user's machine.
The bridge must bind to loopback and explicitly allow the deployed origin. The
browser queue requires both Stop and Reset actions, and stale jobs must not be
replayed after a page reload unless the server confirms they still exist.
