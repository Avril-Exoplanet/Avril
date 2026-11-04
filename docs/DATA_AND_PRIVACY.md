# Data and privacy

Avril is local-first. Computation runs on the researcher's machine. The desktop application starts an owned loopback process bound to `127.0.0.1`; it does not upload FITS products, candidate tables, or reports to an Avril service.

## Network access

Network requests are limited to the public astronomy services selected by the user or required by the requested workflow, including MAST/TESS product discovery and post-search catalog queries. Pixel products are not downloaded for review unless the user enables follow-up or runs an FFI self-photometry path that requires raw cutouts as input.

## Local storage

Set `AVRIL_DATA_DIR` to choose the science-data location. Without it, Avril uses:

- macOS: `~/Library/Application Support/Avril`;
- Windows: `%LOCALAPPDATA%\\Avril`;
- Linux: `$XDG_DATA_HOME/avril` or `~/.local/share/avril`.

Do not commit FITS files, run outputs, engine logs, local environments, API credentials, SSH credentials, or exported candidate reports to the source repository.

## Export

JSON, compressed audit artifacts, CSV tables, and PDF reports are written only when requested. Researchers remain responsible for removing personal paths or unpublished coordinates from artifacts before sharing them.
