# Third-party notices and asset provenance

This file is part of the source distribution. Credits are kept here rather than placed in the application chrome. Inclusion does not imply endorsement by NASA, ESA, CSA, STScI, Apple, Adobe, or any other organization.

## Native application components

### Core Plot

- Project: https://github.com/core-plot/core-plot
- Pinned source commit: `193227944581d0d77e98b6e6fd4185776216a60a`
- Use: native scientific light-curve rendering
- License: BSD-style; exact text in `VendorLicenses/CorePlot-BSD-LICENSE.txt`
- Vendored binary SHA-256: `c0819a4acf4361a216319ca9ef9570ac9e97900703e35233507f864bd6726ba2`

### CodeEdit components

- Project: https://github.com/CodeEditApp/CodeEdit
- Pinned source commit: `fa2aebd86373211c78626074b53ab75010767575`
- Use: selected native split-view, search, segmented-control, visual-effect, panel-divider, and support implementations under `Sources/AvrilNova/Vendor/CodeEditUI/`
- License: MIT; exact text in `VendorLicenses/CodeEdit-MIT-LICENSE.md`

### Apple platform frameworks

SwiftUI, AppKit, SceneKit, Core Graphics, PDFKit, SF Symbols, and the San Francisco system typeface are supplied by macOS. Apple fonts and symbols are not redistributed as standalone files.

Qt Advanced Docking System was evaluated during design research but is not linked into or distributed with the Swift/AppKit application. Its license copy remains only as an evaluation record and may be removed before the first tag.

## Mission imagery

The following images are used for mission context, not for scientific measurement.

### TESS 2022 all-sky mosaic

- File: `Resources/tess-allsky-2022-5k.jpg`
- Source: https://svs.gsfc.nasa.gov/14265
- Original asset: `TESS_NandS_12-2022_5k.jpg`, 5,000 × 2,500
- Credit: NASA/MIT/TESS and Ethan Kruse (University of Maryland College Park)
- SHA-256: `751fb1a15b393e147ee44a684957a4c8dcb82057d96a651c34158c5b87313a65`

### Webb Cosmic Cliffs

- File: `Resources/webb-carina-cosmic-cliffs-2000.png`
- Source: https://science.nasa.gov/asset/webb/cosmic-cliffs-in-the-carina-nebula-nircam-and-miri-composite-image/
- Original asset: 2,000 × 693 NIRCam/MIRI composite
- Credit: NASA, ESA, CSA, STScI
- SHA-256: `52d31c1262cc432b15c25299afdc7c5bb7ac6a1100bac8ad0c11182e53555cd5`

### Blue Marble Earth texture

- File: `Resources/earth-blue-marble.jpg`
- Source record: https://svs.gsfc.nasa.gov/2915
- Description: 2,048 × 1,024 texture derived from NASA's Blue Marble global mosaic
- Credit: NASA Goddard Space Flight Center; Blue Marble source team and contributors as listed on the source record
- SHA-256: `d4dc80a6ef571939d0abe04a9bed3d3d1e6cd63e59514be1c5e43a6b069e6f1e`

NASA content is used for informational and educational scientific visualization under the NASA Media Usage Guidelines: https://www.nasa.gov/nasa-brand-center/images-and-media/ . NASA should be acknowledged as the source, and the imagery must not be used to imply endorsement. Third-party credits shown on the source pages remain applicable.

## Python and CUDA ecosystem

The engine installs separately distributed packages whose licenses remain their own. Runtime projects include NumPy, SciPy, Astropy, Matplotlib, Pandas, Lightkurve, Astroquery, Numba, Exovetter, tess-point, Transit Least Squares, CuPy, and PyCUDA. The exact installed versions are constrained by `Engine/pyproject.toml` and `Engine/requirements/`; binary releases must include a generated dependency and license inventory for the embedded environment.

TESS observations and SPOC products are obtained from MAST. Researchers are responsible for following NASA/STScI acknowledgment and data-use guidance in scientific publications.

## Generated publication artwork

`docs/assets/avril-hero-scene.png` was generated with the built-in OpenAI image-generation tool from a project-authored prompt on 2026-11-05. It contains no copied institutional logo or source photograph. The final typography, crop, scientific wording, and red publication strip in `docs/assets/avril-hero.png` are deterministically composed by `scripts/render_hero.swift`. The method and model figures are original project-authored SVG diagrams; the Diffusion-TS repository informed their paper-style information hierarchy but none of its figure pixels or artwork are redistributed.

The event-first implementation in `Engine/multiplanet_trial/v03/event_search.py` is an independent implementation informed by the public QuickFind description in arXiv:2103.10285. It does not redistribute or claim to reproduce the non-open-source LcTools program.

## Design references

Adobe Spectrum, Apple Human Interface Guidelines, OVITO, napari, KiCad, and other professional systems informed interaction and information-hierarchy research. Their artwork, trademarks, branding, and application code are not bundled unless explicitly listed above. See `DESIGN_RESEARCH.md`.
