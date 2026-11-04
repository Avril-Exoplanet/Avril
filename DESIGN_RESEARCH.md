# Design Research Record

Research date: 2026-11-02

## Primary references

- Adobe Spectrum 2 introduction: https://spectrum.adobe.com/getting-started/introduction/
- Apple toolbars: https://developer.apple.com/design/human-interface-guidelines/toolbars
- Apple sidebars: https://developer.apple.com/design/human-interface-guidelines/sidebars
- OVITO pipeline: https://www.ovito.org/manual/usage/pipeline.html
- napari viewer: https://napari.org/stable/getting_started/viewer.html
- KiCad PCB Editor: https://docs.kicad.org/9.0/en/pcbnew/pcbnew.html
- Swift Charts: https://developer.apple.com/documentation/charts
- SceneKit: https://developer.apple.com/documentation/scenekit
- CodeEdit (MIT, native AppKit/SwiftUI): https://github.com/CodeEditApp/CodeEdit
- Core Plot (BSD-style, native Cocoa scientific plotting): https://github.com/core-plot/core-plot
- Qt Advanced Docking System (LGPL 2.1, evaluated as the Qt fallback): https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System

## Applied decisions

1. The new application is a separate product line, not a theme applied to the legacy Qt shell.
2. Native macOS title-bar integration keeps the traffic lights inside the application surface.
3. Navigation, central work surface, contextual inspector, and task status are persistent and spatially stable.
4. Color communicates state; it is not decorative. Gradients are limited to the mission visualization itself.
5. The mission scene is a functional, zoomable native 3D viewport. Light-curve plots use a scientific chart renderer, not a decorative mockup.
6. The target model is explicitly labelled as a ranking prior rather than a planet probability.
7. Pipeline stages remain visible and auditable; warnings do not silently suppress the rest of the evidence computation.
8. Native search, segmented switching, visual-effect material, and panel dividers reuse CodeEdit's maintained implementations rather than local lookalikes. Window geometry remains in Apple's native layout system so every pane keeps a deterministic visible width.
9. Light curves use Core Plot with interactive native plot spaces, robust display ranges, per-sector scatter series, and scientific axes.
