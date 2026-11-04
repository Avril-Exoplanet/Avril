import AppKit
import SwiftUI

enum NovaTheme {
    static let canvas = Color(hex: 0xEEF1F4)
    static let sidebar = Color(hex: 0xF4F6F8)
    static let chrome = Color(hex: 0xF7F8FA)
    static let chromeRaised = Color(hex: 0xEEF1F4)
    static let chromeBorder = Color(hex: 0xCED5DC)
    static let surface = Color(hex: 0xFBFCFD)
    static let surfaceMuted = Color(hex: 0xF2F5F7)
    static let surfaceStrong = Color(hex: 0xE8EDF2)
    static let border = Color(hex: 0xCBD3DB)
    static let borderStrong = Color(hex: 0xAEBAC5)
    static let ink = Color(hex: 0x18232D)
    static let inkSecondary = Color(hex: 0x4B5966)
    static let inkTertiary = Color(hex: 0x73818E)
    static let accent = Color(hex: 0x0576CE)
    static let accentDark = Color(hex: 0x005A9E)
    static let accentSoft = Color(hex: 0xDCECF8)
    static let cyan = Color(hex: 0x00A6C7)
    static let green = Color(hex: 0x14866D)
    static let amber = Color(hex: 0xB66A00)
    static let red = Color(hex: 0xC43A48)
    static let violet = Color(hex: 0x6554C0)

    static let panelRadius: CGFloat = 6
    static let controlRadius: CGFloat = 5
    static let panelShadow = Color.black.opacity(0.025)
}

enum NovaTypography {
    /// AVRIL is a dense scientific workstation, but its smallest text still needs
    /// to remain comfortably readable on a Retina Mac display. Large headings and
    /// metrics keep their authored size; compact labels and body copy receive a
    /// conservative optical lift without scaling panel geometry.
    static func scaled(_ authoredSize: CGFloat) -> CGFloat {
        switch authoredSize {
        case ...7.0: 8.5
        case ...8.0: 9.5
        case ...9.0: 10
        case ...10.5: 11.5
        case ...11.0: 12
        case ...12.5: 13
        default: authoredSize
        }
    }
}

extension Font {
    static func nova(
        size: CGFloat,
        weight: Font.Weight = .regular,
        design: Font.Design = .default
    ) -> Font {
        .system(size: NovaTypography.scaled(size), weight: weight, design: design)
    }
}

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: alpha
        )
    }
}

enum ResourceLocator {
    static func url(named name: String, extension ext: String) -> URL? {
        if let bundled = Bundle.main.url(forResource: name, withExtension: ext) {
            return bundled
        }

        // SwiftPM exposes declared resources through Bundle.module.  The app
        // bundle path above remains the source of truth for signed releases.
        #if SWIFT_PACKAGE
        if let packaged = Bundle.module.url(forResource: name, withExtension: ext) {
            return packaged
        }
        #endif

        return nil
    }
}

extension View {
    func novaPanel(padding: CGFloat = 16) -> some View {
        self
            .padding(padding)
            .background(NovaTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous)
                    .stroke(NovaTheme.border, lineWidth: 1)
            }
            .shadow(color: NovaTheme.panelShadow, radius: 3, x: 0, y: 1)
    }
}
