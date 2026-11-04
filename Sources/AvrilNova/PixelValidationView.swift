import SwiftUI

struct PixelValidationView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            HStack {
                SectionHeading(
                    eyebrow: "Spatial evidence",
                    title: "Pixel Validation",
                    detail: "Confirm that retained transit power remains on-target across sectors and apertures."
                )
                Spacer()
                StatusPill(label: model.selectedTarget?.title ?? "No target", tint: NovaTheme.cyan)
            }

            HStack(spacing: 14) {
                VStack(spacing: 0) {
                    PanelTitle("Target Pixel File", subtitle: "Median image · aperture · centroid", symbol: "square.grid.3x3.fill")
                        .padding(14)
                    Divider().overlay(NovaTheme.border)
                    EmptyScienceState(
                        symbol: "camera.metering.center.weighted",
                        title: "Pixel evidence is acquired after signal retention",
                        message: "Enable pixel follow-up in the Workbench. Avril will display the aperture, difference image, centroid motion, and neighbor contamination here."
                    )
                }
                .background(NovaTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius))
                .overlay(RoundedRectangle(cornerRadius: NovaTheme.panelRadius).stroke(NovaTheme.border))
                .frame(maxWidth: .infinity)

                VStack(alignment: .leading, spacing: 13) {
                    PanelTitle("Validation Gates", subtitle: "Hard rejects and review-only warnings", symbol: "checklist")
                    gate("Difference-image source", "Waiting", NovaTheme.inkTertiary)
                    gate("Centroid stability", "Waiting", NovaTheme.inkTertiary)
                    gate("Aperture consistency", "Waiting", NovaTheme.inkTertiary)
                    gate("Nearby-source contamination", "Waiting", NovaTheme.inkTertiary)
                    Divider().overlay(NovaTheme.border)
                    Text("A pixel warning never short-circuits significance computation. The complete evidence ledger remains available for audit.")
                        .font(.nova(size: 10.5))
                        .foregroundStyle(NovaTheme.inkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                }
                .novaPanel()
                .frame(width: 350)
            }
        }
        .padding(18)
        .background(NovaTheme.canvas)
    }

    private func gate(_ title: String, _ state: String, _ tint: Color) -> some View {
        HStack {
            Image(systemName: "circle.dotted").foregroundStyle(tint)
            Text(title).font(.nova(size: 11.5)).foregroundStyle(NovaTheme.ink)
            Spacer()
            Text(state).font(.nova(size: 10, weight: .medium)).foregroundStyle(tint)
        }
        .padding(10)
        .background(NovaTheme.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
