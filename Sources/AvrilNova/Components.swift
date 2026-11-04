import SwiftUI

struct SectionHeading: View {
    let eyebrow: String
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(eyebrow.uppercased())
                .font(.nova(size: 10, weight: .semibold))
                .tracking(1.25)
                .foregroundStyle(NovaTheme.accent)
            Text(title)
                .font(.nova(size: 22, weight: .semibold))
                .foregroundStyle(NovaTheme.ink)
            Text(detail)
                .font(.nova(size: 12.5))
                .foregroundStyle(NovaTheme.inkSecondary)
        }
    }
}

struct PanelTitle: View {
    let title: String
    let subtitle: String?
    let symbol: String?

    init(_ title: String, subtitle: String? = nil, symbol: String? = nil) {
        self.title = title
        self.subtitle = subtitle
        self.symbol = symbol
    }

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Rectangle()
                .fill(NovaTheme.accent)
                .frame(width: 2, height: 24)
            if let symbol {
                Image(systemName: symbol)
                    .font(.nova(size: 13, weight: .medium))
                    .foregroundStyle(NovaTheme.accent)
                    .frame(width: 18, height: 24)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.nova(size: 12, weight: .semibold))
                    .foregroundStyle(NovaTheme.ink)
                if let subtitle {
                    Text(subtitle)
                        .font(.nova(size: 10.5))
                        .foregroundStyle(NovaTheme.inkTertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
        }
    }
}

struct StatusPill: View {
    let label: String
    let tint: Color

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(tint).frame(width: 6, height: 6)
            Text(label)
                .font(.nova(size: 10.5, weight: .medium))
                .lineLimit(1)
        }
        .foregroundStyle(NovaTheme.inkSecondary)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(tint.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 4).stroke(tint.opacity(0.26), lineWidth: 1))
    }
}

struct MetricTile: View {
    let label: String
    let value: String
    var note: String? = nil
    var tint: Color = NovaTheme.accent

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label.uppercased())
                .font(.nova(size: 9, weight: .semibold))
                .tracking(0.7)
                .foregroundStyle(NovaTheme.inkTertiary)
            Text(value)
                .font(.nova(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(NovaTheme.ink)
                .monospacedDigit()
            if let note {
                Text(note)
                    .font(.nova(size: 9.5))
                    .foregroundStyle(tint)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(11)
        .background(NovaTheme.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .stroke(NovaTheme.border, lineWidth: 1)
        }
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.nova(size: compact ? 11 : 12, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, compact ? 11 : 15)
            .frame(height: compact ? 30 : 36)
            .background(configuration.isPressed ? NovaTheme.accentDark : NovaTheme.accent)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.controlRadius, style: .continuous))
            .shadow(color: NovaTheme.accent.opacity(configuration.isPressed ? 0.04 : 0.10), radius: 2, y: 1)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.nova(size: compact ? 11 : 12, weight: .medium))
            .foregroundStyle(NovaTheme.ink)
            .padding(.horizontal, compact ? 10 : 14)
            .frame(height: compact ? 30 : 36)
            .background(configuration.isPressed ? NovaTheme.surfaceStrong : NovaTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.controlRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: NovaTheme.controlRadius, style: .continuous)
                    .stroke(NovaTheme.borderStrong, lineWidth: 1)
            }
            .scaleEffect(configuration.isPressed ? 0.99 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct EmptyScienceState: View {
    let symbol: String
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.nova(size: 25, weight: .light))
                .foregroundStyle(NovaTheme.accent)
                .frame(width: 48, height: 48)
                .background(NovaTheme.accentSoft)
                .clipShape(Circle())
            Text(title)
                .font(.nova(size: 13, weight: .semibold))
                .foregroundStyle(NovaTheme.ink)
            Text(message)
                .font(.nova(size: 11))
                .foregroundStyle(NovaTheme.inkSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
}

struct WorkflowStep: View {
    let index: Int
    let title: String
    let detail: String
    let state: StepState

    enum StepState { case waiting, active, done }

    var tint: Color {
        switch state {
        case .waiting: NovaTheme.inkTertiary
        case .active: NovaTheme.accent
        case .done: NovaTheme.green
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle().fill(tint.opacity(0.11)).frame(width: 25, height: 25)
                if state == .done {
                    Image(systemName: "checkmark").font(.nova(size: 10, weight: .bold))
                } else {
                    Text("\(index)").font(.nova(size: 10, weight: .semibold))
                }
            }
            .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.nova(size: 11, weight: .medium)).foregroundStyle(NovaTheme.ink)
                Text(detail).font(.nova(size: 9.5)).foregroundStyle(NovaTheme.inkTertiary).lineLimit(1)
            }
            Spacer()
            if state == .active {
                ProgressView().controlSize(.small).tint(NovaTheme.accent)
            }
        }
    }
}

extension Double {
    var oneDecimal: String { String(format: "%.1f", self) }
    var percentText: String { String(format: "%.0f%%", self <= 1 ? self * 100 : self) }
}
