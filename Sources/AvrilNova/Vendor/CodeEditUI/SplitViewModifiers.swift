//
//  SplitViewModifiers.swift
//  CodeEdit
//
//  Created by Wouter Hennen on 05/03/2023.
//

import SwiftUI

struct SplitViewControllerLayoutValueKey: _ViewTraitKey {
    // CodeEdit targets Swift 5.x.  Under Swift 6 a global closure must carry
    // its UI isolation explicitly; the split-view controller is AppKit UI.
    static let defaultValue: @MainActor () -> SplitViewController? = { nil }
}

struct SplitViewItemCollapsedViewTraitKey: _ViewTraitKey {
    static let defaultValue: Binding<Bool> = .constant(false)
}

struct SplitViewItemCanCollapseViewTraitKey: _ViewTraitKey {
    static let defaultValue: Bool = false
}

struct SplitViewHoldingPriorityTraitKey: _ViewTraitKey {
    static let defaultValue: NSLayoutConstraint.Priority = .defaultLow
}

struct SplitViewItemCanAnimateViewTraitKey: _ViewTraitKey {
    static var defaultValue: Bool { true }
}

extension View {
    func collapsed(_ value: Binding<Bool>) -> some View {
        self
        // Use get/set instead of binding directly, so a view update will be triggered if the binding changes.
            ._trait(SplitViewItemCollapsedViewTraitKey.self, .init {
                value.wrappedValue
            } set: {
                value.wrappedValue = $0
            })
    }

    func collapsable() -> some View {
        self
            ._trait(SplitViewItemCanCollapseViewTraitKey.self, true)
    }

    func holdingPriority(_ priority: NSLayoutConstraint.Priority) -> some View {
        self
            ._trait(SplitViewHoldingPriorityTraitKey.self, priority)
    }

    func splitViewCanAnimate(_ enabled: Binding<Bool>) -> some View {
        self._trait(SplitViewItemCanAnimateViewTraitKey.self, enabled.wrappedValue)
    }
}
