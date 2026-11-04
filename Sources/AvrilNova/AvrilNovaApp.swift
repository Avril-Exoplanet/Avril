import AppKit
import SwiftUI

@MainActor
final class AvrilNovaAppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    func applicationWillTerminate(_ notification: Notification) {
        model.shutdown()
    }
}

@main
struct AvrilNovaApp: App {
    @NSApplicationDelegateAdaptor(AvrilNovaAppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appDelegate.model)
                .background(WindowConfigurator().frame(width: 0, height: 0))
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact(showsTitle: false))
        .defaultSize(width: 1540, height: 920)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Science") {
                Button("Run Selected Target") { appDelegate.model.addSelectedToQueue(runImmediately: true) }
                    .keyboardShortcut("r", modifiers: [.command])
                    .disabled(appDelegate.model.selectedTarget == nil)
                Button("Add to Task Queue") { appDelegate.model.addSelectedToQueue() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(appDelegate.model.selectedTarget == nil)
            }
            CommandMenu("Workspace") {
                ForEach(WorkspaceSection.allCases) { section in
                    Button(section.title) { appDelegate.model.section = section }
                }
                Divider()
                Button("Toggle Inspector") { appDelegate.model.showInspector.toggle() }
                    .keyboardShortcut("i", modifiers: [.command, .option])
            }
        }
    }
}
