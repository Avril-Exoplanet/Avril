import AppKit
@preconcurrency import SceneKit
import SwiftUI

private final class MissionOrbitSCNView: SCNView {
    private let focus = SCNVector3Zero
    private var orbitCamera: SCNNode?
    private var yaw: CGFloat = 0
    private var pitch: CGFloat = 0.104
    private var distance: CGFloat = 10.56
    private var previousPointer: NSPoint?

    func install(camera: SCNNode) {
        orbitCamera = camera
        pointOfView = camera
        updateCamera()
    }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            yaw = 0
            pitch = 0.104
            distance = 10.56
            updateCamera()
            return
        }
        previousPointer = convert(event.locationInWindow, from: nil)
        NSCursor.closedHand.push()
    }

    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        guard let previousPointer else {
            self.previousPointer = point
            return
        }
        yaw -= (point.x - previousPointer.x) * 0.0052
        pitch = min(max(pitch + (point.y - previousPointer.y) * 0.0046, -1.05), 1.05)
        self.previousPointer = point
        updateCamera()
    }

    override func mouseUp(with event: NSEvent) {
        previousPointer = nil
        NSCursor.pop()
    }

    override func scrollWheel(with event: NSEvent) {
        let raw = event.scrollingDeltaY
        guard raw.isFinite, raw != 0 else { return }
        let gain: CGFloat = event.hasPreciseScrollingDeltas ? 0.012 : 0.028
        let bounded = min(max(raw, -24), 24)
        distance = min(max(distance * exp(-bounded * gain), 6.2), 18.0)
        updateCamera()
    }

    override func magnify(with event: NSEvent) {
        distance = min(max(distance * exp(-event.magnification * 1.8), 6.2), 18.0)
        updateCamera()
    }

    private func updateCamera() {
        guard let orbitCamera else { return }
        let horizontal = distance * cos(pitch)
        let x = focus.x + horizontal * sin(yaw)
        let y = focus.y + distance * sin(pitch)
        let z = focus.z + horizontal * cos(yaw)
        orbitCamera.position = SCNVector3(x, y, z)
        orbitCamera.look(at: focus, up: SCNVector3(0, 1, 0), localFront: SCNVector3(0, 0, -1))
    }
}

struct MissionSceneView: NSViewRepresentable {
    let active: Bool

    func makeNSView(context: Context) -> SCNView {
        let view = MissionOrbitSCNView(frame: .zero)
        let scene = makeScene()
        view.scene = scene.scene
        view.install(camera: scene.camera)
        view.backgroundColor = NSColor.black
        view.allowsCameraControl = false
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling2X
        view.preferredFramesPerSecond = 18
        view.rendersContinuously = false
        view.loops = true
        view.isPlaying = active
        view.isJitteringEnabled = false
        return view
    }

    func updateNSView(_ nsView: SCNView, context: Context) {
        nsView.isPlaying = active
    }

    private func makeScene() -> (scene: SCNScene, camera: SCNNode) {
        let scene = SCNScene()
        if let url = ResourceLocator.url(named: "webb-carina-cosmic-cliffs-2000", extension: "png"),
           let image = NSImage(contentsOf: url) {
            scene.background.contents = image
        } else {
            scene.background.contents = NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.075, alpha: 1)
        }

        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 42
        camera.camera?.zNear = 0.1
        camera.camera?.zFar = 100
        camera.position = SCNVector3(0, 1.1, 10.5)
        camera.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(camera)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 420
        ambient.light?.color = NSColor(calibratedRed: 0.35, green: 0.48, blue: 0.72, alpha: 1)
        scene.rootNode.addChildNode(ambient)

        let sun = SCNNode()
        sun.light = SCNLight()
        sun.light?.type = .directional
        sun.light?.intensity = 1_850
        sun.light?.castsShadow = false
        sun.eulerAngles = SCNVector3(-0.6, -0.8, 0)
        scene.rootNode.addChildNode(sun)

        let earthSphere = SCNSphere(radius: 2.05)
        earthSphere.segmentCount = 96
        let earth = SCNNode(geometry: earthSphere)
        let earthMaterial = SCNMaterial()
        earthMaterial.lightingModel = .physicallyBased
        earthMaterial.roughness.contents = 0.72
        earthMaterial.metalness.contents = 0.02
        if let url = ResourceLocator.url(named: "earth-blue-marble", extension: "jpg") {
            earthMaterial.diffuse.contents = NSImage(contentsOf: url)
        }
        earth.geometry?.materials = [earthMaterial]
        earth.eulerAngles.z = -0.18
        earth.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 70)))
        scene.rootNode.addChildNode(earth)

        let atmosphereSphere = SCNSphere(radius: 2.11)
        atmosphereSphere.segmentCount = 64
        let atmosphere = SCNNode(geometry: atmosphereSphere)
        let atmosphereMaterial = SCNMaterial()
        atmosphereMaterial.diffuse.contents = NSColor(calibratedRed: 0.23, green: 0.68, blue: 1, alpha: 0.10)
        atmosphereMaterial.emission.contents = NSColor(calibratedRed: 0.08, green: 0.36, blue: 0.8, alpha: 0.13)
        atmosphereMaterial.transparency = 0.32
        atmosphereMaterial.isDoubleSided = true
        atmosphere.geometry?.materials = [atmosphereMaterial]
        scene.rootNode.addChildNode(atmosphere)

        let orbitPlane = makeMissionOrbit()
        scene.rootNode.addChildNode(orbitPlane)

        return (scene, camera)
    }

    private func makeTESSNode() -> SCNNode {
        let root = SCNNode()
        let body = SCNNode(geometry: SCNBox(width: 0.31, height: 0.22, length: 0.27, chamferRadius: 0.025))
        body.geometry?.firstMaterial?.lightingModel = .physicallyBased
        body.geometry?.firstMaterial?.diffuse.contents = NSColor(calibratedRed: 0.63, green: 0.52, blue: 0.34, alpha: 1)
        body.geometry?.firstMaterial?.metalness.contents = 0.48
        body.geometry?.firstMaterial?.roughness.contents = 0.38
        root.addChildNode(body)

        for side: CGFloat in [-1, 1] {
            let panel = SCNNode(geometry: SCNBox(width: 0.58, height: 0.018, length: 0.29, chamferRadius: 0.004))
            panel.position.x = side * 0.46
            panel.geometry?.firstMaterial?.lightingModel = .physicallyBased
            panel.geometry?.firstMaterial?.diffuse.contents = NSColor(calibratedRed: 0.028, green: 0.12, blue: 0.29, alpha: 1)
            panel.geometry?.firstMaterial?.emission.contents = NSColor(calibratedRed: 0.015, green: 0.08, blue: 0.23, alpha: 1)
            panel.geometry?.firstMaterial?.metalness.contents = 0.18
            panel.geometry?.firstMaterial?.roughness.contents = 0.42
            root.addChildNode(panel)

            for stripe in -2...2 {
                let grid = SCNNode(geometry: SCNBox(width: 0.008, height: 0.021, length: 0.275, chamferRadius: 0))
                grid.position = SCNVector3(side * 0.46 + CGFloat(stripe) * 0.092, 0.011, 0)
                grid.geometry?.firstMaterial?.diffuse.contents = NSColor(calibratedRed: 0.24, green: 0.46, blue: 0.66, alpha: 0.85)
                root.addChildNode(grid)
            }
        }

        for x: CGFloat in [-0.083, 0.083] {
            for z: CGFloat in [-0.072, 0.072] {
                let camera = SCNNode(geometry: SCNCylinder(radius: 0.032, height: 0.09))
                camera.position = SCNVector3(x, 0.145, z)
                camera.eulerAngles.x = .pi / 2
                camera.geometry?.firstMaterial?.diffuse.contents = NSColor(calibratedWhite: 0.08, alpha: 1)
                camera.geometry?.firstMaterial?.emission.contents = NSColor(calibratedRed: 0.08, green: 0.28, blue: 0.48, alpha: 0.5)
                root.addChildNode(camera)
            }
        }
        root.scale = SCNVector3(0.88, 0.88, 0.88)
        return root
    }

    /// Native equivalent of the earlier browser mission scene. TESS follows a
    /// visibly eccentric 13.7-day high-Earth-orbit diagram with Earth at the
    /// focus; the geometry is illustrative and intentionally not to scale.
    private func makeMissionOrbit() -> SCNNode {
        let plane = SCNNode()
        plane.eulerAngles = SCNVector3(0.64, 0.08, -0.18)

        let semiMajor: CGFloat = 4.05
        let eccentricity: CGFloat = 0.48
        let semiMinor = semiMajor * sqrt(1 - eccentricity * eccentricity)
        let sampleCount = 300
        var vertices: [SCNVector3] = []
        vertices.reserveCapacity(sampleCount)
        for index in 0..<sampleCount {
            let angle = CGFloat(index) / CGFloat(sampleCount) * .pi * 2
            vertices.append(
                SCNVector3(
                    semiMajor * (cos(angle) - eccentricity),
                    semiMinor * sin(angle),
                    0
                )
            )
        }

        var indices: [Int32] = []
        indices.reserveCapacity(sampleCount * 2)
        for index in 0..<sampleCount {
            indices.append(Int32(index))
            indices.append(Int32((index + 1) % sampleCount))
        }
        let orbitSource = SCNGeometrySource(vertices: vertices)
        let orbitElement = SCNGeometryElement(indices: indices, primitiveType: .line)
        let orbitGeometry = SCNGeometry(sources: [orbitSource], elements: [orbitElement])
        let orbitMaterial = SCNMaterial()
        orbitMaterial.lightingModel = .constant
        orbitMaterial.diffuse.contents = NSColor(calibratedRed: 0.70, green: 0.54, blue: 0.28, alpha: 0.92)
        orbitMaterial.emission.contents = NSColor(calibratedRed: 0.44, green: 0.27, blue: 0.07, alpha: 0.52)
        orbitGeometry.materials = [orbitMaterial]
        let orbitLine = SCNNode(geometry: orbitGeometry)
        plane.addChildNode(orbitLine)

        let apsisLine = makeDiagnosticLine(
            from: SCNVector3(-semiMajor * (1 + eccentricity), 0, 0),
            to: SCNVector3(semiMajor * (1 - eccentricity), 0, 0),
            color: NSColor(calibratedRed: 0.30, green: 0.58, blue: 0.74, alpha: 0.18)
        )
        plane.addChildNode(apsisLine)

        let tess = makeTESSNode()
        plane.addChildNode(tess)

        let fieldOfView = SCNNode(geometry: SCNCone(topRadius: 0.015, bottomRadius: 0.32, height: 1.15))
        fieldOfView.geometry?.firstMaterial?.lightingModel = .constant
        fieldOfView.geometry?.firstMaterial?.diffuse.contents = NSColor(calibratedRed: 0.22, green: 0.64, blue: 0.90, alpha: 0.11)
        fieldOfView.geometry?.firstMaterial?.emission.contents = NSColor(calibratedRed: 0.08, green: 0.34, blue: 0.62, alpha: 0.08)
        fieldOfView.geometry?.firstMaterial?.transparency = 0.20
        fieldOfView.geometry?.firstMaterial?.isDoubleSided = true
        plane.addChildNode(fieldOfView)

        let duration: TimeInterval = 18
        let initialPhase: CGFloat = 0.72
        let action = SCNAction.customAction(duration: duration) { _, elapsed in
            let phase = initialPhase + CGFloat(elapsed / duration) * .pi * 2
            let x = semiMajor * (cos(phase) - eccentricity)
            let y = semiMinor * sin(phase)
            let dx = -semiMajor * sin(phase)
            let dy = semiMinor * cos(phase)
            tess.position = SCNVector3(x, y, 0)
            tess.eulerAngles = SCNVector3(0.08, 0.18, atan2(dy, dx))

            let radialLength = max(sqrt(x * x + y * y), 0.001)
            let radial = SCNVector3(x / radialLength, y / radialLength, 0)
            fieldOfView.position = SCNVector3(
                x + radial.x * 0.62,
                y + radial.y * 0.62,
                0
            )
            fieldOfView.eulerAngles = SCNVector3(0, 0, atan2(radial.y, radial.x) - .pi / 2)
        }
        tess.runAction(.repeatForever(action))
        return plane
    }

    private func makeDiagnosticLine(from start: SCNVector3, to end: SCNVector3, color: NSColor) -> SCNNode {
        let geometry = SCNGeometry(
            sources: [SCNGeometrySource(vertices: [start, end])],
            elements: [SCNGeometryElement(indices: [Int32(0), Int32(1)], primitiveType: .line)]
        )
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = color
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }
}

struct MissionSceneCard: View {
    @EnvironmentObject private var model: AppModel
    @Binding var mode: Int
    let target: TargetRecord?
    let active: Bool

    var body: some View {
        ZStack(alignment: .top) {
            MissionSceneView(active: active)
                .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
            LinearGradient(
                colors: [.black.opacity(0.55), .clear],
                startPoint: .top,
                endPoint: .center
            )
            .frame(height: 110)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))

            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.t("TESS MISSION CONTEXT", "TESS 任务环境"))
                        .font(.nova(size: 9, weight: .bold))
                        .tracking(1.2)
                        .foregroundStyle(.cyan.opacity(0.9))
                    Text(target?.title ?? model.t("No target selected", "未选择目标"))
                        .font(.nova(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(model.t("Drag to orbit · Scroll to zoom", "拖动旋转 · 滚轮缩放"))
                        .font(.nova(size: 10))
                        .foregroundStyle(.white.opacity(0.62))
                }
                Spacer()
                SegmentedControl(
                    $mode,
                    options: [model.t("Orbit", "轨道"), model.t("Light Curve", "光变曲线")],
                    prominent: true
                )
                .frame(width: 188)
            }
            .padding(14)

            VStack {
                Spacer()
                HStack(spacing: 14) {
                    missionLegend(color: Color(red: 0.70, green: 0.54, blue: 0.28), model.t("TESS HIGH-EARTH ORBIT", "TESS 高地球轨道"))
                    missionLegend(color: Color(red: 0.24, green: 0.69, blue: 0.94), model.t("OBSERVING FIELD", "观测视场"))
                    Spacer()
                    Text(model.t("13.7 d resonance · illustrative geometry", "13.7 天共振 · 示意几何"))
                        .font(.nova(size: 8.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.58))
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 11)
            }
            .allowsHitTesting(false)
        }
        .overlay {
            RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous)
                .stroke(NovaTheme.borderStrong.opacity(0.55), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.08), radius: 14, y: 5)
    }

    private func missionLegend(color: Color, _ title: String) -> some View {
        HStack(spacing: 6) {
            Capsule().fill(color).frame(width: 14, height: 2)
            Text(title)
                .font(.nova(size: 8.5, weight: .semibold))
                .tracking(0.35)
                .foregroundStyle(.white.opacity(0.72))
        }
    }
}
